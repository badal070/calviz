"""
Colab-ready RAG + QLoRA notebook script for CalViz.

This file is meant to be used section-by-section inside Google Colab.
It does four jobs:

1. Reads the real CalViz project files so the model learns the app's schema.
2. Builds project-grounded retrieval documents and supervised tuning examples.
3. Fine-tunes a Microsoft Phi-3 instruct model with LoRA adapters.
4. Serves `/ask`, `/payload`, and `/health` so the local CalViz app can
   receive validated visualization payloads over HTTP exactly as described in
   "LLM Integration - Change Documentation.pdf".

Suggested Colab flow:

    from pathlib import Path
    from calviz_phi3_rag_colab import *

    config = NotebookConfig(project_dir="/content/calviz")
    install_colab_packages()
    docs, examples, catalog = prepare_project_assets(config)
    adapter_dir = finetune_phi3_for_calviz(config, examples)
    runtime = load_runtime(config, docs, catalog, adapter_dir=adapter_dir)
    public_url = start_bridge_server(runtime, ngrok_auth_token="YOUR_TOKEN")
    print(public_url)

The local app already expects a strict JSON object with these fields:
`concept_title`, `equation`, `x_range`, `y_range`, `step`, `explanation`.
"""

from __future__ import annotations

import ast
import json
import queue
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are CalViz Tutor, a university-level mathematics tutor for the CalViz desktop visualizer.
Your job is to explain multivariable calculus surfaces and output ONLY one JSON object.

Return exactly this schema:
{
  "concept_title": "<short concept title>",
  "equation": "<f(x,y) in Python/SymPy syntax>",
  "x_range": [<float min>, <float max>],
  "y_range": [<float min>, <float max>],
  "step": <float between 0.05 and 0.5>,
  "explanation": "<plain-text explanation>"
}

Hard rules:
- Output JSON only. No markdown, no code fences, no prose outside the object.
- The local app currently injects payloads into the calculus surface visualizer.
- Therefore `equation` must always be a calculus-style `f(x,y)` expression, not an implicit relation like `x**2 + y**2 = 9`.
- Use only symbols/functions supported by the project parser: x, y, sin, cos, tan, asin, acos, atan, exp, log, ln, sqrt, Abs, sinh, cosh, tanh, ceiling, floor, pi, e, E.
- Use `**` for powers.
- Choose `x_range`, `y_range`, and `step` so the surface is informative and stable in the CalViz plotter.
- `explanation` must clearly cover: definition, geometric intuition, what the partial derivatives mean, and what critical points reveal.
- Keep the explanation educational and specific to the emitted equation.
"""


@dataclass(slots=True)
class NotebookConfig:
    project_dir: str = "/content/calviz"
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    adapter_output_dir: str = "/content/calviz-phi3-adapter"
    max_seq_length: int = 2048
    num_train_epochs: float = 2.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    retrieval_top_k: int = 5
    server_port: int = 8000
    include_pdf_context: bool = True
    random_seed: int = 7
    project_files: list[str] = field(
        default_factory=lambda: [
            "README.md",
            "gui.py",
            "calculus.py",
            "coordinate_geometry.py",
            "llm_bridge.py",
            "viz_bus.py",
            "chat_panel.py",
        ]
    )
    optional_pdfs: list[str] = field(
        default_factory=lambda: ["LLM Integration — Change Documentation.pdf"]
    )

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir)


@dataclass(slots=True)
class KnowledgeDoc:
    doc_id: str
    title: str
    content: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class ConceptPayload:
    concept_title: str
    equation: str
    x_range: list[float]
    y_range: list[float]
    step: float
    explanation: str


@dataclass(slots=True)
class RuntimeBundle:
    config: NotebookConfig
    tokenizer: Any
    model: Any
    retriever: Any
    concept_catalog: list[KnowledgeDoc]
    result_queue: "queue.Queue[dict[str, Any]]"


def install_colab_packages() -> None:
    """
    Install the packages used by this notebook from inside Colab.

    Run this once near the top of the notebook:
        install_colab_packages()
    """
    packages = [
        "torch",
        "transformers>=4.41.2",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "trl",
        "peft",
        "fastapi",
        "uvicorn",
        "pyngrok",
        "nest-asyncio",
        "scikit-learn",
        "sympy",
        "pypdf",
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def _chunk_text(text: str, chunk_size: int = 1100, overlap: int = 200) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _extract_assignment_literal(py_path: Path, variable_name: str) -> Any:
    tree = ast.parse(_read_text(py_path), filename=str(py_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise KeyError(f"Could not locate assignment for {variable_name!r} in {py_path}")


def _extract_supported_symbols(calculus_path: Path) -> list[str]:
    source = _read_text(calculus_path)
    match = re.search(r"SAFE_NAMESPACE\s*=\s*\{(.*?)\n\}", source, flags=re.S)
    if not match:
        return []
    keys = re.findall(r'"([^"]+)":', match.group(1))
    ordered = []
    seen = set()
    for key in keys:
        if key == "__builtins__":
            continue
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _estimate_ranges(equation: str) -> tuple[list[float], list[float], float]:
    eq = equation.replace(" ", "")
    degree = max((int(num) for num in re.findall(r"\*\*(\d+)", eq)), default=1)

    if "exp(" in eq:
        return [-4.0, 4.0], [-4.0, 4.0], 0.12
    if "sqrt(" in eq or "Abs(" in eq:
        return [-6.0, 6.0], [-6.0, 6.0], 0.16
    if any(token in eq for token in ("sin(", "cos(", "tan(", "sinh(", "cosh(", "tanh(")):
        return [-6.0, 6.0], [-6.0, 6.0], 0.18
    if degree >= 4:
        return [-2.5, 2.5], [-2.5, 2.5], 0.08
    if degree == 3:
        return [-3.5, 3.5], [-3.5, 3.5], 0.10
    if degree == 2:
        return [-5.0, 5.0], [-5.0, 5.0], 0.15
    return [-6.0, 6.0], [-6.0, 6.0], 0.20


def _build_symbolic_summary(name: str, equation: str) -> ConceptPayload:
    import sympy as sp

    x, y = sp.symbols("x y", real=True)
    namespace = {
        "x": x,
        "y": y,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "ceiling": sp.ceiling,
        "floor": sp.floor,
        "pi": sp.pi,
        "e": sp.E,
        "E": sp.E,
        "__builtins__": {},
    }

    expr = sp.sympify(eval(equation, namespace))  # noqa: S307
    dfdx = sp.simplify(sp.diff(expr, x))
    dfdy = sp.simplify(sp.diff(expr, y))

    real_critical_points: list[str] = []
    if int(expr.count_ops()) <= 40:
        try:
            raw_points = sp.solve((sp.Eq(dfdx, 0), sp.Eq(dfdy, 0)), (x, y), dict=True)
            for point in raw_points[:5]:
                xv = sp.N(point[x])
                yv = sp.N(point[y])
                if getattr(xv, "is_real", False) and getattr(yv, "is_real", False):
                    real_critical_points.append(f"({float(xv):.3g}, {float(yv):.3g})")
        except Exception:
            pass

    critical_text = (
        f"The gradient vanishes at {', '.join(real_critical_points)}, which marks candidate extrema or saddle behavior."
        if real_critical_points
        else "The gradient should be inspected to locate stationary behavior; depending on the surface, critical points may be isolated, repeated, or absent in the visible window."
    )

    if "gaussian" in name.lower():
        intuition = "The graph forms a smooth bell centered near the origin, with height decaying radially as the distance from the center grows."
    elif "saddle" in name.lower():
        intuition = "The surface bends upward in one principal direction and downward in another, producing the classic saddle geometry."
    elif "ripple" in name.lower():
        intuition = "The geometry oscillates outward from the center in rings, so the surface alternates between crests and troughs."
    elif "wave" in name.lower():
        intuition = "The shape oscillates across the domain, so peaks and valleys repeat in a patterned way."
    else:
        intuition = "The surface can be understood by tracing slices in the x- and y-directions and watching how height changes across the domain."

    explanation = (
        f"{name} is modeled in CalViz by the surface f(x, y) = {equation}. "
        f"This payload is intended for the calculus surface view, so the equation is expressed directly in SymPy/Python syntax.\n\n"
        f"Geometrically, {intuition} The plot window is chosen to show the most informative portion of the surface without overwhelming the visualizer.\n\n"
        f"The partial derivatives are df/dx = {sp.sstr(dfdx)} and df/dy = {sp.sstr(dfdy)}. "
        f"These derivatives describe how the height changes when you move parallel to the x-axis or y-axis while holding the other variable fixed. "
        f"{critical_text}"
    )

    x_range, y_range, step = _estimate_ranges(equation)
    return ConceptPayload(
        concept_title=name,
        equation=equation,
        x_range=x_range,
        y_range=y_range,
        step=step,
        explanation=explanation,
    )


def _make_supervised_examples(
    calculus_presets: list[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[KnowledgeDoc]]:
    examples: list[dict[str, Any]] = []
    concept_docs: list[KnowledgeDoc] = []

    prompt_variants = [
        "Explain the {name} surface and give CalViz the equation payload.",
        "Teach me the idea behind {name} and return a JSON payload for the visualizer.",
        "I want to visualize {name} in CalViz. Explain it and choose a good viewing window.",
    ]

    for index, (name, equation) in enumerate(calculus_presets):
        payload = _build_symbolic_summary(name, equation)
        response_text = json.dumps(asdict(payload), ensure_ascii=True)

        for template in prompt_variants:
            examples.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": template.format(name=name)},
                    ],
                    "completion": [
                        {"role": "assistant", "content": response_text},
                    ],
                }
            )

        concept_docs.append(
            KnowledgeDoc(
                doc_id=f"concept-{index}",
                title=name,
                content=(
                    f"Concept: {name}\n"
                    f"Equation: {payload.equation}\n"
                    f"Suggested x_range: {payload.x_range}\n"
                    f"Suggested y_range: {payload.y_range}\n"
                    f"Suggested step: {payload.step}\n"
                    f"Explanation: {payload.explanation}"
                ),
                metadata={"equation": equation, "payload": asdict(payload)},
            )
        )

    return examples, concept_docs


def prepare_project_assets(
    config: NotebookConfig,
) -> tuple[list[KnowledgeDoc], list[dict[str, Any]], list[KnowledgeDoc]]:
    """
    Build the RAG corpus and the supervised fine-tuning examples from the real
    project files. This is the part that makes the notebook project-specific.
    """
    project_dir = config.project_path
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    docs: list[KnowledgeDoc] = []

    for relative_path in config.project_files:
        full_path = project_dir / relative_path
        if not full_path.exists():
            continue
        text = _read_text(full_path)
        for idx, chunk in enumerate(_chunk_text(text)):
            docs.append(
                KnowledgeDoc(
                    doc_id=f"{relative_path}-{idx}",
                    title=relative_path,
                    content=chunk,
                    metadata={"source": relative_path, "kind": "project_file"},
                )
            )

    if config.include_pdf_context:
        for relative_path in config.optional_pdfs:
            full_path = project_dir / relative_path
            if not full_path.exists():
                continue
            text = _read_pdf(full_path)
            for idx, chunk in enumerate(_chunk_text(text)):
                docs.append(
                    KnowledgeDoc(
                        doc_id=f"{relative_path}-{idx}",
                        title=relative_path,
                        content=chunk,
                        metadata={"source": relative_path, "kind": "pdf"},
                    )
                )

    gui_path = project_dir / "gui.py"
    calculus_path = project_dir / "calculus.py"
    calculus_presets = _extract_assignment_literal(gui_path, "CALCULUS_PRESETS")
    geometry_presets = _extract_assignment_literal(gui_path, "GEOMETRY_PRESETS")
    supported_symbols = _extract_supported_symbols(calculus_path)

    setup_summary = (
        "CalViz integration facts:\n"
        "- The local bridge validates a JSON object with concept_title, equation, x_range, y_range, step, explanation.\n"
        "- gui.py currently forces incoming LLM payloads into the calculus surface mode.\n"
        "- Therefore generated equations must be explicit f(x, y) expressions.\n"
        f"- Supported parser symbols: {', '.join(supported_symbols)}.\n"
        f"- Available calculus presets: {', '.join(name for name, _eq in calculus_presets)}.\n"
        f"- Geometry presets exist in the app, but the current LLM payload path should not emit implicit relations: {', '.join(name for name, _eq in geometry_presets)}."
    )
    docs.append(
        KnowledgeDoc(
            doc_id="calviz-contract",
            title="CalViz LLM Contract",
            content=setup_summary,
            metadata={"kind": "contract"},
        )
    )

    examples, concept_docs = _make_supervised_examples(calculus_presets)
    docs.extend(concept_docs)
    return docs, examples, concept_docs


def finetune_phi3_for_calviz(
    config: NotebookConfig,
    examples: list[dict[str, Any]],
) -> str:
    """
    QLoRA fine-tuning for Phi-3 using project-grounded conversational
    prompt/completion examples. Returns the adapter directory.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(config.random_seed)
    adapter_dir = Path(config.adapter_output_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ready = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ready else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False

    train_dataset = Dataset.from_list(examples)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        max_seq_length=config.max_seq_length,
        gradient_checkpointing=True,
        bf16=bf16_ready,
        fp16=not bf16_ready,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


class TfidfRetriever:
    def __init__(self, docs: list[KnowledgeDoc]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._docs = docs
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([doc.content for doc in docs])
        self._cosine_similarity = cosine_similarity

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeDoc]:
        if not query.strip():
            return self._docs[:top_k]
        query_vec = self._vectorizer.transform([query])
        scores = self._cosine_similarity(query_vec, self._matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        return [self._docs[idx] for idx, _score in ranked[:top_k]]


def load_runtime(
    config: NotebookConfig,
    docs: list[KnowledgeDoc],
    concept_catalog: list[KnowledgeDoc],
    adapter_dir: str | None = None,
) -> RuntimeBundle:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir or config.base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ready = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ready else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model
    model.eval()

    return RuntimeBundle(
        config=config,
        tokenizer=tokenizer,
        model=model,
        retriever=TfidfRetriever(docs),
        concept_catalog=concept_catalog,
        result_queue=queue.Queue(),
    )


def _format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    formatted: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        formatted.append(f"<|{role}|>\n{content}<|end|>")
    formatted.append("<|assistant|>\n")
    return "\n".join(formatted)


def _extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("Model output did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("Could not isolate a complete JSON object.")


def _coerce_range(value: Any, fallback: list[float]) -> list[float]:
    try:
        numbers = [float(item) for item in value]
    except Exception:
        return fallback
    if len(numbers) != 2 or numbers[0] >= numbers[1]:
        return fallback
    return numbers


def _best_catalog_fallback(
    user_text: str,
    concept_catalog: list[KnowledgeDoc],
) -> dict[str, Any] | None:
    tokens = set(re.findall(r"[a-z0-9]+", user_text.lower()))
    best_score = -1
    best_payload = None

    for doc in concept_catalog:
        title_tokens = set(re.findall(r"[a-z0-9]+", doc.title.lower()))
        score = len(tokens & title_tokens)
        if score > best_score:
            best_score = score
            best_payload = doc.metadata.get("payload")

    if best_score <= 0:
        return concept_catalog[0].metadata.get("payload") if concept_catalog else None
    return best_payload


def normalize_payload(
    raw_payload: dict[str, Any],
    user_text: str,
    concept_catalog: list[KnowledgeDoc],
) -> dict[str, Any]:
    fallback = _best_catalog_fallback(user_text, concept_catalog) or {
        "concept_title": "CalViz Surface",
        "equation": "x**2 + y**2",
        "x_range": [-5.0, 5.0],
        "y_range": [-5.0, 5.0],
        "step": 0.15,
        "explanation": "This fallback surface keeps the payload valid for the CalViz calculus visualizer.",
    }

    equation = str(raw_payload.get("equation") or fallback["equation"]).strip()
    if "=" in equation or not equation:
        equation = fallback["equation"]

    x_range, y_range, estimated_step = _estimate_ranges(equation)
    payload = {
        "concept_title": str(raw_payload.get("concept_title") or fallback["concept_title"]).strip(),
        "equation": equation,
        "x_range": _coerce_range(raw_payload.get("x_range"), x_range),
        "y_range": _coerce_range(raw_payload.get("y_range"), y_range),
        "step": float(raw_payload.get("step") or estimated_step),
        "explanation": str(raw_payload.get("explanation") or fallback["explanation"]).strip(),
    }

    if not (0.05 <= payload["step"] <= 0.5):
        payload["step"] = estimated_step
    if not payload["concept_title"]:
        payload["concept_title"] = fallback["concept_title"]
    if not payload["explanation"]:
        payload["explanation"] = fallback["explanation"]
    return payload


def generate_visualization_payload(runtime: RuntimeBundle, user_text: str) -> dict[str, Any]:
    import torch

    retrieved_docs = runtime.retriever.search(user_text, top_k=runtime.config.retrieval_top_k)
    context = "\n\n".join(
        f"[{doc.title}] {doc.content}" for doc in retrieved_docs
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Project context for grounding:\n"
                f"{context}\n\n"
                f"User request:\n{user_text}"
            ),
        },
    ]

    prompt = _format_messages(runtime.tokenizer, messages)
    inputs = runtime.tokenizer(prompt, return_tensors="pt").to(runtime.model.device)

    with torch.no_grad():
        output_ids = runtime.model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=False,
            eos_token_id=runtime.tokenizer.eos_token_id,
            pad_token_id=runtime.tokenizer.pad_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    raw_output = runtime.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    json_text = _extract_json_block(raw_output)
    raw_payload = json.loads(json_text)
    return normalize_payload(raw_payload, user_text, runtime.concept_catalog)


def start_bridge_server(
    runtime: RuntimeBundle,
    ngrok_auth_token: str,
) -> str:
    """
    Starts the FastAPI bridge expected by `llm_bridge.py`.
    Returns the public ngrok URL.
    """
    import nest_asyncio
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from pyngrok import ngrok

    nest_asyncio.apply()

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class AskRequest(BaseModel):
        text: str

    @app.post("/ask")
    async def ask_endpoint(req: AskRequest) -> dict[str, str]:
        def _worker() -> None:
            try:
                payload = generate_visualization_payload(runtime, req.text)
                runtime.result_queue.put({"status": "ok", "data": payload})
            except Exception as exc:
                runtime.result_queue.put({"status": "error", "error": str(exc)})

        threading.Thread(target=_worker, daemon=True).start()
        return {"status": "queued"}

    @app.get("/payload")
    async def payload_endpoint() -> dict[str, Any]:
        try:
            return runtime.result_queue.get_nowait()
        except queue.Empty:
            return {"status": "pending"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "alive"}

    ngrok.set_auth_token(ngrok_auth_token)
    tunnel = ngrok.connect(runtime.config.server_port)
    public_url = tunnel.public_url
    print("=" * 72)
    print(f"CalViz Colab bridge live at: {public_url}")
    print(f"POST {public_url}/ask")
    print(f"GET  {public_url}/payload")
    print(f"GET  {public_url}/health")
    print("=" * 72)

    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=runtime.config.server_port),
        daemon=True,
    )
    server.start()
    return public_url


def notebook_quickstart() -> str:
    return """
Cell 1:
    config = NotebookConfig(project_dir="/content/calviz")
    install_colab_packages()

Cell 2:
    docs, examples, catalog = prepare_project_assets(config)
    print(len(docs), len(examples), len(catalog))

Cell 3:
    adapter_dir = finetune_phi3_for_calviz(config, examples)
    print(adapter_dir)

Cell 4:
    runtime = load_runtime(config, docs, catalog, adapter_dir=adapter_dir)
    PUBLIC_URL = start_bridge_server(runtime, ngrok_auth_token="YOUR_NGROK_TOKEN")
    print(PUBLIC_URL)

Cell 5:
    sample = generate_visualization_payload(runtime, "Explain the Gaussian bell surface.")
    print(json.dumps(sample, indent=2))
"""


if __name__ == "__main__":
    print(notebook_quickstart())
