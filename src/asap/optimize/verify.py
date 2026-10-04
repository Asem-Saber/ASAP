from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _onnx_logits_fn(onnx_dir: Path, model_file: str, max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    session = ort.InferenceSession(
        str(onnx_dir / model_file), providers=["CPUExecutionProvider"]
    )

    names = [i.name for i in session.get_inputs()]

    def run(batch: list[str]) -> np.ndarray:
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = {n: np.asarray(enc[n], dtype=np.int64) for n in names}
        return session.run(None, feed)[0]

    return run


def padding_consistency(
    onnx_dir: Path,
    texts: list[str],
    *,
    max_length: int,
    model_file: str = "model.onnx",
) -> float:
    
    if not texts:
        raise ValueError("need at least one text to compare")

    onnx_logits = _onnx_logits_fn(onnx_dir, model_file, max_length)
    alone = np.vstack([onnx_logits([t]) for t in texts])
    batched = onnx_logits(texts)
    return float(np.abs(alone - batched).max())


def graphs_match(
    onnx_dir: Path,
    texts: list[str],
    *,
    max_length: int,
    baseline_file: str = "model.onnx",
    candidate_file: str = "model_quantized.onnx",
) -> dict[str, float]:
    if not texts:
        raise ValueError("need at least one text to compare")

    baseline = _onnx_logits_fn(onnx_dir, baseline_file, max_length)
    candidate = _onnx_logits_fn(onnx_dir, candidate_file, max_length)

    reference = np.vstack([baseline([t]) for t in texts])
    derived = np.vstack([candidate([t]) for t in texts])

    return {
        "max_abs_delta": float(np.abs(derived - reference).max()),
        "label_agreement": float((derived.argmax(-1) == reference.argmax(-1)).mean()),
    }


def logits_match(
    torch_dir: Path,
    onnx_dir: Path,
    texts: list[str],
    *,
    max_length: int,
    model_file: str = "model.onnx",
) -> dict[str, float]:
    if not texts:
        raise ValueError("need at least one text to compare")

    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    session = ort.InferenceSession(
        str(onnx_dir / model_file), providers=["CPUExecutionProvider"]
    )
    model = AutoModelForSequenceClassification.from_pretrained(str(torch_dir)).eval()
    names = [i.name for i in session.get_inputs()]

    def onnx_logits(batch: list[str]) -> np.ndarray:
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = {n: np.asarray(enc[n], dtype=np.int64) for n in names}
        return session.run(None, feed)[0]

    def torch_logits(batch: list[str]) -> np.ndarray:
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            return model(**enc).logits.numpy()

    alone = np.vstack([onnx_logits([t]) for t in texts])
    batched = onnx_logits(texts)
    reference = np.vstack([torch_logits([t]) for t in texts])

    return {
        "max_abs_delta": float(np.abs(alone - reference).max()),
        "padding_delta": float(np.abs(alone - batched).max()),
        "label_agreement": float((alone.argmax(-1) == reference.argmax(-1)).mean()),
    }
