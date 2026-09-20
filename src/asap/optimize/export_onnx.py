import argparse
import logging
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from asap.config import Settings, get_settings

logger = logging.getLogger(__name__)

DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "sequence"},
    "attention_mask": {0: "batch", 1: "sequence"},
    "logits": {0: "batch"},
}


def export_onnx(
    src_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    
    cfg = settings or get_settings()
    src = Path(src_dir) if src_dir is not None else cfg.paths.cls_dir
    out = Path(out_dir) if out_dir is not None else cfg.paths.onnx_dir

    if not src.is_dir():
        raise FileNotFoundError(f"no classifier checkpoint at {src}")

    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(src))
    model = AutoModelForSequenceClassification.from_pretrained(str(src))
    model.eval()
    
    dummy = tokenizer(
        ["المنتج رائع", "خدمة سيئة جدا"],
        padding=True,
        truncation=True,
        max_length=cfg.inference.max_length,
        return_tensors="pt",
    )
    inputs = (dummy["input_ids"], dummy["attention_mask"])

    graph = out / "model.onnx"
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings(
            "ignore", message=".*aten::index.*", category=UserWarning
        )
        torch.onnx.export(
            model,
            inputs,
            str(graph),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=DYNAMIC_AXES,
            opset_version=17,
            do_constant_folding=True,
        )

    tokenizer.save_pretrained(str(out))
    model.config.save_pretrained(str(out))

    logger.info("exported %s -> %s (%.1f MB)", src, graph, graph.stat().st_size / 1e6)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=None, help="torch checkpoint dir")
    parser.add_argument("--out", type=Path, default=None, help="onnx output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(export_onnx(args.src, args.out))


if __name__ == "__main__":
    main()
