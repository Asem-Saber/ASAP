from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from asap.config import Settings, get_settings

logger = logging.getLogger(__name__)

QUANTIZED_FILENAME = "model_quantized.onnx"

MIN_LABEL_AGREEMENT = 0.99
MAX_F1_DROP = 0.005


def quantize_onnx(
    src: Path | None = None,
    out: Path | None = None,
    *,
    settings: Settings | None = None,
) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    cfg = settings or get_settings()
    src = Path(src) if src is not None else cfg.inference.model_dir / cfg.inference.model_file
    out = Path(out) if out is not None else src.parent / QUANTIZED_FILENAME

    if not src.is_file():
        raise FileNotFoundError(f"no ONNX graph at {src}. Run asap-export-onnx first.")

    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8)

    before, after = src.stat().st_size, out.stat().st_size
    logger.info(
        "quantized %s -> %s (%.1f MB -> %.1f MB, %.2fx smaller)",
        src.name, out.name, before / 1e6, after / 1e6, before / after,
    )
    return out


def f1_on_test_split(
    model_file: str,
    *,
    sample_rows: int | None = None,
    settings: Settings | None = None,
) -> float:
    import pandas as pd
    from sklearn.metrics import f1_score

    from asap.inference.predictor import OnnxPredictor

    cfg = settings or get_settings()
    split = cfg.paths.processed_dir / "test.parquet"
    if not split.is_file():
        raise FileNotFoundError(
            f"no test split at {split}; build it with `python -m asap.data.build`"
        )

    frame = pd.read_parquet(split)
    if sample_rows is not None and sample_rows < len(frame):
        frame = frame.sample(n=sample_rows, random_state=cfg.project["seed"])

    labels = {name: idx for idx, name in cfg.data.labels.items()}
    texts = frame["text"].tolist()

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    predictor = OnnxPredictor(model_file=model_file, settings=cfg)
    sorted_predictions = predictor.predict_batch([texts[i] for i in order])

    predicted = [0] * len(texts)
    for position, index in enumerate(order):
        predicted[index] = labels[sorted_predictions[position]["sentiment"]]

    return float(f1_score(frame["label"].tolist(), predicted, average="weighted"))


def check_quantized(
    *,
    sample_size: int = 256,
    settings: Settings | None = None,
) -> dict[str, float | bool]:
    from asap.optimize.verify import graphs_match

    cfg = settings or get_settings()
    fixture = cfg.paths.bench_sample
    if not fixture.is_file():
        raise FileNotFoundError(
            f"no sample at {fixture}; build it with `python scripts/sample_data.py`"
        )

    with fixture.open(encoding="utf-8") as handle:
        texts = [json.loads(line)["text"] for line in handle if line.strip()]
    texts = texts[:sample_size]

    agreement = graphs_match(
        cfg.inference.model_dir,
        texts,
        max_length=cfg.inference.max_length,
        baseline_file=cfg.inference.model_file,
        candidate_file=QUANTIZED_FILENAME,
    )

    f1_fp32 = f1_on_test_split(cfg.inference.model_file, settings=cfg)
    f1_int8 = f1_on_test_split(QUANTIZED_FILENAME, settings=cfg)
    drop = f1_fp32 - f1_int8

    return {
        "label_agreement": agreement["label_agreement"],
        "max_abs_delta": agreement["max_abs_delta"],
        "f1_fp32": f1_fp32,
        "f1_int8": f1_int8,
        "f1_drop": drop,
        "agreement_ok": agreement["label_agreement"] >= MIN_LABEL_AGREEMENT,
        "f1_ok": drop <= MAX_F1_DROP,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=None, help="fp32 graph to quantize")
    parser.add_argument("--out", type=Path, default=None, help="INT8 graph to write")
    parser.add_argument(
        "--check",
        action="store_true",
        help="measure the result against the fp32 graph and apply the gate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="texts used for the label-agreement check",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    graph = quantize_onnx(args.src, args.out)
    print(graph)

    if not args.check:
        return

    result = check_quantized(sample_size=args.sample_size)
    print(
        f"label agreement : {result['label_agreement']:.4f}"
        f"  (gate >= {MIN_LABEL_AGREEMENT})"
    )
    print(f"max abs delta   : {result['max_abs_delta']:.4f}")
    print(f"test_f1 fp32    : {result['f1_fp32']:.4f}")
    print(f"test_f1 int8    : {result['f1_int8']:.4f}")
    print(f"f1 drop         : {result['f1_drop']:+.4f}  (gate <= {MAX_F1_DROP})")

    if result["agreement_ok"] and result["f1_ok"]:
        print("GATE PASSED")
        return

    print("GATE FAILED")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
