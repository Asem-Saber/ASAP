import logging
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from asap.config import Settings, get_settings
from asap.inference.base import SentimentResult
from asap.preprocessing import normalize

logger= logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxPredictor:
    def __init__(
        self,
        model_dir: Path | None = None,
        model_file: str | None = None,
        max_length: int | None = None,
        provider: str | None = None,
        intra_op_num_threads: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cfg= settings or get_settings()
        inf= self.cfg.inference

        self.model_dir= Path(model_dir) if model_dir is not None else Path(inf.model_dir)
        self.model_file= model_file or inf.model_file
        self.max_length= max_length or inf.max_length
        self.batch_size= inf.batch_size
        self.truncate= inf.truncate

        graph = self.model_dir / self.model_file
        if not graph.is_file():
            raise FileNotFoundError(
                f"no ONNX graph at {graph}. Run asap-export-onnx first."
            )

        self.tokenizer= AutoTokenizer.from_pretrained(str(self.model_dir))

        threads= (
            intra_op_num_threads
            if intra_op_num_threads is not None
            else inf.intra_op_num_threads
        )
        options= ort.SessionOptions()
        if threads:
            options.intra_op_num_threads= threads
        self.intra_op_num_threads= threads

        self.provider= provider or inf.provider
        self.session= ort.InferenceSession(
            str(graph), options, providers=[self.provider]
        )
        self.providers= self.session.get_providers()

        self._input_names= [i.name for i in self.session.get_inputs()]

    def _run(self, texts: list[str]) -> list[SentimentResult]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=self.truncate,
            max_length=self.max_length,
            return_tensors="np",
        )
        feed = {n: np.asarray(enc[n], dtype=np.int64) for n in self._input_names}
        probs = _softmax(self.session.run(None, feed)[0])
        idxs = probs.argmax(axis=-1)

        return [
            {
                "sentiment": self.cfg.data.labels[int(idx)],
                "confidence": float(probs[row, idx])
            }
            for row, idx in enumerate(idxs)
        ]

    def warmup(self) -> None:
        self.predict("المنتج رائع")

    def predict(self, text: str) -> SentimentResult:
        return self._run([normalize(text)])[0]

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []

        cleaned_texts= [normalize(text) for text in texts]
        results: list[SentimentResult] = []
        for start in range(0, len(cleaned_texts), self.batch_size):
            results.extend(self._run(cleaned_texts[start : start + self.batch_size]))
        return results


sentiment_model: OnnxPredictor | None = None

def load_model(settings: Settings | None = None) -> OnnxPredictor:
    global sentiment_model
    if sentiment_model is None:
        cfg = settings or get_settings()
        sentiment_model = OnnxPredictor(settings=cfg)
        sentiment_model.warmup()
    return sentiment_model

def reset_model() -> None:
    global sentiment_model
    sentiment_model = None