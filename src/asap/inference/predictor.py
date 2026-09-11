import logging
from pathlib import Path
import torch
from transformers import pipeline
from asap.config import Settings, get_settings
from asap.inference.base import SentimentResult
from asap.preprocessing import normalize

logger= logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("cuda requested but unavailable, falling back to cpu")
        return "cpu"
    return device


def _resolve_dtype(dtype: str, device: str) -> str:
    if dtype == "float16" and device == "cpu":
        logger.warning("float16 requested on cpu, falling back to float32")
        return "float32"
    return dtype


class SentimentModel:
    def __init__(
        self,
        model_path: Path | None = None,
        device: str | None = None,
        max_length: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.cfg= settings or get_settings()
        inf= self.cfg.inference

        self.model_path= model_path or inf.model_dir
        self.device= _resolve_device(device or inf.device)
        self.dtype= _resolve_dtype(inf.dtype, self.device)
        self.max_length= max_length or inf.max_length
        self.batch_size= inf.batch_size
        self.truncate= inf.truncate

        self.classifier= pipeline(
            "text-classification",
            model= str(self.model_path),
            tokenizer= str(self.model_path),
            device= self.device,
            dtype= self.dtype,
            batch_size= self.batch_size,
            max_length= self.max_length,
            truncation= self.truncate
        )

    def _to_label(self, label: str) -> str:
        if label.startswith("LABEL_"):
            return self.cfg.data.labels[int(label.removeprefix("LABEL_"))]
        return label

    def warmup(self) -> None:
        self.predict("المنتج رائع")

    def predict(self, text: str) -> SentimentResult:
        clean_text= normalize(text)
        result= self.classifier(clean_text)[0]

        return {
            "sentiment": self._to_label(result["label"]),
            "confidence": result["score"]
        }

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []

        cleaned_texts= [normalize(text) for text in texts]
        results= self.classifier(cleaned_texts)

        return [
            {
                "sentiment": self._to_label(result["label"]),
                "confidence": result["score"]
            }
            for result in results
        ]


sentiment_model: SentimentModel | None = None

def load_model(settings: Settings | None = None) -> SentimentModel:
    global sentiment_model
    if sentiment_model is None:
        cfg = settings or get_settings()
        sentiment_model = SentimentModel(settings=cfg)
        sentiment_model.warmup()
    return sentiment_model

def reset_model() -> None:
    global sentiment_model
    sentiment_model = None