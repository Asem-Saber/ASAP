from typing import Protocol, TypedDict, runtime_checkable


class SentimentResult(TypedDict):
    """A single classification result."""
    sentiment: str
    confidence: float


@runtime_checkable
class SentimentModel(Protocol):
    def predict(self, text: str) -> SentimentResult:
        """Classify one raw text."""
        ...

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Classify many raw texts at once, preserving input order."""
        ...

    def warmup(self) -> None:
        """Run one throwaway inference so the first real request is not
        charged for lazy CUDA context creation and kernel autotuning."""
        ...