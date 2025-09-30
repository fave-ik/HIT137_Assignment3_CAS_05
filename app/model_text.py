from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin, timed, log_call

def _parse_labels(text: str):
    # Allow a line like: labels: sports, tech, politics
    lines = [ln.strip() for ln in text.splitlines()]
    for ln in reversed(lines):
        if ln.lower().startswith("labels:"):
            raw = ln.split(":", 1)[1]
            return [x.strip() for x in raw.split(",") if x.strip()]
    # fallback labels if none provided
    return ["sports", "tech", "politics", "finance"]

class TextClassifier(ModelInfoMixin, ModelBase):
    """Zero-shot text classifier using BART-MNLI."""
    def __init__(self, model_id: str = "facebook/bart-large-mnli"):
        super().__init__(
            name="BART-MNLI Zero-shot",
            task="zero-shot-classification",
            description="Classify arbitrary text into labels provided at runtime."
        )
        self._pipe = pipeline("zero-shot-classification", model=model_id)

    @timed
    @log_call
    def run(self, text: str):
        labels = _parse_labels(text)
        # remove labels: line from the premise
        premise = "\n".join(
            ln for ln in text.splitlines() if not ln.lower().startswith("labels:")
        ).strip() or text.strip()
        out = self._pipe(premise, candidate_labels=labels, multi_label=False)
        result = [{"label": l, "score": float(s)} for l, s in zip(out["labels"], out["scores"])]
        return {"result": result}
