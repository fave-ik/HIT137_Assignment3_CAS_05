# app/model_text.py
from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin

class TextClassifier(ModelInfoMixin, ModelBase):
    """
    Zero-shot text classification using facebook/bart-large-mnli.
    Call run({"text": "...", "labels": ["a","b","c"]})
    """
    def __init__(self):
        super().__init__(
            name="BART-MNLI Zero-shot",
            task="zero-shot-classification",
            category="Text",
            description="Classify arbitrary text into labels provided at runtime."
        )
        self._pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def preprocess(self, x):
        # x is {"text": str, "labels": List[str]}
        return x

    def _infer(self, x):
        return self._pipe(x["text"], candidate_labels=x["labels"])

    def postprocess(self, y):
        # Convert to a compact list of dicts (label, score)
        labels = y["labels"]
        scores = y["scores"]
        return [{"label": lbl, "score": float(scr)} for lbl, scr in zip(labels, scores)][:5]
