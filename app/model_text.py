# app/model_text.py
from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin

class TextClassifier(ModelInfoMixin, ModelBase):
    """
    Zero-shot text classification (facebook/bart-large-mnli).
    Pass {"text": "...", "labels": ["label1", "label2", ...]} to run().
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
        # x is a dict: {"text": str, "labels": List[str]}
        return x

    def _infer(self, x):
        return self._pipe(x["text"], candidate_labels=x["labels"])

    def postprocess(self, y):
        # Keep top-5 label/score pairs for the GUI
        labels = y["labels"]
        scores = y["scores"]
        return [{"label": lbl, "score": float(scr)} for lbl, scr in zip(labels, scores)][:5]
