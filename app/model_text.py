from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin, timed, log_call

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"


class TextClassifier(ModelInfoMixin, ModelBase):
    """
    Zero-shot text classification using facebook/bart-large-mnli.
    If you only type text, we classify it against a default label set.
    """

    def __init__(self):
        super().__init__(
            name="BART-MNLI Zero-shot",
            category="Text",                      # <-- REQUIRED by your ModelBase
            task="zero-shot-classification",
            description="Classify arbitrary text into labels provided at runtime."
        )
        self._pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device_map=None)
        self._default_labels = ["sports", "politics", "technology", "entertainment", "business"]

    def preprocess(self, x):
        # Accept either a plain string or (text, labels) tuple
        if isinstance(x, tuple) and len(x) == 2:
            text, labels = x
        else:
            text = x
            labels = self._default_labels
        return {"text": text, "labels": labels}

    def _infer(self, items):
        return self._pipe(items["text"], candidate_labels=items["labels"])

    def postprocess(self, raw):
        # Return HF dict with 'labels' and 'scores'; GUI knows how to display it
        return raw

    @timed
    @log_call
    def run(self, x):
        items = self.preprocess(x)
        raw = self._infer(items)
        return raw
