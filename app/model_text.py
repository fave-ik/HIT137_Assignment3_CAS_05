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
            task="zero-shot-classification",
            description="Classify arbitrary text into labels provided at runtime."
        )
        # CPU by default; remove device_map=None if you have GPU configured
        self._pipe = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL, device_map=None)

        # Default labels if the caller doesn't supply any
        self._default_labels = ["sports", "politics", "technology", "entertainment", "business"]

    # ----------------- ModelBase hook methods -----------------

    def preprocess(self, x):
        """
        Accept either:
          - text string  -> use default labels
          - (text, labels_list) tuple -> use caller-provided labels
        """
        if isinstance(x, tuple) and len(x) == 2:
            text, labels = x
        else:
            text = x
            labels = self._default_labels
        return {"text": text, "labels": labels}

    def _infer(self, items):
        return self._pipe(items["text"], candidate_labels=items["labels"])
        # HF returns a dict with keys: 'sequence', 'labels', 'scores'

    def postprocess(self, raw):
        # Return the HF dict as-is; the GUI knows how to display 'labels' + 'scores'
        return raw

    # ----------------- public API -----------------

    @timed
    @log_call
    def run(self, x):
        """
        Return a dict: {"result": <HF dict>, "elapsed_ms": <int>}
        ModelBase decorators @timed and @log_call will time and log this call.
        """
        items = self.preprocess(x)
        raw = self._infer(items)
        result = self.postprocess(raw)
        return result
