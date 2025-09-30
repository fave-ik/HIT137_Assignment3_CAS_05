# app/model_text.py
from __future__ import annotations
import time
from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin

class TextClassifier(ModelInfoMixin, ModelBase):
    def __init__(self):
        super().__init__(
            name="BART-MNLI Zero-shot",
            category="Text",
            task="zero-shot-classification",
            description="Classify arbitrary text into labels provided at runtime.",
        )
        self._pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def run(self, text: str):
        t0 = time.time()

        # your label set (adjust if you like)
        candidate_labels = ["sports", "entertainment", "business", "technology", "politics"]

        out = self._pipe(text, candidate_labels=candidate_labels)

        # Normalize Hugging Face outputs to a list of {label, score}
        def to_list_of_dicts(d):
            return [{"label": lbl, "score": float(scr)} for lbl, scr in zip(d["labels"], d["scores"])]

        if isinstance(out, dict):
            result = to_list_of_dicts(out)
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            result = to_list_of_dicts(out[0])   # batched case
        else:
            result = out

        elapsed_ms = int((time.time() - t0) * 1000)
        return {"result": result, "elapsed_ms": elapsed_ms}
