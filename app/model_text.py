# app/model_text.py

from __future__ import annotations

from typing import Any, Dict, List

from transformers import pipeline

from .models_base import ModelBase, ModelInfoMixin


DEFAULT_LABELS = ["sports", "entertainment", "business", "technology", "politics"]


class TextClassifier(ModelInfoMixin, ModelBase):
    """
    Zero-shot text classifier using facebook/bart-large-mnli.
    Implements the ModelBase abstract hooks: preprocess, _infer, postprocess.
    """

    def __init__(
        self,
        candidate_labels: List[str] | None = None,
    ) -> None:
        # metadata for the GUI
        super().__init__(
            name="BART-MNLI Zero-shot",
            category="Text",
            task="zero-shot-classification",
            description="Classify arbitrary text into labels provided at runtime.",
        )

        # store labels
        self._labels = candidate_labels or DEFAULT_LABELS

        # build HF pipeline (CPU by default: device=-1)
        try:
            self._pipe = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
        except Exception as e:
            # bubble a clear error up to the GUI
            raise RuntimeError(f"Failed to load zero-shot pipeline: {e}") from e

    # required ModelBase hooks

    def preprocess(self, text: str) -> str:
        """Basic cleaning/validation for input text."""
        if not isinstance(text, str):
            raise TypeError("TextClassifier.preprocess expected a string")
        return text.strip()

    def _infer(self, cleaned_text: str) -> Any:
        """Run the HF pipeline and return its raw result."""
        # HF returns a dict with 'labels' and 'scores'
        return self._pipe(cleaned_text, candidate_labels=self._labels)

    def postprocess(self, raw_result: Any) -> List[Dict[str, float]]:
        """
        Normalize the HF output into a list of {label, score} dicts,
        sorted by score desc. This format is what the GUI expects.
        """
        # Case 1: regular HF dict
        if isinstance(raw_result, dict) and "labels" in raw_result and "scores" in raw_result:
            pairs = list(zip(raw_result["labels"], raw_result["scores"]))
            out = [{"label": lbl, "score": float(scr)} for lbl, scr in pairs]
            out.sort(key=lambda x: x["score"], reverse=True)
            return out

        # Case 2: already a list of dicts
        if isinstance(raw_result, list) and raw_result and isinstance(raw_result[0], dict):
            # label/score keys exist and sort
            cleaned = [
                {"label": d.get("label", "-"), "score": float(d.get("score", 0.0))}
                for d in raw_result
            ]
            cleaned.sort(key=lambda x: x["score"], reverse=True)
            return cleaned

        # Fallback:
        return [{"label": "output", "score": 0.0}, {"label": str(raw_result), "score": 0.0}]
