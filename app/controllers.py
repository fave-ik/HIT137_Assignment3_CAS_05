# app/controllers.py
from __future__ import annotations

from .model_text import TextClassifier
from .model_image_caption import ImageCaptioner


class Controller:
    def __init__(self) -> None:
        self.text_model = None
        self.image_model = None
        self._import_error = None

        # Load text model (zero-shot)
        try:
            self.text_model = TextClassifier()
        except Exception as e:
            self._import_error = e

        # Load image model (captioning)
        try:
            self.image_model = ImageCaptioner()
        except Exception as e:
            # keep first error if it exists; otherwise store this one
            if self._import_error is None:
                self._import_error = e

    # ---------- Run methods ----------
    def run_text(self, text: str):
        if not self.text_model:
            raise RuntimeError(f"Failed to load TextClassifier: {self._import_error}")
        return self.text_model.run(text)

    def run_image_caption(self, image_path: str):
        if not self.image_model:
            raise RuntimeError(f"Failed to load ImageCaptioner: {self._import_error}")
        return self.image_model.run(image_path)

    # ---------- Metadata for GUI panel ----------
    def model_info(self, task_name: str) -> dict:
        """Return model metadata for the 'Selected Model Info' panel."""
        if "Zero-shot" in task_name:
            return {
                "name": "BART-MNLI Zero-shot",
                "category": "Text",
                "description": "Classify arbitrary text into labels provided at runtime.",
            }
        elif "Caption" in task_name:
            return {
                "name": "ViT-GPT2 Image Captioning",
                "category": "Vision",
                "description": "Generates captions for images using ViT encoder + GPT-2 decoder.",
            }
        else:
            return {"name": "-", "category": "-", "description": "-"}
