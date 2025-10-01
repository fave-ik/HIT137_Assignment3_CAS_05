# app/model_image_caption.py
from __future__ import annotations

from PIL import Image
from transformers import pipeline

from .models_base import ModelBase


class ImageCaptioner(ModelBase):
    """
    HF image-to-text pipeline (ViT-GPT2). Returns a list of captions.
    Output format matches the GUI contract:
        {"result": ["caption 1", "caption 2", ...], "elapsed_ms": <int>}
    """

    def __init__(self) -> None:
        super().__init__(
            name="ViT-GPT2 Image Captioning",
            task="image-to-text",
            category="Vision",
            description="Generates captions for images using ViT encoder + GPT-2 decoder.",
        )
        # Lazy / simple HF pipeline
        self._pipe = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    # ---- Required abstract methods from ModelBase ----
    def preprocess(self, image_path: str) -> Image.Image:
        # PIL image (RGB) works directly with the pipeline
        return Image.open(image_path).convert("RGB")

    def _infer(self, img: Image.Image):
        # Return 1–3 candidate captions
        return self._pipe(img, max_new_tokens=32, num_return_sequences=3)

    def postprocess(self, raw) -> list[str]:
        # raw looks like: [{"generated_text": "a brown dog..."}, ...]
        if not raw:
            return []
        return [r.get("generated_text", "").strip() for r in raw if r.get("generated_text")]
