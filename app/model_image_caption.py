# app/model_image_caption.py
from __future__ import annotations

from PIL import Image
from transformers import pipeline

from .models_base import ModelBase, ModelInfoMixin


class ImageCaptioner(ModelInfoMixin, ModelBase):
    """
    ViT-GPT2 image captioning using the HF 'image-to-text' pipeline.
    Compatible with older transformers that don't support num_return_sequences.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ViT-GPT2 Image Captioning",
            category="Vision",
            task="image-captioning",
            description="Generates captions for images using ViT encoder + GPT-2 decoder.",
        )
        # Create the pipeline once
        self._pipe = pipeline(
            task="image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
        )

    # ------------- required abstract methods -------------

    def preprocess(self, image_path: str):
        """Open image and return a PIL.Image"""
        img = Image.open(image_path).convert("RGB")
        return img

    def _infer(self, pil_image):
        """
        Run the pipeline. Don't pass num_return_sequences (not supported
        in some versions). Keep it simple and fast.
        """
        # Common args that are widely supported:
        outputs = self._pipe(pil_image, max_new_tokens=20)
        return outputs

    def postprocess(self, raw_outputs):
        """
        Normalize pipeline outputs into a list[str] of captions.
        HF may return:
          - [{'generated_text': '...'}]
          - {'generated_text': '...'}
        """
        captions: list[str] = []

        if isinstance(raw_outputs, list):
            for item in raw_outputs:
                if isinstance(item, dict) and "generated_text" in item:
                    captions.append(item["generated_text"])
                else:
                    captions.append(str(item))
        elif isinstance(raw_outputs, dict) and "generated_text" in raw_outputs:
            captions.append(raw_outputs["generated_text"])
        else:
            captions.append(str(raw_outputs))

        # Return the normalized result the GUI expects
        return captions
