from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin, timed, log_call

class ImageCaptioner(ModelInfoMixin, ModelBase):
    """Image → text captioning (Vision–Language)."""
    def __init__(self, model_id: str = "nlpconnect/vit-gpt2-image-captioning"):
        super().__init__(
            name="ViT-GPT2 Image Captioner",
            task="image-to-text",
            description="Generates a short natural-language caption for an image."
        )
        self._pipe = pipeline("image-to-text", model=model_id)

    @timed
    @log_call
    def run(self, image_path: str):
        out = self._pipe(image_path)      # [{'generated_text': '...'}, ...]
        captions = [d.get("generated_text", str(d)) for d in out]
        return {"result": captions}
