try:
    from .model_text import TextClassifier
except Exception as e:
    TextClassifier = None
    _text_import_error = e


try:
    from .model_image import ImageClassifier
except Exception as e:
    ImageClassifier = None
    _image_import_error = e

_MODEL_INFO = {
    "Text Classification": {
        "name": "DistilBERT SST-2",
        "category": "Text",
        "description": "Sentiment analysis model fine-tuned on SST-2; returns POSITIVE/NEGATIVE with confidence.",
    },
    "Image Classification": {
        "name": "ViT Base Patch16-224",
        "category": "Vision",
        "description": "Vision Transformer (ViT) image classifier; returns top labels with scores.",
    },
}

class Controller:
    def __init__(self):
        if TextClassifier is None:
            raise RuntimeError(f"Failed to load TextClassifier: {_text_import_error}")
        self.text_model = TextClassifier()

       
        self.image_model = ImageClassifier() if ImageClassifier else None

    # ---- run methods ----
    def run_text(self, text: str):
        return self.text_model.run(text)

    def run_image(self, image_path: str):
        if self.image_model is None:
            return {"result": f"[Image model pending] {image_path}", "elapsed_ms": 0}
        return self.image_model.run(image_path)

    # ---- info for GUI ----
    def model_info(self, task: str):
        """Return dict with name, category, description for the selected task."""
        return _MODEL_INFO.get(task, {"name": "Unknown", "category": "-", "description": "-"})


class DummyController:
    def run_text(self, text: str):
        return {"result": f"[Dummy TEXT output] {text[:60]}...", "elapsed_ms": 0}
    def run_image(self, image_path: str):
        return {"result": f"[Dummy IMAGE output] {image_path}", "elapsed_ms": 0}
