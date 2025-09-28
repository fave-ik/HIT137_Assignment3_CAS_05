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
        "description": "Sentiment analysis model (POSITIVE/NEGATIVE) fine-tuned on SST-2.",
    },
    "Image Classification": {
        "name": "ViT Base 16/224",
        "category": "Vision",
        "description": "Vision Transformer image classifier; returns top labels with scores.",
    },
}

class Controller:
    def __init__(self):
        if TextClassifier is None:
            raise RuntimeError(f"Failed to load TextClassifier: {_text_import_error}")
        self.text_model = TextClassifier()
        self.image_model = ImageClassifier() if ImageClassifier else None

    def run_text(self, text: str):
        return self.text_model.run(text)

    def run_image(self, image_path: str):
        if self.image_model is None:
            return {"result": f"[Image model pending] {image_path}", "elapsed_ms": 0}
        return self.image_model.run(image_path)

    def model_info(self, task: str):
        return _MODEL_INFO.get(task, {"name": "-", "category": "-", "description": "-"})
