try:
    from .model_text import TextClassifier
except Exception as e:
    TextClassifier = None
    _text_import_error = e

try:
    from .model_image_caption import ImageCaptioner
except Exception as e:
    ImageCaptioner = None
    _cap_import_error = e

class Controller:
    def __init__(self):
        if TextClassifier is None:
            raise RuntimeError(f"Failed to load TextClassifier: {_text_import_error}")
        self.text_model = TextClassifier()

        if ImageCaptioner is None:
            raise RuntimeError(f"Failed to load ImageCaptioner: {_cap_import_error}")
        self.caption_model = ImageCaptioner()

    # ---- run methods ----
    def run_text(self, text: str):
        return self.text_model.run(text)

    def run_image_caption(self, image_path: str):
        return self.caption_model.run(image_path)

    # ---- info for GUI ----
    def model_info(self, task: str):
        t = (task or "").lower()
        if "text" in t:
            return {
                "name": self.text_model.name,
                "category": "NLP (Zero-shot)",
                "description": self.text_model.description
            }
        if "caption" in t:
            return {
                "name": self.caption_model.name,
                "category": "Vision–Language",
                "description": self.caption_model.description
            }
        return {"name": "-", "category": "-", "description": "-"}
