try:
    from .model_text import TextClassifier
except Exception as e:
    TextClassifier = None
    _import_error = e

class Controller:
    def __init__(self):
        if TextClassifier is None:
            raise RuntimeError(f"Failed to load TextClassifier: {_import_error}")
        self.text_model = TextClassifier()

    def run_text(self, text: str):
        return self.text_model.run(text)

    def run_image(self, image_path: str):
        return {"result": f"[Dummy IMAGE output] {image_path}", "elapsed_ms": 0}

class DummyController:
    def run_text(self, text: str):
        return {"result": f"[Dummy TEXT output] {text[:60]}...", "elapsed_ms": 0}
    def run_image(self, image_path: str):
        return {"result": f"[Dummy IMAGE output] {image_path}", "elapsed_ms": 0}
