# app/controllers.py

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
        out = self.text_model.run(text)
        # Normalize to {"result": ..., "elapsed_ms": ...}
        if isinstance(out, tuple) and len(out) == 2:
            result, elapsed_ms = out
            return {"result": result, "elapsed_ms": elapsed_ms}
        elif isinstance(out, dict) and ("result" in out or "elapsed_ms" in out):
            return out
        else:
            return {"result": out, "elapsed_ms": 0}

    def run_image(self, image_path: str):
        return {"result": f"[Dummy IMAGE output] {image_path}", "elapsed_ms": 0}

    def model_info(self, task: str):
        return getattr(self.text_model, "info", lambda: {})()
