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

    def run_text(self, raw_text: str):
        """
        Allows labels in the first line:
        labels: tech, sports, politics
        The rest of the textarea is the input text.
        If not provided, we use a sensible default label set.
        """
        lines = [l for l in raw_text.splitlines() if l.strip()]
        labels = None

        if lines and lines[0].lower().startswith("labels:"):
            # parse comma-separated labels
            labels_str = lines[0].split(":", 1)[1]
            labels = [s.strip() for s in labels_str.split(",") if s.strip()]
            text = "\n".join(lines[1:]).strip()
        else:
            text = raw_text.strip()

        if not labels:
            labels = ["technology", "sports", "politics", "education", "entertainment"]

        return self.text_model.run({"text": text, "labels": labels})

    # keep a dummy image path hook so GUI still runs even if someone clicks it
    def run_image(self, image_path: str):
        return {"result": [{"label": "[Image model not implemented in my part]", "score": 1.0}], "elapsed_ms": 0}

    def model_info(self, task_name: str):
        # We only have text implemented here
        return self.text_model.model_info()
