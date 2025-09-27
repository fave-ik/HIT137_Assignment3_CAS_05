from transformers import pipeline
from .models_base import ModelBase, ModelInfoMixin, timed, log_call

class TextClassifier(ModelInfoMixin, ModelBase):  # multiple inheritance
    def __init__(self, model_id: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        super().__init__("DistilBERT SST-2", "Text Classification")
        # Encapsulated pipeline object
        self._pipe = pipeline("sentiment-analysis", model=model_id)

    @timed
    @log_call
    def run(self, text: str):
        text = self.preprocess(text)
        preds = self._pipe(text)  # [{'label': 'POSITIVE', 'score': ...}]
        return {"result": preds}
