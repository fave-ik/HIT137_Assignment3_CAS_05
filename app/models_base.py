# app/models_base.py
from abc import ABC, abstractmethod
import time
from functools import wraps

def timed(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        out = fn(self, *args, **kwargs)
        return out, int((time.time() - t0) * 1000)
    return wrapper

def log_call(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        
        return fn(self, *args, **kwargs)
    return wrapper

class ModelInfoMixin:
    def model_info(self):
        return {
            "name": getattr(self, "_name", "-"),
            "category": getattr(self, "_category", "-"),
            "description": getattr(self, "_description", "-"),
        }

class ModelBase(ABC):
    """
    Base class all models inherit. Now accepts description for GUI.
    """
    def __init__(self, name: str, task: str, category: str, description: str = "-"):
        self._name = name
        self._task = task
        self._category = category
        self._description = description

    @abstractmethod
    def preprocess(self, x): ...

    @abstractmethod
    def _infer(self, x): ...

    @abstractmethod
    def postprocess(self, y): ...

    @timed
    @log_call
    def run(self, x):
        x_p = self.preprocess(x)
        y = self._infer(x_p)
        out = self.postprocess(y)
        return out
