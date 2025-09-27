from abc import ABC, abstractmethod
import time
from functools import wraps

# ---------- Decorators (multiple decorators requirement) ----------
def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)
        # Normalize return shape for the GUI
        if isinstance(out, dict) and "result" in out:
            out["elapsed_ms"] = elapsed_ms
            return out
        return {"result": out, "elapsed_ms": elapsed_ms}
    return wrapper

def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        cls = args[0].__class__.__name__ if args else fn.__name__
        print(f"[LOG] {cls}.{fn.__name__} called")
        return fn(*args, **kwargs)
    return wrapper

# ---------- Mixins / base classes ----------
class ModelInfoMixin:
    def describe(self):
        return f"{getattr(self, '_name', 'Unknown')} — {getattr(self, '_task', 'Task')}"

class PrePostMixin:
    # Overridable hooks (method overriding shown in child classes)
    def preprocess(self, x): return x
    def postprocess(self, y): return y

class ModelBase(ABC, PrePostMixin):
    """Abstract base for all models (encapsulation via underscored attrs)."""
    def __init__(self, name: str, task: str):
        self._name = name   # encapsulated attributes
        self._task = task

    @abstractmethod
    def run(self, x):
        """Run model on input x and return a dict with 'result' (polymorphic API)."""
        raise NotImplementedError
