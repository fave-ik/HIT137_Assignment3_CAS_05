class DummyController:
    """Temporary controller so the GUI runs today.
    Member 3 will replace with actual logic calling Hugging Face models.
    """
    def run_text(self, text: str):
        return {"result": f"[Dummy TEXT output] {text[:60]}...", "elapsed_ms": 0}
    def run_image(self, image_path: str):
        return {"result": f"[Dummy IMAGE output] {image_path}", "elapsed_ms": 0}
