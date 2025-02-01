import modal
import numpy as np

app = modal.App("modal-vector-db")
image = modal.Image.debian_slim()\
    .pip_install("sentence_transformers")

@app.cls(image=image)
class SentenceTransformersEmbedder:
    model: str = modal.parameter()

    @modal.enter()
    def enter(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model)

    @modal.method()
    def embed(self, text: str) -> np.array:
        return self.model.encode(text)
