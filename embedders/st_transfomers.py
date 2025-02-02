import modal
import numpy as np
from embedders.base import BASE_EMBEDDERS_APP as app, BaseEmbedder

image = modal.Image.debian_slim()\
    .pip_install("sentence_transformers")

@app.cls(image=image)
class SentenceTransformersEmbedder(BaseEmbedder):
    model_name: str = modal.parameter()

    @property
    def embedder_name(self):
        return f"{self.__class__.__name__}-{self.model_name}"
    
    @modal.enter()
    def enter(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)

    @modal.method()
    def embed(self, text: str) -> np.array:
        return self.model.encode(text)
