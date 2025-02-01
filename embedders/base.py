import modal
import numpy as np
app = modal.App("embedders")


class BaseEmbedder:
    def __init__(self):
        pass

    def embed(self, text: str) -> np.array:
        pass