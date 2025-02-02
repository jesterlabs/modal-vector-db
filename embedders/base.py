import modal
import numpy as np

BASE_EMBEDDER_APP_NAME = "embedders"
BASE_EMBEDDERS_APP = modal.App(BASE_EMBEDDER_APP_NAME)

class BaseEmbedder:    

    @property
    def embedder_name(self):
        return f"{self.__class__.__name__}"

    def embed(self, text: str) -> np.array:
        raise NotImplementedError("Subclass must implement embed method")