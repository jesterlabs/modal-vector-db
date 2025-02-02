import modal
import numpy as np
from dataclasses import dataclass
from typing import Any

BASE_EMBEDDER_APP_NAME = "embedders"
BASE_EMBEDDERS_APP = modal.App(BASE_EMBEDDER_APP_NAME)

@dataclass
class EmbedderConfig:
    name: str
    dims: int
    params: dict[str, Any]
    app_name: str = BASE_EMBEDDER_APP_NAME



class BaseEmbedder:    
    def embed(self, text: str) -> np.array:
        raise NotImplementedError("Subclass must implement embed method")