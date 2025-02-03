import modal
import numpy as np
from dataclasses import dataclass
from typing import Any
from typing import Union, List

BASE_EMBEDDER_APP_NAME = "embedders"
app = modal.App(BASE_EMBEDDER_APP_NAME)

@dataclass
class EmbedderConfig:
    name: str
    dims: int
    params: dict[str, Any]
    app_name: str = BASE_EMBEDDER_APP_NAME


class BaseEmbedder:   

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @property
    def embedder_name(self):
        return self.__class__.__name__
    
    def get_dimensions(self):
        raise NotImplementedError("Subclass must implement dims method")
    
    def embed(self, text: str) -> np.array:
        raise NotImplementedError("Subclass must implement embed method")
    

# OPENAI EMBEDDER
openai_image = modal.Image.debian_slim()\
    .pip_install("openai")
@app.cls(image=openai_image, secrets=[modal.Secret.from_name("openai-secret")], serialized=True)
class OpenAIEmbedder(BaseEmbedder):
    MODEL_DIMS_MAP = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    @modal.enter()
    def enter(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def get_dimensions(self):
        return self.MODEL_DIMS_MAP[self.model_name]

    @modal.method()
    def embed(self, text: Union[str, List[str]]) -> np.array:
        kwargs = {
            "model": self.model_name,
            "input": text,
            "encoding_format": self.encoding_format,
        }
        # Only add optional parameters if they are set
        if self.user is not None:
            kwargs["user"] = self.user
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dims

        response = self.client.embeddings.create(model=self.model_name, input=text, **self.kwargs)
        
        # Handle both single string and list inputs
        if isinstance(text, str):
            return np.array(response.data[0].embedding)
        else:
            return np.array([data.embedding for data in response.data]) 
        

# SENTENCE TRANSFORMERS EMBEDDER
sentence_transformers_image = modal.Image.debian_slim()\
    .pip_install("sentence_transformers")
@app.cls(image=sentence_transformers_image, serialized=True)
class SentenceTransformersEmbedder(BaseEmbedder):

    def get_dimensions(self):
        return self._dims
    
    @modal.enter()
    def enter(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.model_name)
        self._dims = self.model.get_sentence_embedding_dimension()

    @modal.method()
    def embed(self, text: str) -> np.array:
        return self.model.encode(text)
