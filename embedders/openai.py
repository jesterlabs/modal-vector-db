import modal
import numpy as np
from typing import Optional, Union, List
from embedders.base import BASE_EMBEDDERS_APP as app, BaseEmbedder

image = modal.Image.debian_slim()\
    .pip_install("openai")

@app.cls(image=image, secrets=[modal.Secret.from_name("openai-secret")])
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        
    @modal.enter()
    def enter(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

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
            kwargs["dimensions"] = self.dimensions

        response = self.client.embeddings.create(model=self.model_name, input=text, **self.kwargs)
        
        # Handle both single string and list inputs
        if isinstance(text, str):
            return np.array(response.data[0].embedding)
        else:
            return np.array([data.embedding for data in response.data]) 