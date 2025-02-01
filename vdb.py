import modal
import numpy as np
from duckvdb import DuckVDB
from typing import List, Optional, Any
from embedders.sentence_transformers import SentenceTransformersEmbedder


embedding_fn = SentenceTransformersEmbedder(model="all-MiniLM-L6-v2")
app = modal.App("modal-vector-db")
image = modal.Image.debian_slim()\
    .pip_install(
        "duckdb", 
        "pandas", 
        "pyarrow", 
        "numpy"
    )
vol = modal.Volume.from_name("modal-vector-db-volume", create_if_missing=True)
MOUNT_PATH = "/db"
@app.cls(volumes={MOUNT_PATH: vol}, image=image)
class WrapperClass:
    name: str = modal.parameter()

    @modal.enter()
    def enter(self):
        # if embedding_fn is callable:
        #     embedding_model = embedding_fn
        # else:
        #     embedding_model = modal.Function.lookup(embedding_fn)
        embedding_fn = modal.Cls.from_name("embedders", "SentenceTransformersEmbedder")
        self.db: DuckVDB = DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_function=embedding_fn)

    @modal.method()
    def insert(self, metadatas: list[dict[str, Any]], embeddings: List[np.array]):
        self.db.write(metadatas, embeddings)

    @modal.method()
    def query(self, query: str, k: int = 10, filters: Optional[dict[str, Any]] = None):
        return self.db.query(query, k, filters)
    

    
        
@app.local_entrypoint()
def main():
    import json
    pokedata = json.load(open("pokemon.json"))
    first_10_pokemon: list[dict] = pokedata[:10]
    metadatas: list[dict] = [json.dumps(pokemon) for pokemon in first_10_pokemon] #[{"name": pokemon["name"], "description": pokemon["description"]} for pokemon in first_10_pokemon]
    embedding_fn = modal.Cls.from_name("embedders", "SentenceTransformersEmbedder")()
    embeddings: list[np.array] = list(embedding_fn.embed.map([pokemon["description"] for pokemon in first_10_pokemon]))
    remote_db = WrapperClass(name="pokemon_db")
    remote_db.insert.remote(metadatas, embeddings)
    remote_db.query.remote("water", k=10, filters={"base.Attack": ("<=", 49)})