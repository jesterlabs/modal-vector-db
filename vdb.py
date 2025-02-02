import modal
import numpy as np
from duckvdb import DuckVDB
from typing import List, Optional, Any, Union, Callable


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
    def __init__(self, name: str, embedder: Union[Callable[..., Any], modal.Cls], embedding_dim: int):
        self.name = name
        self.embedding_fn = embedder.embed.remote if isinstance(embedder, modal.Cls) else embedder
        self.embedding_dim = embedding_dim

    @modal.enter()
    def enter(self):
        self.db: DuckVDB = DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_function=self.embedding_fn, embedding_dim=self.embedding_dim)

    @modal.method()
    def insert(self, metadatas: list[dict[str, Any]], embeddings: List[np.array]):
        self.db.write(metadatas, embeddings)

    @modal.method()
    def query(self, query: str, k: int = 10, filters: Optional[dict[str, Any]] = None):
        return self.db.query(query, k, filters)
    

    
        
@app.local_entrypoint()
def main():
    import json
    pokedata = json.load(open("examples/data/pokemon.json"))
    first_10_pokemon: list[dict] = pokedata[:10]
    metadatas: list[dict] = [json.dumps(pokemon) for pokemon in first_10_pokemon] #[{"name": pokemon["name"], "description": pokemon["description"]} for pokemon in first_10_pokemon]
    embedding_fn = modal.Cls.from_name("embedders", "SentenceTransformersEmbedder")(model_name="all-MiniLM-L6-v2")
    embeddings: list[np.array] = list(embedding_fn.embed.map([pokemon["description"] for pokemon in first_10_pokemon]))
    remote_db = WrapperClass(name="pokemon_two_db", embedder=embedding_fn, embedding_dim=384)
    remote_db.insert.remote(metadatas, embeddings)
    results = remote_db.query.remote("water", k=10, filters={"base.Attack": ("<=", 49)})
    print(results)