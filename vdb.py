import modal
import numpy as np
from duckvdb import DuckVDB
from embedders import BaseEmbedder, BASE_EMBEDDER_APP_NAME
from typing import Any, List, Optional

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
class ModalVectorDB:
    def __init__(self, name: str, embedder_name: str, embedding_dim: int, embedder_kwargs: Optional[dict] = None, create_new_table: bool = False):
        self.name = name
        self.embedder_name = embedder_name
        try:
            self.embedder = modal.Cls.from_name(BASE_EMBEDDER_APP_NAME, embedder_name)(**(embedder_kwargs or {}))
            self.embedding_fn = self.embedder.embed if hasattr(self.embedder, "embed") else self.embedder
            self.embedding_dim = embedding_dim
        except modal.exception.NotFoundError:
            raise ValueError(f"Embedder {embedder_name} not found in {BASE_EMBEDDER_APP_NAME}")
        
        self.create_new_table = create_new_table
        self.table_exists = self._check_table_exists()

    # def with_embedder(self, embedder: modal.Function | modal.Cls):
    #     self.embedding_fn = embedder.embed #if isinstance(embedder, BaseEmbedder) else embedder
    #     return self

    @modal.enter()
    def enter(self):
        self.db = DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_dim=self.embedding_dim, new_table=self.create_new_table)

    def _create_db(self) -> DuckVDB:
        return DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_dim=self.embedding_dim, new_table=self.create_new_table)

    def _check_table_exists(self) -> bool:
        # check .duckdb file exists
        import os
        return os.path.exists(f"{MOUNT_PATH}/{self.name}.duckdb")
    
    @modal.method()
    def insert(self, metadatas: list[dict], embeddings: Optional[List[np.array]] = None, embed_field: Optional[str] = None):
        import json
        stringified_metadatas: list[str] = [json.dumps(metadata) if isinstance(metadata, dict) else metadata for metadata in metadatas]
        if embeddings is None:
            if embed_field is None:
                to_embed = stringified_metadatas
            else:
                to_embed = [metadata[embed_field] for metadata in metadatas]
            embeddings = self.embedding_fn.map(to_embed)

        self.db.write(stringified_metadatas, embeddings)

    @modal.method()
    def query(self, query: str, k: int = 10, filters: Optional[dict[str, Any]] = None):
        query_vector: np.array = self.embedding_fn.remote(query)
        return self.db.query(query_vector, k, filters)
    
    @modal.method()
    def num_rows(self):
        return self.db.num_rows()
    

    
        
@app.local_entrypoint()
def main():
    import json
    from pprint import pprint
    from embedders import BASE_EMBEDDER_APP_NAME
    embedder = modal.Cls.from_name(BASE_EMBEDDER_APP_NAME, "SentenceTransformersEmbedder")(model_name="all-MiniLM-L6-v2")
    pokedata = json.load(open("examples/data/pokemon.json"))
    print(f" Number of pokemon: {len(pokedata)}")
    metadatas: list[dict] = [pokemon for pokemon in pokedata]
    to_embed: list[str] = [metadata["description"] for metadata in metadatas]
    embeddings = list(embedder.embed.map(to_embed))

    # Insert w/ vectors & embeddings
    remote_db = ModalVectorDB(name="pokemon", create_new_table=True, embedding_dim=384, embedder_name="SentenceTransformersEmbedder", embedder_kwargs={"model_name": "all-MiniLM-L6-v2"})
    remote_db.insert.remote(metadatas[10:], embeddings[10:])
    print(f"Number of rows after bulk insert (metadatas & embeddings): {remote_db.num_rows.remote()}")


    # Insert with just metadatas, should embed using specified embedder
    remote_db.insert.remote(metadatas[:10], embed_field="description")
    print(f"Number of rows afer insert of metadatas: {remote_db.num_rows.remote()}")

    # Similarity search
    results = remote_db.query.remote("psychic", k=3)
    print("## Top 3 most 'psychic' like Pokemon (based on description) ##")
    pprint([(result.metadata["name"]["english"], result.metadata["description"], result.metadata["type"], result.distance) for result in results])
    print("===\n")

    # Similarity search with filters
    results = remote_db.query.remote("rainbow", k=3, filters={"base.Attack": (">", 50), "type": ("contains", "Flying")})
    print("## Rainbow Flying Pokemon with Attack > 50 ##")
    pprint([(result.metadata["name"]["english"], result.metadata["description"], result.metadata["type"], result.metadata["base"]["Attack"], result.distance) for result in results])