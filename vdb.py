import modal
import numpy as np
from duckvdb import DuckVDB
from embedders import EmbedderConfig
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
    def __init__(self, name: str, embedder_config: EmbedderConfig, create_new_table: bool = False):
        self.name = name
        self.embedding_fn = modal.Cls.from_name(embedder_config.app_name, embedder_config.name)(**embedder_config.params).embed
        self.embedding_dim = embedder_config.dims
        self.create_new_table: bool = create_new_table

    @modal.enter()
    def enter(self):
        self.db: DuckVDB = DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_function=self.embedding_fn.remote, embedding_dim=self.embedding_dim, new_table=self.create_new_table)

    @modal.method()
    def insert(self, metadatas: list[dict[str, Any]], embeddings: Optional[List[np.array]] = None, embed_field: Optional[str] = None):
        import json
        stringified_metadatas = [json.dumps(metadata) for metadata in metadatas]
        if embeddings is None:
            if embed_field is None:
                to_embed = stringified_metadatas
            else:
                to_embed = [metadata[embed_field] for metadata in metadatas]
            embeddings = self.embedding_fn.map(to_embed)

        self.db.write(stringified_metadatas, embeddings)

    @modal.method()
    def query(self, query: str, k: int = 10, filters: Optional[dict[str, Any]] = None):
        return self.db.query(query, k, filters)
    
    @modal.method()
    def num_rows(self):
        return self.db.num_rows()
    

    
        
@app.local_entrypoint()
def main():
    import json
    pokedata = json.load(open("examples/data/pokemon.json"))
    print(f" Number of pokemon: {len(pokedata)}")
    metadatas: list[dict] = [pokemon for pokemon in pokedata]
    to_embed = [metadata["description"] for metadata in metadatas]
    embeddings = list(modal.Cls.from_name("embedders", "SentenceTransformersEmbedder")(model_name="all-MiniLM-L6-v2").embed.map(to_embed))

    # Insert w/ vectors & embeddings
    remote_db = ModalVectorDB(name="pokemon", embedder_config=EmbedderConfig(name="SentenceTransformersEmbedder", dims=384, params={"model_name": "all-MiniLM-L6-v2"}), create_new_table=True)
    remote_db.insert.remote(metadatas[10:], embeddings[10:])
    print(f"Number of rows after bulk insert (metadatas & embeddings): {remote_db.num_rows.remote()}")


    # Insert with just metadatas, should embed using specified embedder
    remote_db.insert.remote(metadatas[:10], embed_field="description")
    print(f"Number of rows afer insert of metadatas: {remote_db.num_rows.remote()}")

    # Similarity search
    results = remote_db.query.remote("water", k=3)
    print("## Top 3 most 'Water' like Pokemon (based on description) ##")
    print([(result.metadata["name"], result.metadata["description"], result.metadata["type"], result.distance) for result in results])

    # Similarity search with filters
    results = remote_db.query.remote("fire", k=3, filters={"base.Attack": ("<=", 49)})
    print("## Fire with Attack <= 49 ##")
    print([(result.metadata["name"], result.metadata["description"], result.metadata["type"], result.metadata["base"]["Attack"], result.distance) for result in results])
