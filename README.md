# Modal Vector DB

A hacky example implementation of a vector database using DuckDB + Modal for serverless vector search. This is meant as a proof of concept and learning example - not for production use.

## Overview

This project demonstrates how to:
- Create and manage vector embeddings using different embedding models (OpenAI, SentenceTransformers)
- Store vectors and metadata in DuckDB with vector similarity search capabilities
- Run everything serverlessly using Modal
- Perform filtered vector similarity search

## Installation

```bash
pip install modal
# Do all of your modal setup
```

## Example Usage
Run the following under a modal app entrypoint (can see an example in `vdb.py`)
```python
import json
from pprint import pprint
from embedders import BASE_EMBEDDER_APP_NAME
from vdb import BASE_DB_APP_NAME
pokedata = json.load(open("examples/data/pokemon.json"))
print(f" Number of pokemon: {len(pokedata)}")

# Embed the data for bulk insert
embedder = modal.Cls.from_name(BASE_EMBEDDER_APP_NAME, "SentenceTransformersEmbedder")(model_name="all-MiniLM-L6-v2")
metadatas: list[dict] = [pokemon for pokemon in pokedata]
to_embed: list[str] = [metadata["description"] for metadata in metadatas]
embeddings = list(embedder.embed.map(to_embed))

# Insert w/ vectors & embeddings
vdb = modal.Cls.from_name(BASE_DB_APP_NAME, "ModalVectorDB")
remote_db = vdb(name="pokemon", create_new_table=True, embedding_dim=384, embedder_name="SentenceTransformersEmbedder", embedder_kwargs={"model_name": "all-MiniLM-L6-v2"})
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
```
