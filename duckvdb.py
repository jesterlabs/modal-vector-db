import os
import duckdb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import uuid
from dataclasses import dataclass

@dataclass
class Result:
    id: uuid.UUID
    metadata: dict
    distance: float

def json_to_uuid(json_data, namespace=uuid.NAMESPACE_DNS):
    """Convert JSON data to a deterministic UUID."""
    json_str = json.dumps(json_data, sort_keys=True)  # Ensure consistent ordering
    return uuid.uuid5(namespace, json_str)

class EmbeddingFunction:
    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim

    def __call__(self, x: str) -> np.array:
        return np.random.rand(self.vector_dim)

class DuckVDB:
    def __init__(self, db_path: str, embedding_function: EmbeddingFunction, new_table: bool = False):
        self.db_path = db_path
        self.embedding_function = embedding_function
        self.conn = duckdb.connect(self.db_path)
        # install vss extension
        self.conn.sql("INSTALL vss; LOAD vss")
        self.conn.sql("INSTALL json; LOAD json")
        self.conn.sql("PRAGMA memory_limit='100GB'")
        self.conn.sql("PRAGMA threads=16")
        self.conn.sql("PRAGMA temp_directory='/tmp'")
        if new_table:
            self.drop_table()
        self.create_table()

    def create_table(self):
        embedding_dim = self.embedding_function.vector_dim
        # only create table if doesn't exist
        self.conn.sql(f"CREATE TABLE IF NOT EXISTS items (id UUID PRIMARY KEY, metadata JSON, embedding FLOAT[{embedding_dim}])")

    def drop_table(self):
        self.conn.sql("DROP TABLE IF EXISTS items;")

    def create_index(self):
        # drop index if it exists
        self.conn.sql("DROP INDEX IF EXISTS hnsw_index;")
        self.conn.sql("CREATE INDEX hnsw_index ON items USING HNSW (embedding) WITH (metric = 'cosine');")
        self.conn.sql("SET hnsw_enable_experimental_persistence = true;")

    def load_from_parquet(self, parquet_path: str):
        self.conn.sql(f"CREATE TABLE items AS SELECT * FROM read_parquet('{parquet_path}');")
        self.create_index()

    def write(self, metadatas: List[Dict[str, Any]], embeddings: List[np.array]):
        import pandas as pd
        
        ids: list[uuid.UUID] = [json_to_uuid(metadata) for metadata in metadatas]
        df = pd.DataFrame({
            'id': ids,
            'metadata': metadatas,
            'embedding': embeddings
        })
        # Skip duplicates silently
        self.conn.execute("""
            INSERT INTO items 
            SELECT * FROM df 
            ON CONFLICT (id) DO NOTHING
        """)

    def format_filters(self, filters: Dict[str, Any]) -> str:
        def format_filter(key: str, value: Any) -> str:
            # Check if this is a JSON path (contains dots)
            if '.' in key:
                # Use json_extract for nested fields
                json_path = key.split('.')
                field = f"json_extract(metadata, '{json_path[-1]}')"
                
                # For nested paths deeper than one level, use multiple json_extract calls
                for part in reversed(json_path[:-1]):
                    field = f"json_extract(json_extract(metadata, '{part}'), '{json_path[-1]}')"
            else:
                # Regular column
                field = key
            
            # Handle operator tuples like ("<", 20)
            if isinstance(value, tuple) and len(value) == 2:
                operator, val = value
                if isinstance(val, str):
                    return f"{field} {operator} '{val}'"
                return f"{field} {operator} {val}"
            
            # Handle direct value comparison
            if isinstance(value, str):
                return f"{field} = '{value}'"
            return f"{field} = {value}"

        return " AND ".join(format_filter(key, value) for key, value in filters.items())

    def query(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> list[Result]:
        embedding = self.embedding_function(query)
        
        if filters:
            filter_string = self.format_filters(filters)
        else:
            filter_string = "1=1"
        
        query_sql = f"""
            SELECT DISTINCT id, metadata, array_cosine_distance(embedding, ?::FLOAT[{self.embedding_function.vector_dim}]) as distance
            FROM items 
            WHERE {filter_string}
            ORDER BY distance
            LIMIT ?;
        """
        
        # Only include embedding and k in params since filters are now formatted in the SQL string
        params = [embedding, k]
        
        result = self.conn.execute(query_sql, params).fetchall()
        return [Result(id=row[0], metadata=row[1], distance=row[2]) for row in result]



