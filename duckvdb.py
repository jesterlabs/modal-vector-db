import duckdb
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
from utils import json_to_uuid
from dataclasses import dataclass
import json

@dataclass
class Result:
    id: uuid.UUID
    metadata: dict
    distance: float


class DuckVDB:
    def __init__(self, db_path: str, embedding_dim: Optional[int] = None, new_table: bool = False):
        self.db_path = db_path
        self.embedding_dim=embedding_dim
        self.conn = duckdb.connect(self.db_path)
        self.conn.sql("INSTALL vss; LOAD vss")
        self.conn.sql("INSTALL json; LOAD json")
        self.conn.sql("PRAGMA memory_limit='100GB'")
        self.conn.sql("PRAGMA threads=16")
        self.conn.sql("PRAGMA temp_directory='/tmp'")
        if new_table:
            self.drop_table()
        self.create_table()

    def create_table(self):
        # only create table if doesn't exist
        self.conn.sql(f"CREATE TABLE IF NOT EXISTS items (id UUID PRIMARY KEY, metadata JSON, embedding FLOAT[{self.embedding_dim}])")

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
                json_path = key.split('.')
                field = 'metadata'
                for part in json_path:
                    field = f"json_extract({field}, '{part}')"
            else:
                field = f"json_extract(metadata, '{key}')"
            
            # Handle operator tuples like (">", 50) or ("contains", "Flying")
            if isinstance(value, tuple) and len(value) == 2:
                operator, val = value
                if operator == "=":
                    if isinstance(val, list):
                        return f"{field}::JSON = '{json.dumps(val)}'::JSON"
                elif operator == "contains":
                    # Determine the appropriate type cast based on the value type
                    if isinstance(val, str):
                        return f"list_contains({field}::JSON::VARCHAR[], '{val}'::VARCHAR)"
                    elif isinstance(val, int):
                        return f"list_contains({field}::JSON::INTEGER[], {val}::INTEGER)"
                    elif isinstance(val, float):
                        return f"list_contains({field}::JSON::DOUBLE[], {val}::DOUBLE)"
                    else:
                        return f"list_contains({field}::JSON::VARCHAR[], '{str(val)}'::VARCHAR)"
                if isinstance(val, str):
                    return f"{field} {operator} '{val}'"
                return f"{field} {operator} {val}"
            
            # Handle direct value comparison
            if isinstance(value, str):
                return f"{field} = '{value}'"
            if isinstance(value, list):
                return f"{field}::JSON = '{json.dumps(value)}'::JSON"
            return f"{field} = {value}"

        return " AND ".join(format_filter(key, value) for key, value in filters.items())

    def query(self, embedding: np.array, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> list[Result]:
        embedding_dim: int = embedding.shape[0]
        if filters:
            filter_string = self.format_filters(filters)
        else:
            filter_string = "1=1"
        
        query_sql = f"""
            SELECT DISTINCT id, metadata, array_cosine_distance(embedding, ?::FLOAT[{embedding_dim}]) as distance
            FROM items 
            WHERE {filter_string}
            ORDER BY distance ASC
            LIMIT ?;
        """
        
        params = [embedding, k]
        result = self.conn.execute(query_sql, params).fetchall()
        return [Result(id=row[0], metadata=json.loads(row[1]), distance=row[2]) for row in result]
    
    def num_rows(self):
        return self.conn.sql("SELECT COUNT(*) FROM items;").fetchone()[0]



