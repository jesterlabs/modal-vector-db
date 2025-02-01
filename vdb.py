import modal
from duckvdb import DuckVDB
APP_NAME = "modal-vector-db"
app = modal.App(APP_NAME)
image = modal.Image.debian_slim()\
    .pip_install(
        "duckdb", 
        "pandas", 
        "pyarrow", 
        "numpy"
    )
vol = modal.Volume.from_name("modal-vector-db-volume", create_if_missing=True)
MOUNT_PATH = "/db"

@app.cls(volumes={MOUNT_PATH: vol})
class VectorDB:
    name: str = modal.parameter()
    embedding_model_name: str = modal.parameter()

    @modal.enter()
    def enter(self):
        embedding_model = modal.Function.lookup(self.embedding_model_name)
        self.db: DuckVDB = DuckVDB(db_path=f"{MOUNT_PATH}/{self.name}.duckdb", embedding_function=self.embedding_model)

        

    @modal.exit()
    def exit(self):
        pass
