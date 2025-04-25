# tools/retrieve_cell_context.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class RetrieveArgs(BaseModel):
    clustered_h5ad_path: str
    db_dir: str
    collection_name: str
    top_k: int = 5

@tool(
      description="Return JSON list of {cluster_id, top_score, docs}.",
      args_schema=RetrieveArgs)
def retrieve_cell_context(clustered_h5ad_path: str, db_dir: str,
                          collection_name: str, top_k: int) -> str:
    # TODO: vector search
    return '[{"cluster":0,"top_score":0.9,"docs":[...]}]'
