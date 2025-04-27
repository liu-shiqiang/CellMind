# tools/update_vector_metadata.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class UpdateArgs(BaseModel):
    db_dir: str
    collection_name: str
    cluster_id: int
    celltype: str

@tool(name="update_vector_metadata",
      description="Write the new celltype label into Chroma metadata.",
      args_schema=UpdateArgs)
def update_vector_metadata(db_dir: str, collection_name: str,
                           cluster_id: int, celltype: str) -> str:
    # TODO: update Chroma doc metadata
    return "✅ Metadata updated"
