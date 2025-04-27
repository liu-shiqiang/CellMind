# tools/clustering_marker.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class ClusterArgs(BaseModel):
    embedding_path: str
    output_dir: str
    top_k: int = 10
    ref_csv: str | None = None

@tool(name="cluster_and_rank_markers",
      description="Leiden clustering + UMAP + Wilcoxon marker discovery.",
      args_schema=ClusterArgs)
def cluster_and_rank_markers(embedding_path: str, output_dir: str,
                             top_k: int, ref_csv: str | None) -> str:
    # TODO: clustering + marker csv
    clustered_path = embedding_path.replace("_emb", "_clustered")
    marker_csv = clustered_path.replace(".h5ad", "_markers.csv")
    return f"{clustered_path}|{marker_csv}"   # '|' separator for planner
