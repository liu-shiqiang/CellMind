# tools/annotate_fallback.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List

class FallbackArgs(BaseModel):
    cluster_id: int
    marker_genes: List[str]
    docs: str | None = None       # JSON-encoded docs
    ref_csv: str
    output_dir: str

@tool(name="annotate_with_marker_knowledge",
      description="LLM fallback annotation using marker gene reference.",
      args_schema=FallbackArgs)
def annotate_with_marker_knowledge(cluster_id: int, marker_genes: List[str],
                                   docs: str | None, ref_csv: str,
                                   output_dir: str, llm=None) -> str:
    # TODO: build prompt & llm.invoke
    return f'{{"cluster":{cluster_id},"celltype":"Unknown"}}'
