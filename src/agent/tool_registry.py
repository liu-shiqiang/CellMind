import sys
import os

# 获取项目根目录（genomix-agent）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)  # 将项目根目录加入Python的搜索路径

from src.tools.validata_input import validate_input_file
from src.tools.load_h5ad import load_h5ad_data
from src.tools.extract_embeddings import extract_embeddings_with_scgpt
from src.tools.retrieve_cell_context import retrieve_cell_context
from src.tools.clustering_marker import cluster_and_rank_markers
from src.tools.annotate_with_markers import annotate_with_markers
from src.tools.cluster_diff import cluster_and_diff
from src.tools.annotate_with_cellrag import annotate_with_cellrag
from src.tools.get_celltype_markers import get_celltype_markers_langchain 
from src.tools.gene_set_enrichment_analysis import gene_set_enrichment_analysis



TOOLS = [
    load_h5ad_data,
    extract_embeddings_with_scgpt,
    cluster_and_rank_markers,
    annotate_with_cellrag,
    gene_set_enrichment_analysis,
]
