from src.tools.validata_input import validate_input_file
from src.tools.load_h5ad import load_h5ad_data
from src.tools.extract_embeddings import extract_embeddings_with_scgpt
from src.tools.retrieve_cell_context import retrieve_cell_context
from src.tools.clustering_marker import cluster_and_rank_markers
from src.tools.annotate_with_markers import annotate_with_markers
from src.tools.cluster_diff import cluster_and_diff
from src.tools.annotate_with_cellrag import annotate_with_cellrag

TOOLS = [
    load_h5ad_data,
    extract_embeddings_with_scgpt,
    cluster_and_diff,
    annotate_with_markers,
]
