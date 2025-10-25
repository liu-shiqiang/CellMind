from src.tools.validata_input import validate_input_file
from src.tools.load_h5ad import load_h5ad_data
from src.tools.extract_embeddings import extract_embeddings_with_scgpt
from src.tools.retrieve_cell_context import retrieve_cell_context
from src.tools.clustering_marker import cluster_and_rank_markers
from src.tools.annotate_with_markers import annotate_with_markers
from src.tools.cluster_diff import cluster_and_diff
from src.tools.annotate_with_cellrag import annotate_with_cellrag
from src.tools.interpret_cluster_results import interpret_cluster_results
from src.tools.interpret_celltype_results import interpret_celltype_results
from src.tools.dataset_qa import dataset_bio_qa
from src.tools.enrichment_analysis.ora import run_ora_enrichment
from src.tools.enrichment_analysis.ssgsea import run_ssgsea_enrichment
from src.tools.cellphoneDB import run_cellphonedb_core
from src.tools.pseudotime_analysis import run_pseudotime_analysis

TOOLS = [
    load_h5ad_data,
    extract_embeddings_with_scgpt,
    cluster_and_diff,
    annotate_with_markers,
    interpret_cluster_results,
    interpret_celltype_results,
    dataset_bio_qa,
    run_ssgsea_enrichment,
    run_cellphonedb_core,
    run_pseudotime_analysis,
]
