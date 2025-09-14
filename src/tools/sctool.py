import os

import gget
import gseapy
import numpy as np
import pandas as pd
import scanpy as sc

from biomni.llm import get_llm

###细胞注释
def annotate_celltype_scRNA(
    adata_filename,
    data_dir,
    data_info,
    data_lake_path,
    cluster="leiden",
    llm="claude-3-5-sonnet-20241022",
    composition=None,
):
    """Annotate cell types based on gene markers and transferred labels using LLM.
    After leiden clustering, annotate clusters using differentially expressed genes
    and optionally incorporate transferred labels from reference datasets.

    Parameters
    ----------
    - adata_filename (str): Name of the AnnData file containing scRNA-seq data
    - data_dir (str): Directory containing the data files
    - data_info (str): Information about the scRNA-seq data (e.g., "homo sapiens, brain tissue, normal")
    - data_lake_path (str): Path to the data lake
    - llm (str): Language model instance for cell type prediction, such as 'claude-3-haiku-20240307'
    - composition (pd.DataFrame, optional): Transferred cell type composition for each cluster
    Returns:
    - str: Steps performed and file paths where results were saved

    """

    def _cluster_info(cluster_id, marker_genes, composition_df=None):
        """Format cluster information for LLM prompt."""
        if composition_df is None:
            return f"The enriched genes in this cluster are: {', '.join(marker_genes)}."

        info = [
            f"The enriched genes in this cluster are: {', '.join(marker_genes)}.",
            f"For a starting point, the transferred reference cell type composition {cluster_id} is:",
        ]

        cluster_comp = []
        for celltype, proportion in composition_df.loc[cluster_id].items():
            if proportion > 0:
                cluster_comp.append(f"{celltype}:{proportion:.2f}")

        return "\n".join(info) + " " + "; ".join(cluster_comp) + "\n"

    from langchain_core.prompts import PromptTemplate
    # from langchain.chains import LLMChain

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Identifying marker genes for clusters defined by {cluster} clustering.")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=False)
    genes = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).head(20)
    scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"]).head(20)

    markers = {}
    for i in range(genes.shape[1]):
        gene_names = genes.iloc[:, i].tolist()
        gene_scores = scores.iloc[:, i].tolist()
        markers[i] = list(np.array(gene_names)[np.array(gene_scores) > 0])

    # TODO: this can be optimized
    czi_celltype_path = data_lake_path + "/czi_census_datasets_v4.parquet"
    df = pd.read_parquet(czi_celltype_path)
    czi_celltype_set = {cell_type.strip() for cell_types in df["cell_type"] for cell_type in str(cell_types).split(";")}
    czi_celltype = ", ".join(sorted(czi_celltype_set))

    prompt_template = f"""
Please think carefully, and identify the cell type in {data_info} based on the gene markers.
Optionally refer to the transferred cell type information but do not trust it when the percentage is lower than 0.5.

{{cluster_info}}

The cell type names should come from cell ontology: {czi_celltype}.
Only provide the cell type name, confidence score (0-1), and detailed reason.
Output format: "name; score; reason".
No numbers before name or spaces before number.
"""
    # Some can be a mixture of multiple cell types.

    llm = get_llm(llm)
    prompt = PromptTemplate(input_variables=["cluster_info"], template=prompt_template)
    chain = prompt | llm

    steps.append("Annotating cell types of each cluster based on gene markers and transferred labels.")
    # valid_celltypes = set(czi_celltype.split(";"))
    cluster_annotations = {}
    annotation_reasons = []

    print(f"Annotate each cluster of {cluster}")
    for _idx in range(len(adata.obs[cluster].unique())):
        cluster_info = _cluster_info(str(_idx), markers[_idx], composition)

        while True:
            response = chain.invoke({"cluster_info": cluster_info})

            # Handle different response types
            if hasattr(response, "content"):  # For AIMessage
                response = response.content
            elif isinstance(response, dict) and "text" in response:
                response = response["text"]
            elif isinstance(response, str):
                response = response
            else:
                response = str(response)

            try:
                predicted_celltype, confidence, reason = [x.strip() for x in response.split(";", 2)]
                if predicted_celltype in czi_celltype_set or predicted_celltype.lower() in czi_celltype_set:
                    cluster_annotations[str(_idx)] = predicted_celltype
                    annotation_reasons.append((predicted_celltype, reason))
                    break
                else:
                    cluster_info += "\nAssigned cell type name must be in cell ontology!"
            except ValueError:
                cluster_info += "\nPlease follow the format: name; score; reason"
        print(f"Cluster {_idx}: {response}")

    # create reason dictionary
    reason_dict = {}
    for celltype, reason in annotation_reasons:
        if celltype not in reason_dict:
            reason_dict[celltype] = []
        reason_dict[celltype].append(reason)

    reason_dict = {k: "\n".join(v) for k, v in reason_dict.items()}

    adata.obs["cell_type"] = adata.obs[cluster].map(cluster_annotations)
    adata.obs["cell_type_reason"] = adata.obs["cell_type"].map(reason_dict).astype(str)

    steps.append(f"Saving annotated adata to {data_dir}/annotated.h5ad, the annotations are in the 'cell_type' column.")
    adata.write(f"{data_dir}/annotated.h5ad", compression="gzip")

    return "\n".join(steps)

###用panhumanpy细胞注释
def annotate_celltype_with_panhumanpy(
    adata_path,
    feature_names_col=None,
    refine=True,
    umap=True,
    output_dir="./output",
):
    """Perform hierarchical cell type annotation using panhumanpy and Azimuth Neural Network.

    This function implements the panhumanpy workflow for cell type annotation using the
    Azimuth Neural Network, providing hierarchical cell type labels with confidence scores.

    Parameters
    ----------
    adata_path : str
        Path to the AnnData file containing scRNA-seq data
    feature_names_col : str, optional
        Column name in adata.var containing gene names (default: None, uses index)
    refine : bool, optional
        Whether to perform label refinement for consistent granularity (default: True)
    umap : bool, optional
        Whether to generate ANN embeddings and UMAP (default: True)
    output_dir : str, optional
        Directory to save results (default: "./output")

    Returns
    -------
    str
        Research log summarizing the analysis steps and results

    Notes
    -----
    Performance is not ensured for diseased and/or non-human cells.
    """
    import json
    import shutil
    import subprocess
    import sys
    import tempfile

    def conda_env_exists(env_name):
        try:
            result = subprocess.run(["conda", "env", "list"], capture_output=True, text=True, check=True)
            return any(env_name in line.split() for line in result.stdout.splitlines())
        except Exception:
            return False

    def create_panhumanpy_env(env_name):
        # Create env and install panhumanpy
        subprocess.run(["conda", "create", "-y", "-n", env_name, "python=3.10"], check=True)
        # Install panhumanpy in the new env
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", "git+https://github.com/satijalab/panhumanpy.git"],
            check=True,
        )

    PANHUMANPY_ENV = "panhumanpy_env"

    # 1. Check/create panhumanpy_env
    if not conda_env_exists(PANHUMANPY_ENV):
        create_panhumanpy_env(PANHUMANPY_ENV)

    # 2. Write a temp script to run in the panhumanpy_env
    temp_dir = tempfile.mkdtemp()
    script_path = os.path.join(temp_dir, "run_panhumanpy.py")
    result_path = os.path.join(temp_dir, "result.json")
    with open(script_path, "w") as f:
        f.write(f"""
import os
import sys
import json
import numpy as np
import scanpy as sc
import pandas as pd
try:
    import panhumanpy as ph
except ImportError as e:
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"error": str(e)}}))
    sys.exit(1)

adata_path = r'''{adata_path}'''
feature_names_col = {repr(feature_names_col)}
refine = {refine}
umap = {umap}
output_dir = r'''{output_dir}'''
log = []
try:
    os.makedirs(output_dir, exist_ok=True)
    log.append("# Performing cell type annotation with Panhuman Azimuth")
    log.append(f"Loading object from: {{adata_path}}")
    adata = sc.read_h5ad(adata_path)
    log.append(f"✓ Successfully loaded object with {{adata.n_obs}} cells and {{adata.n_vars}} genes")
    if feature_names_col is None:
        log.append("Using gene names from adata.var.index")
    else:
        log.append(f"Using gene names from column: {{feature_names_col}}")
        if feature_names_col not in adata.var.columns:
            log.append(f"⚠ Warning: Column '{{feature_names_col}}' not found in adata.var")
            log.append(f"Available columns: {{list(adata.var.columns)}}")
            log.append("Falling back to index")
            feature_names_col = None
    if feature_names_col is None:
        azimuth = ph.AzimuthNN(adata)
    else:
        azimuth = ph.AzimuthNN(adata, feature_names_col=feature_names_col)
    cell_metadata = azimuth.cells_meta
    log.append("✓ Successfully annotated all cells")
    if umap:
        log.append("## Generating ANN embeddings")
        try:
            embeddings = azimuth.azimuth_embed()
        except Exception as e:
            log.append(f"✗ Error generating embeddings: {{str(e)}}")
            with open(r'{result_path}', 'w') as out:
                out.write(json.dumps({{"log": log}}))
            sys.exit(0)
        log.append("## Calculating UMAP")
        try:
            azimuth.azimuth_umap()
            log.append("✓ Generated UMAP of ANN embeddings")
        except Exception as e:
            log.append(f"✗ Error generating UMAP: {{str(e)}}")
            with open(r'{result_path}', 'w') as out:
                out.write(json.dumps({{"log": log}}))
            sys.exit(0)
    else:
        log.append("## Skipping embeddings and UMAP generation")
        embeddings = None
        umap = None
    if refine:
        log.append("## Performing label refinement")
        try:
            azimuth.azimuth_refine()
            cell_metadata = azimuth.cells_meta
            refined_columns = [col for col in cell_metadata.columns if col.startswith("azimuth_")]
            log.append(f"✓ Applied label refinement, results are in columns: {{refined_columns}}")
        except Exception as e:
            log.append(f"✗ Error during label refinement: {{str(e)}}")
    log.append("## Saving results")
    try:
        metadata_file = f"{output_dir}/annotated_cell_metadata.csv"
        cell_metadata.to_csv(metadata_file)
        log.append(f"✓ Saved cell metadata to: {{metadata_file}}")
        if umap and embeddings is not None:
            embeddings_file = f"{output_dir}/ann_embeddings.npy"
            np.save(embeddings_file, embeddings)
            log.append(f"✓ Saved embeddings to: {{embeddings_file}}")
            umap_file = f"{output_dir}/ann_umap.npy"
            np.save(umap_file, umap)
            log.append(f"✓ Saved UMAP to: {{umap_file}}")
        else:
            log.append("Skipped saving embeddings and UMAP (umap=False)")
        annotated_save_path = f"{output_dir}/annotated_obj.h5ad"
        azimuth.pack_adata(save_path=annotated_save_path)
        log.append(f"✓ Saved annotated object to: {{annotated_save_path}}")
    except Exception as e:
        log.append(f"✗ Error saving results: {{str(e)}}")
        with open(r'{result_path}', 'w') as out:
            out.write(json.dumps({{"log": log}}))
        sys.exit(0)
    log.append(f"- All results saved to: {output_dir}")
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"log": log}}))
except Exception as e:
    with open(r'{result_path}', 'w') as out:
        out.write(json.dumps({{"error": str(e)}}))
    sys.exit(1)
""")

    # 3. Run the script in the panhumanpy_env
    try:
        run_cmd = ["conda", "run", "-n", PANHUMANPY_ENV, "python", script_path]
        subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(temp_dir)
        return f"Error running panhumanpy in conda env: {e}"

    # 4. Read the result
    try:
        with open(result_path) as f:
            result = json.load(f)
        if "log" in result:
            log = result["log"]
        elif "error" in result:
            log = [f"Error: {result['error']}"]
        else:
            log = ["Unknown error running panhumanpy script."]
    except Exception as e:
        log = [f"Error reading result: {e}"]

    # 5. Clean up temp files
    shutil.rmtree(temp_dir)

    return "\n".join(log)


# === Integration ===


def create_scvi_embeddings_scRNA(adata_filename, batch_key, label_key, data_dir):
    # Import scvi-tools correctly - the package name is still 'scvi' when installed
    try:
        import scvi
    except ImportError:
        return "Please install scvi-tools: pip install scvi-tools"

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Setting up AnnData for scVI with batch key: {batch_key}")
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)

    steps.append("Creating and training scVI model")
    model = scvi.model.SCVI(adata)
    model.train()

    steps.append("Generating latent representation using scVI")
    adata.obsm["X_scVI"] = model.get_latent_representation()

    steps.append(f"Creating scANVI model with label key: {label_key}")
    lvae = scvi.model.SCANVI.from_scvi_model(
        model,
        adata=adata,
        labels_key=label_key,
        unlabeled_category="Unknown",
    )
    steps.append("Training scANVI model")
    lvae.train()

    steps.append("Generating latent representation using scANVI")
    adata.obsm["X_scANVI"] = lvae.get_latent_representation(adata)

    output_filename = f"{data_dir}/scvi_emb_data.h5ad"
    adata.write(output_filename)
    steps.append(f"Saving AnnData with scVI and scANVI embeddings to {output_filename}")

    return "\n".join(steps)


def create_harmony_embeddings_scRNA(adata_filename, batch_key, data_dir):
    # https://pypi.org/project/harmony-pytorch/
    from harmony import harmonize

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    steps.append(f"Running Harmony integration with batch key: {batch_key}")
    adata.obsm["X_harmony"] = harmonize(adata.obsm["X_pca"], adata.obs, batch_key=batch_key)

    output_filename = f"{data_dir}/harmony_emb_data.h5ad"
    steps.append(f"Saving the Harmony embeddings to {output_filename}.")
    adata.write(output_filename)

    return "\n".join(steps)


## TODO: the environment is not ready for this tool
def get_uce_embeddings_scRNA(
    adata_filename,
    data_dir,
    DATA_ROOT="/dfs/project/bioagentos/data/singlecell/",
    custom_args=None,
):
    """The UCE embeddings are usually our default tools to get cell embeddings, we map UCE embeddings to IMA referece dataset and get the cell types for a better understanding.
    The custom_args is a list of strings that will be passed as command line arguments to the UCE script,
    like ["--adata_path", adata_file, "--dir", output_dir]. The default value is None.
    """
    import sys

    steps = []

    try:
        from eval_single_anndata import main, parse_args_uce

        steps.append("Successfully imported UCE main function")
    except Exception:
        steps.append("Please install the UCE package first. Follow https://github.com/snap-stanford/UCE.git.")
        return "\n".join(steps)

    from accelerate import Accelerator

    _base_name = os.path.basename(adata_filename).split(".")[0]
    adata_file_proc = f"{data_dir}/{_base_name}_uce_adata.h5ad"
    if os.path.exists(adata_file_proc):
        steps.append(f"{adata_file_proc} already exists, skipping. The UCE embeddings are stored as adata.obs['X_uce']")
        return "\n".join(steps)

    uce_dir = f"{DATA_ROOT}/UCE"
    sys.path.append(uce_dir)
    steps.append(f"Added {uce_dir} to sys.path")

    # Prepare and parse arguments
    if custom_args is None:
        custom_args = []
    custom_args.extend(["--adata_path", f"{data_dir}/{adata_filename}", "--dir", f"{data_dir}/uce/"])
    parsed_args = parse_args_uce(custom_args)
    steps.append(f"Parsed arguments: {vars(parsed_args)}")

    # Initialize Accelerator
    accelerator = Accelerator(project_dir=parsed_args.dir)
    steps.append("Initialized Accelerator")

    # Run UCE main function
    main(parsed_args, accelerator)
    steps.append("UCE main function completed successfully.")
    steps.append(f"{adata_file_proc} is saved, the UCE embeddings are stored as adata.obs['X_uce']")

    return "\n".join(steps)


def map_to_ima_interpret_scRNA(adata_filename, data_dir, custom_args=None):
    """Map cell embeddings from the input dataset to the Integrated Megascale Atlas reference dataset using UCE embeddings."""
    from sklearn.neighbors import NearestNeighbors

    steps = []
    steps.append(f"Loading AnnData from {data_dir}/{adata_filename}")
    adata = sc.read_h5ad(f"{data_dir}/{adata_filename}")

    if "X_uce" not in adata.obsm:
        raise ValueError("Error: adata.obsm['X_uce'] not found. Please run get_uce_embeddings() first.")

    steps.append("adata.obs['X_uce'] found. Proceeding with cell type mapping.")

    IMA_adata = sc.read_h5ad(f"{data_dir}/uce_10000_per_dataset_33l_8ep_coarse_ct.h5ad")
    steps.append("Loaded Integrated Megascale Atlas (IMA) reference dataset")

    if adata.obsm["X_uce"].shape[1] != IMA_adata.X.shape[1]:
        raise ValueError("UCE embedding dimensions do not match between datasets.")

    # Create a NearestNeighbors object
    n_neighbors = custom_args.get("n_neighbors", 3)
    metric = custom_args.get("metric", "euclidean")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(IMA_adata.X)

    # Find the nearest neighbors for each cell in adata
    distances, indices = nn.kneighbors(adata.obsm["X_uce"])

    # Extract cell types of the nearest neighbors
    mapped_cell_types = IMA_adata.obs["coarse_cell_type_yanay"].iloc[indices.flatten()].values

    # from umap import UMAP
    if n_neighbors > 1:
        # Implement majority voting

        def custom_mode(x):
            unique, counts = np.unique(x, return_counts=True)
            return unique[np.argmax(counts)]

        mapped_cell_types = np.apply_along_axis(custom_mode, 1, mapped_cell_types.reshape(-1, n_neighbors))

    # Add mapped cell types and confidence scores to adata
    adata.obs["mapped_cell_type"] = mapped_cell_types
    adata.obs["mapping_confidence"] = 1 / (1 + distances.mean(axis=1))

    steps.append("Mapped cell types based on nearest neighbors in UCE space")
    steps.append("Mapped cell types and confidence scores added to adata.obs")

    # Generate summary statistics
    mapping_summary = adata.obs["mapped_cell_type"].value_counts().to_dict()
    steps.append(f"Mapping summary: {mapping_summary}")

    # Save the updated adata object
    output_filename = f"{data_dir}/adata_with_mapped_celltypes.h5ad"
    adata.write_h5ad(output_filename, compression="gzip")
    steps.append(f"Updated adata saved to {output_filename}")

    return "\n".join(steps)



    """Given a gene name, this function returns the steps it performs and the max K transcripts-per-million (TPM)
    per tissue from the RNA-seq expression.

    Parameters
    ----------
    - gene_name (str): The gene name for which RNA-seq data is being fetched.
    - K (int): The number of tissues to return. Default is 10.

    Returns
    -------
    - str: The steps performed and the result.

    """
    steps_log = f"Starting RNA-seq data fetch for gene: {gene_name} with K: {K}\n"

    try:
        # Fetch RNA-seq data using gget
        steps_log += "Fetching RNA-seq data using gget.archs4...\n"
        data = gget.archs4(gene_name, which="tissue")

        if data.empty:
            steps_log += f"No RNA-seq data found for the gene {gene_name}.\n"
            return steps_log

        # Create a readable output string
        steps_log += f"RNA-seq expression data for {gene_name} fetched successfully. Formatting the top {K} tissues:\n"
        readable_output = ""
        for index, row in data.iterrows():
            if index < K:
                tissue = row["id"]
                median_tpm = row["median"]
                readable_output += f"\nTissue: {tissue}\n  - Median TPM: {median_tpm}\n"
            else:
                break

        steps_log += readable_output
        return steps_log

    except Exception as e:
        return f"An error occurred: {e}"

###富集分析前准备
def get_gene_set_enrichment_analysis_supported_database_list() -> list:
    return gseapy.get_library_name()

###基因富集分析
def gene_set_enrichment_analysis(
    genes: list,
    top_k: int = 10,
    database: str = "ontology",
    background_list: list = None,
    plot: bool = False,
) -> str:
    """Perform enrichment analysis for a list of genes, with optional background gene set and plotting functionality.

    Parameters
    ----------
    - genes (list): List of gene symbols to analyze.
    - top_k (int): Number of top pathways to return. Default is 10.
    - database (str): User-friendly name of the database to use for enrichment analysis.
        Popular options include:
        - 'pathway'      (KEGG_2021_Human)
        - 'transcription'   (ChEA_2016)
        - 'ontology'     (GO_Biological_Process_2021)
        - 'diseases_drugs'  (GWAS_Catalog_2019)
        - 'celltypes'     (PanglaoDB_Augmented_2021)
        - 'kinase_interactions' (KEA_2015)
        You can use get_gene_set_enrichment_analysis_supported_database_list tool to get the list of supported databases.

    - background_list (list, optional): List of background genes to use for enrichment analysis.
    - plot (bool, optional): If True, generates a bar plot of the top K enrichment results.

    Returns
    -------
    - str: The steps performed and the top K enrichment results.

    """
    steps_log = (
        f"Starting enrichment analysis for genes: {', '.join(genes)} using {database} database and top_k: {top_k}\n"
    )

    if background_list:
        steps_log += f"Using background list with {len(background_list)} genes.\n"

    try:
        # Perform enrichment analysis with or without background list
        steps_log += f"Performing enrichment analysis using gget.enrichr with the {database} database...\n"
        df = gget.enrichr(genes, database=database, background_list=background_list, plot=plot)

        # Limit to top K results
        steps_log += f"Filtering the top {top_k} enrichment results...\n"
        df = df.head(top_k)

        # Format the result
        output_str = ""
        for _idx, row in df.iterrows():
            output_str += (
                f"Rank: {row['rank']}\n"
                f"Path Name: {row['path_name']}\n"
                f"P-value: {row['p_val']:.2e}\n"
                f"Z-score: {row['z_score']:.6f}\n"
                f"Combined Score: {row['combined_score']:.6f}\n"
                f"Overlapping Genes: {', '.join(row['overlapping_genes'])}\n"
                f"Adjusted P-value: {row['adj_p_val']:.2e}\n"
                f"Database: {row['database']}\n"
                "----------------------------------------\n"
            )

        steps_log += output_str

        return steps_log

    except Exception as e:
        return f"An error occurred: {e}"

