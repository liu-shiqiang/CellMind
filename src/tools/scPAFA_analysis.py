import os
import json
import logging
import pandas as pd
import scanpy as sc
from pathlib import Path
from typing import Optional, List, Dict, Any
import re

# ==================== Dependencies ====================
try:
    import scPAFA
    import mofapy2
    import mofax
except ImportError as e:
    raise ImportError("Required packages not installed. Please install: scPAFA, mofapy2, mofax") from e
# =====================================================

PATHWAY_DICT = "/home/share/huadjyin/home/zhangzilin/genomix-agent/test/pathwaydict_bioplanet.json"

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _is_likely_technical_column(col_name: str) -> bool:
    col_lower = col_name.lower()
    clinical_whitelist = {"total_dose", "total_time", "score_clinical", "score_response"}
    if col_name in clinical_whitelist or any(w in col_lower for w in ["dose", "time", "clinical", "response"]):
        return False
    technical_patterns = [
        r"^n_", r"^pct_", r"^percent_", r"mito", r"^mt-", r"ribo", r"doublet",
        r"qc_", r"_qc$", r"score$", r"umi", r"reads", r"total_", r"detected",
        r"log10", r"library", r"mapping",
    ]
    return any(re.search(pattern, col_lower) for pattern in technical_patterns)


def load_sample_metadata(
    adata: sc.AnnData,
    sample_column: str,
    sample_metadata_path: Optional[str] = None,
    required_columns: Optional[List[str]] = None,
    max_unique_ratio: float = 0.8,
    min_non_missing_ratio: float = 0.5,
    filter_technical_from_obs: bool = True,
) -> pd.DataFrame:
    required_columns = set(required_columns or [])
    if sample_metadata_path is None:
        logger.info("Using adata.obs as sample metadata...")
        sample_meta = adata.obs.groupby(sample_column, observed=True).first()
        if filter_technical_from_obs:
            cols_to_keep = [col for col in sample_meta.columns
                            if col in required_columns or not _is_likely_technical_column(col)]
            sample_meta = sample_meta[cols_to_keep]
            logger.info(f"After filtering, {len(cols_to_keep)} metadata columns remain.")
    else:
        logger.info(f"Loading external sample metadata from {sample_metadata_path}...")
        path = Path(sample_metadata_path)
        if path.suffix == ".csv":
            sample_meta = pd.read_csv(path, index_col=0)
        elif path.suffix in (".tsv", ".txt"):
            sample_meta = pd.read_csv(path, sep="\t", index_col=0)
        else:
            raise ValueError("sample_metadata_path must be .csv or .tsv")
        if sample_column in sample_meta.columns:
            sample_meta = sample_meta.set_index(sample_column, drop=True)
        elif sample_meta.index.name != sample_column:
            raise ValueError(
                f"Metadata must have '{sample_column}' as index or column. "
                f"Current index: {sample_meta.index.name}, columns: {list(sample_meta.columns)}"
            )

    adata_samples = adata.obs[sample_column].dropna().unique()
    sample_meta = sample_meta.loc[sample_meta.index.intersection(adata_samples)]
    if sample_meta.empty:
        raise ValueError("No overlapping samples between AnnData and metadata.")

    candidate_cols = [col for col in sample_meta.columns if col not in required_columns]
    n_samples = len(sample_meta)
    selected_cols = list(required_columns)
    for col in candidate_cols:
        series = sample_meta[col].dropna()
        if len(series) / n_samples < min_non_missing_ratio:
            continue
        if series.nunique() <= 1:
            continue
        if series.nunique() / n_samples > max_unique_ratio:
            continue
        if series.dtype == "object":
            try:
                avg_len = series.astype(str).str.len().mean()
                if avg_len > 30:
                    continue
            except Exception:
                pass
        selected_cols.append(col)

    final_meta = sample_meta[selected_cols].copy()
    logger.info(f"Final clinical metadata columns: {list(final_meta.columns)}")
    missing_required = required_columns - set(final_meta.columns)
    if missing_required:
        raise ValueError(f"Required columns missing: {sorted(missing_required)}")
    return final_meta


def _compute_pas_with_scpafa(
    adata: sc.AnnData,
    pathways: Dict[str, List[str]],
    min_overlap_gene: int = 6,
    max_rank: int = 2000,
    n_cores_rank: int = 10,
    n_cores_score: int = 10,
    rank_batch_size: int = 100000,
    score_batch_size: int = 100000,
) -> pd.DataFrame:
    pathway_input = scPAFA.tl.generate_pathway_input(
        adata=adata,
        pathway_dict=pathways,
        min_overlap_gene=min_overlap_gene
    )
    rank_matrix = scPAFA.tl.fast_ucell_rank(
        adata=adata,
        maxRank=max_rank,
        n_cores_rank=n_cores_rank,
        rank_batch_size=rank_batch_size
    )
    pas_df = scPAFA.tl.fast_ucell_score(
        cell_index=list(adata.obs.index),
        rankmatrix=rank_matrix,
        maxRank=max_rank,
        n_cores_score=n_cores_score,
        score_batch_size=score_batch_size,
        input_dict=pathway_input
    )
    return pas_df


def run_pas_mofa_pipeline(
    adata_path: str,
    sample_column: str,
    view_column: str,
    label_column: str,
    sample_metadata_path: Optional[str] = None,
    batch_column: Optional[str] = None,
    pas_method: str = "UCell",
    factor_number: int = 10,
    min_cells_per_sample_view: int = 3,
    output_dir: str = "./pas_mofa_results",
    random_seed: int = 42,
    max_rank: int = 2000,
    standardize_gene_names: bool = True,
) -> Dict[str, Any]:
    # --- Validation ---
    adata_path = Path(adata_path)
    pathway_dict_path = Path(PATHWAY_DICT)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    if not pathway_dict_path.exists():
        raise FileNotFoundError(f"Pathway JSON file not found: {pathway_dict_path}")
    if sample_metadata_path:
        sample_metadata_path = Path(sample_metadata_path)
        if not sample_metadata_path.exists():
            raise FileNotFoundError(f"Sample metadata file not found: {sample_metadata_path}")

    adata = sc.read_h5ad(adata_path)
    if sample_column not in adata.obs.columns:
        raise ValueError(f"'{sample_column}' not in adata.obs.")
    if view_column not in adata.obs.columns:
        raise ValueError(f"'{view_column}' not in adata.obs.")

    # --- Load clinical metadata ---
    required_meta_cols = [label_column]
    if batch_column:
        required_meta_cols.append(batch_column)
    clinical_metadata = load_sample_metadata(
        adata=adata,
        sample_column=sample_column,
        sample_metadata_path=sample_metadata_path,
        required_columns=required_meta_cols,
    )

    # --- Step 1: Pathway Activity Scoring (PAS) ---
    logger.info("Running Pathway Activity Scoring with scPAFA UCell...")
    with open(pathway_dict_path) as f:
        pathways = json.load(f)
    if not isinstance(pathways, dict):
        raise ValueError("Pathway file must be a JSON dict.")
    if standardize_gene_names:
        adata.var_names = adata.var_names.str.upper()
        pathways = {k: [g.upper() for g in v] for k, v in pathways.items()}

    if pas_method != "UCell":
        raise ValueError("Only 'UCell' is supported in this version.")
    pas_df = _compute_pas_with_scpafa(
        adata=adata,
        pathways=pathways,
        min_overlap_gene=6,
        max_rank=max_rank,
        n_cores_rank=10,
        n_cores_score=10,
        rank_batch_size=100000,
        score_batch_size=100000,
    )
    logger.info(f"PAS matrix shape: {pas_df.shape} (cells × pathways)")

    # --- Step 2: Pseudobulk using scPAFA (with batch support) ---
    logger.info("Aggregating pseudobulk using scPAFA...")
    pseudobulk_long = scPAFA.pb.generate_scpafa_input_multigroup(
        metadata=adata.obs,
        PAS_dataframe=pas_df,
        sample_column=sample_column,
        view_column=view_column,
        group_column=batch_column,  # ← batch used for filtering
        min_cell_number_per_sample=min_cells_per_sample_view,
        min_percentage_sample_per_view=0.1,
        min_sample_per_view=2,
        top_percentage=1.0
    )

    pseudobulk_df = pseudobulk_long.pivot_table(
        index="sample", columns="feature", values="value"
    )
    sample_view_to_sample = dict(zip(pseudobulk_long["sample"], pseudobulk_long["sample"]))
    logger.info(f"Pseudobulk matrix shape: {pseudobulk_df.shape}")

    # --- Step 3: MOFA+ ---
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    mofa_model_path = output_dir / "mofa_model.hdf5"
    n_samples = len(pseudobulk_df)
    K = min(factor_number, n_samples - 1)
    if K < 1:
        raise ValueError(f"Not enough samples ({n_samples}) for MOFA+ with factor_number={factor_number}")

    logger.info(f"Training MOFA+ with K={K} factors...")
    ent = mofapy2.run.entry_point.EntryPoint()
    ent.set_data_options(scale_views=False, scale_groups=False)
    ent.set_data_matrix(
        views=[pseudobulk_df.values.T],
        samples_names=[pseudobulk_df.index.tolist()],
        features_names=[pseudobulk_df.columns.tolist()],
        view_names=["pathway_activity"]
    )
    ent.set_model_options(factors=K)
    ent.set_train_options(iter=1000, seed=random_seed, verbose=False)
    ent.build()
    ent.run()
    ent.save(str(mofa_model_path))
    logger.info(f"MOFA+ model saved to {mofa_model_path}")

    # Save pseudobulk
    pseudobulk_path = output_dir / "pseudobulk.csv"
    pseudobulk_df.to_csv(pseudobulk_path)

    # Load factors and annotate with metadata
    mfx_model = mofax.mofa_model(mofa_model_path)
    factors_df = mfx_model.factors  # index: sample_view
    factors_df = factors_df.reset_index().rename(columns={"index": "sample_view"})
    factors_df["sample_id"] = factors_df["sample_view"].map(sample_view_to_sample)

    # Add label and batch info
    meta_for_factors = clinical_metadata.reindex(factors_df["sample_id"])
    factors_df = pd.concat([factors_df.set_index("sample_id"), meta_for_factors], axis=1)
    factors_df = factors_df.reset_index().rename(columns={"index": "sample_id"})

    factors_path = output_dir / "factors.csv"
    factors_df.to_csv(factors_path, index=False)

    # --- Step 4: Visualization ---
    figures_dir = None
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        plot_df = factors_df.set_index("sample_id")
        if label_column in plot_df.columns:
            n_factors = sum(col.startswith("factor") for col in plot_df.columns)
            ncols = 3
            nrows = (n_factors + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
            axes = axes.flatten()
            factor_cols = [col for col in plot_df.columns if col.startswith("factor")]
            for i, factor in enumerate(factor_cols):
                ax = axes[i]
                sns.boxplot(data=plot_df, x=label_column, y=factor, ax=ax)
                ax.set_title(f"{factor} vs {label_column}")
                ax.tick_params(axis='x', rotation=45)
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            plt.savefig(figures_dir / "factor_vs_label_boxplot.png", dpi=150)
            plt.close()

        # Heatmap
        heatmap_data = plot_df[factor_cols]
        if label_column in plot_df.columns:
            label_colors = plot_df[label_column].astype("category")
            lut = dict(zip(label_colors.cat.categories, sns.color_palette("Set2", len(label_colors.cat.categories))))
            row_colors = label_colors.map(lut)
        else:
            row_colors = None
        g = sns.clustermap(
            heatmap_data,
            row_colors=row_colors,
            cmap="RdBu_r",
            center=0,
            figsize=(10, max(6, 0.2 * heatmap_data.shape[0])),
            yticklabels=False,
            xticklabels=True,
            cbar_kws={'label': 'Factor activity'}
        )
        plt.savefig(figures_dir / "factor_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Top pathways
        loadings = mfx_model.weights
        for factor in loadings.columns:
            top_pathways = loadings[factor].abs().sort_values(ascending=False).head(10).index
            top_vals = loadings.loc[top_pathways, factor]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=top_vals.values, y=top_vals.index, ax=ax, palette="viridis")
            ax.set_title(f"Top pathways driving {factor}")
            ax.set_xlabel("Loading weight")
            plt.tight_layout()
            plt.savefig(figures_dir / f"{factor}_top_pathways.png", dpi=150)
            plt.close()

        logger.info(f"Visualizations saved to {figures_dir}")

    except Exception as e:
        logger.warning(f"Visualization failed (non-fatal): {e}")

    return {
        "mofa_model_path": str(mofa_model_path),
        "factors_path": str(factors_path),
        "pseudobulk_path": str(pseudobulk_path),
        "figures_dir": str(figures_dir) if figures_dir else None,
        "sample_metadata_used_from_adata": sample_metadata_path is None,
        "n_samples": n_samples,
        "n_factors": K,
        "output_dir": str(output_dir),
    }