import os
import torch
import json

import scanpy as sc
import pandas as pd

from src.bio_pretrained_model.data_prep._scgpt_data_processor import ScGPTDataProcessor
from src.bio_pretrained_model.scgpt._scgpt_model import ScGPTModelWrapper


class ScgptAnno:
    def __init__(self, 
                 adata_path: str, 
                 model_path: str,
                 output_dir: str, 
                 device='cuda' ):

        self.adata_path = adata_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device

    def run_scgpt_inference(self, **kwargs):

        processor = ScGPTDataProcessor(
            raw_adata_file_name=self.adata_path,
            is_count_raw_data=True
        )

        adata_preprocessed = processor.preprocess_data(
            gene_vocab=os.path.join(self.model_path, "vocab.json"),
            output_dir=self.output_dir,
            use_raw=True,
            n_hvg=1200,
            gene_col="gene_name",
        )

        base_file_name = os.path.splitext(os.path.basename(self.adata_path))[0]

        model = ScGPTModelWrapper.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            device=self.device,
        )

        adata_emb = model.extract_sample_embedding(
            adata_or_file=adata_preprocessed,
            gene_col="gene_name",  # 如果报错可以换成 'index'
            max_length=1200,
            cell_embedding_mode="cls",
            batch_size=64,
            obs_to_save=None,
            return_new_adata=True,
        )

        adata_preprocessed.obsm["X_scgpt"] = adata_emb.X.copy()

        embedding_output_path = os.path.join(self.output_dir, f"{base_file_name}_with_embedding.h5ad")
        adata_preprocessed.write_h5ad(embedding_output_path)

        return adata_preprocessed

    def run_umap_and_diff_expr(self, adata_preprocessed,**kwargs):

        adata = adata_preprocessed

        sc.pp.neighbors(adata, use_rep="X_scgpt", n_neighbors=15)
        sc.tl.leiden(adata, key_added="scGPT_clusters", resolution=0.5, flavor="igraph", n_iterations=2, directed=False)

        sc.tl.umap(adata)
        sc.pl.umap(adata, color="scGPT_clusters", save="_umap_scgpt_clustered.png", show=False)

        adata.var_names = adata.var["gene_name"].astype(str)
        adata.var_names_make_unique()

        sc.tl.rank_genes_groups(
            adata,
            groupby="scGPT_clusters",
            method="wilcoxon"
        )
        sc.pl.rank_genes_groups(
            adata,
            n_genes=5,
            sharey=False,
            save="_marker_genes_scgpt.png",
            show=False
        )

        result = adata.uns["rank_genes_groups"]
        groups = result["names"].dtype.names

        marker_df = pd.DataFrame({group: result["names"][group][:self.diff_gene_top_k] for group in groups})
        marker_df = marker_df.melt(var_name="group", value_name="names")

        # Using Celltype Markergene
        if self.marker_reference_path:
            reference_df = pd.read_csv(self.marker_reference_path)
            reference_grouped = reference_df.groupby("celltype")["markergene"].apply(set).to_dict()
            # Add an overlap rating column
            def compute_overlap_score(g):
                candidate = set(marker_df[marker_df["group"] == g]["names"].tolist())
                overlaps = {ct: len(candidate & ref_genes) for ct, ref_genes in reference_grouped.items()}
                return sorted(overlaps.items(), key=lambda x: x[1], reverse=True)

            marker_df["matched_celltype"] = marker_df["group"].apply(lambda g: compute_overlap_score(g)[:3])

        preprocessed_base = os.path.splitext(os.path.basename(self.adata_path))[0]
        marker_gene_path = os.path.join(self.output_dir, f"{preprocessed_base}_marker_genes.csv")
        marker_df.to_csv(marker_gene_path, index=False)

        for group in groups:
            top_genes = result["names"][group][:5]
            print(f"Cluster {group}: {', '.join(top_genes)}")

        final_path = os.path.join(self.output_dir, "c_data_with_clusters_annotated_final.h5ad")
        adata.var.index.name = None
        adata.write_h5ad(final_path)

        return  adata, marker_df



