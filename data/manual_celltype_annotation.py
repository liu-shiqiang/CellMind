import scanpy as sc
import pandas as pd
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict
import uuid

def single_cell_cluster_annotation(data_path, marker_csv_path, output_dir, cluster_key, celltype_layer):
    """
    Perform single-cell cluster annotation, generate visualizations, and analyze clusters with shared cell types.
    
    Parameters:
    - data_path: Path to the h5ad file containing single-cell data
    - marker_csv_path: Path to the CSV file with marker genes
    - output_dir: Directory to save output files (figures, CSVs, etc.)
    - cluster_key: Column name in adata.obs for clustering results (e.g., 'leiden_res_2.70', 'leiden_res_1.80', 'leiden_res_0.05')
    - celltype_layer: Column name in marker_df for cell type (e.g., 'CIMA_l3', 'CIMA_l2')
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    adata = sc.read_h5ad(data_path)
    marker_df = pd.read_csv(marker_csv_path)
    marker_df["markergene"] = marker_df["markergene"].str.split(",").apply(lambda genes: [g.strip() for g in genes])
    marker_genes = dict(zip(marker_df[celltype_layer], marker_df["markergene"]))

    # Filter marker genes to those present in adata.var_names
    filtered_marker_genes = {
        ct: [g for g in genes if g in adata.var_names]
        for ct, genes in marker_genes.items()
        if any(g in adata.var_names for g in genes)
    }

    # Generate and save dotplot
    dotplot_save_path = os.path.join(output_dir, f"marker_dotplot_{cluster_key}.png")
    os.makedirs(os.path.dirname(dotplot_save_path), exist_ok=True)  # Ensure directory exists
    sc.pl.dotplot(
        adata,
        filtered_marker_genes,
        groupby=cluster_key,
        standard_scale="var",
        save=f"marker_dotplot_{cluster_key}.png"  # Only filename, path handled manually
    )
    # Move the saved file to the correct location
    default_save_path = f"figures/dotplot_marker_dotplot_{cluster_key}.png"
    if os.path.exists(default_save_path):
        os.makedirs(os.path.dirname(dotplot_save_path), exist_ok=True)
        os.rename(default_save_path, dotplot_save_path)

    # Perform differential gene expression analysis
    sc.tl.rank_genes_groups(adata, groupby=cluster_key, method="wilcoxon")
    diff_genes_df = sc.get.rank_genes_groups_df(adata, group=None)

    def manual_cluster_annotation(adata, marker_df, diff_genes_df, cluster_key, celltype_layer):
        cluster_to_celltype = {}
        cluster_to_markers = {}
        
        for cluster in range(adata.obs[cluster_key].nunique()):
            cluster_str = str(cluster)
            mask = adata.obs[cluster_key] == cluster_str
            cluster_adata = adata[mask].copy()
            
            # Get top 20 differential genes
            cluster_diff_genes = diff_genes_df[diff_genes_df['group'] == cluster_str]['names'].tolist()[:20]
            print(f"Cluster {cluster} differential genes:", cluster_diff_genes)
            
            # Match marker genes
            best_match = None
            max_overlap = 0
            for idx, row in marker_df.iterrows():
                overlap = len(set(cluster_diff_genes) & set(row['markergene']))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = row[celltype_layer]
            
            suggested_type = best_match if max_overlap > 0 else 'Unknown'
            suggested_markers = filtered_marker_genes.get(suggested_type, [])
            suggested_markers_str = ", ".join(suggested_markers) if suggested_markers else "N/A"
            
            print(f"Suggested cell type for Cluster {cluster}: {suggested_type} (overlap: {max_overlap})")
            print(f"Marker genes for {suggested_type}: {suggested_markers_str}")
            final_type = input(f"Final cell type for Cluster {cluster} (press Enter for {suggested_type}, or enter new type): ")
            cluster_to_celltype[cluster] = final_type if final_type else suggested_type
            cluster_to_markers[cluster] = cluster_diff_genes
        
        return cluster_to_celltype, cluster_to_markers

    # Run manual annotation
    cluster_to_celltype, cluster_to_markers = manual_cluster_annotation(adata, marker_df, diff_genes_df, cluster_key, celltype_layer)

    # Save cluster to cell type mappings
    celltype_df = pd.DataFrame.from_dict(cluster_to_celltype, orient='index', columns=['cell_type'])
    celltype_df.index.name = 'cluster'
    celltype_df.to_csv(os.path.join(output_dir, f"cluster_to_celltype_{cluster_key}.csv"))

    # Save top 20 differential genes per cluster
    markers_df = pd.DataFrame.from_dict(
        {k: ", ".join(v) for k, v in cluster_to_markers.items()},
        orient='index',
        columns=['top_20_diff_genes']
    )
    markers_df.index.name = 'cluster'
    markers_df.to_csv(os.path.join(output_dir, f"cluster_diff_genes_{cluster_key}.csv"))

    # Add manual cell type annotations to adata
    adata.obs['manual_celltype'] = adata.obs[cluster_key].map(lambda x: cluster_to_celltype.get(int(x), 'Unknown'))

    # Generate and save UMAP plot
    umap_save_path = os.path.join(output_dir, f"umap_manual_celltype_{cluster_key}.png")
    os.makedirs(os.path.dirname(umap_save_path), exist_ok=True)  # Ensure directory exists
    sc.pl.umap(
        adata,
        color='manual_celltype',
        title=f'Manual Cell Type Annotations ({cluster_key})',
        save=f"umap_manual_celltype_{cluster_key}.png"  # Only filename
    )
    # Move the saved file to the correct location
    default_umap_path = f"figures/umap_manual_celltype_{cluster_key}.png"
    if os.path.exists(default_umap_path):
        os.makedirs(os.path.dirname(umap_save_path), exist_ok=True)
        os.rename(default_umap_path, umap_save_path)

    # Identify cell types with multiple clusters
    celltype_to_clusters = defaultdict(list)
    for cluster, celltype in cluster_to_celltype.items():
        celltype_to_clusters[celltype].append(str(cluster))
    
    multi_cluster_celltypes = {ct: clusters for ct, clusters in celltype_to_clusters.items() if len(clusters) > 1}
    
    # Save cell types with multiple clusters
    multi_cluster_df = pd.DataFrame.from_dict(
        {ct: ", ".join(clusters) for ct, clusters in multi_cluster_celltypes.items()},
        orient='index',
        columns=['clusters']
    )
    multi_cluster_df.index.name = 'cell_type'
    multi_cluster_df.to_csv(os.path.join(output_dir, f"multi_cluster_celltypes_{cluster_key}.csv"))

    # Compare differential genes for clusters with shared cell types
    for celltype, clusters in multi_cluster_celltypes.items():
        diff_genes_dfs = {}
        for cluster in clusters:
            diff_genes_dfs[cluster] = sc.get.rank_genes_groups_df(adata, group=cluster).head(10)
        
        # Combine genes for plotting
        genes_to_plot = list(set().union(*[df['names'] for df in diff_genes_dfs.values()]))

        # Save differential genes for these clusters
        for cluster, df in diff_genes_dfs.items():
            df.to_csv(os.path.join(output_dir, f"cluster_{cluster}_top10_diff_genes_{cluster_key}.csv"), index=False)

        # Subset adata for cerlusters of interest
        adata_subset = adata[adata.obs[cluster_key].isin(clusters)].copy()

        # Generate and save dotplot for compared clusters
        dotplot_comparison_path = os.path.join(output_dir, f"cluster_comparison_dotplot_{celltype}_{cluster_key}.png")
        os.makedirs(os.path.dirname(dotplot_comparison_path), exist_ok=True)  # Ensure directory exists
        sc.pl.dotplot(
            adata_subset,
            var_names=genes_to_plot,
            groupby=cluster_key,
            standard_scale='var',
            save=f"cluster_comparison_dotplot_{celltype}_{cluster_key}.png"  # Only filename
        )
        # Move the saved file to the correct location
        default_dotplot_path = f"figures/dotplot_cluster_comparison_dotplot_{celltype}_{cluster_key}.png"
        if os.path.exists(default_dotplot_path):
            os.makedirs(os.path.dirname(dotplot_comparison_path), exist_ok=True)
            os.rename(default_dotplot_path, dotplot_comparison_path)

        # Generate and save UMAP for compared clusters
        umap_comparison_path = os.path.join(output_dir, f"umap_cluster_comparison_{celltype}_{cluster_key}.png")
        os.makedirs(os.path.dirname(umap_comparison_path), exist_ok=True)  # Ensure directory exists
        sc.pl.umap(
            adata,
            color=[cluster_key],
            groups=clusters,
            title=f'Clusters for {celltype} ({cluster_key})',
            save=f"umap_cluster_comparison_{celltype}_{cluster_key}.png"  # Only filename
        )



        
        # Move the saved file to the correct location
        default_umap_comparison_path = f"figures/umap_cluster_comparison_{celltype}_{cluster_key}.png"
        if os.path.exists(default_umap_comparison_path):
            os.makedirs(os.path.dirname(umap_comparison_path), exist_ok=True)
            os.rename(default_umap_comparison_path, umap_comparison_path)

    return adata, cluster_to_celltype, cluster_to_markers, multi_cluster_celltypes

if __name__ == "__main__":
    # Example usage
    data_path = "/home/share/huadjyin/home/zhaodanning/genomix-agent/data/clustered.h5ad"
    marker_csv_path = "/home/share/huadjyin/home/zhaodanning/genomix-agent/data/newest_CIMA_l2.csv"
    output_dir = "./output"
    cluster_key = "leiden_res_1.80"  # Can be changed to 'leiden_res_1.80' or 'leiden_res_0.20'
    celltype_layer = "CIMA_l2"  # Can be changed to 'CIMA_l2'
    adata, cluster_to_celltype, cluster_to_markers, multi_cluster_celltypes = single_cell_cluster_annotation(
        data_path, marker_csv_path, output_dir, cluster_key, celltype_layer
    )
    print("Final cell type annotations:", cluster_to_celltype)
    print("Top 20 differential genes per cluster:", cluster_to_markers)
    print("Cell types with multiple clusters:", multi_cluster_celltypes)