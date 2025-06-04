import os 
import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def split_cima_data(data_path, output_dir):
    """
    Split the original CIMA data based on celltypel3.
    """
    adata = sc.read_h5ad(data_path)
    adata = adata.to_memory()  # Ensure we have enough memory to work with

    l3_dir = Path(output_dir) / "split_by_l3"
    l3_dir.mkdir(exist_ok=True)

    for ct in tqdm(adata.obs['celltype_l3'].unique(),desc = "Saving per-l3"):
        adata_ct = adata[adata.obs['celltype_l3'] == ct]
        ct_safe = ct.replace("/", "_").replace(" ", "_")
        adata_ct.write_h5ad(l3_dir / f"CIMA_{ct_safe}.h5ad")
    
    print(f"Data split by celltype_l3 saved to {l3_dir}")

    return None

def prepare_cima_from_multiple_files_global_sampling(
    input_dir: str,
    output_dir: str,
    test_fraction: float = 0.1,
    downsample_total: int = 500_000,
    alpha: float = 0.5,
    rare_threshold: int = 5000,
    max_cells_per_class: int = 50000,
    seed: int = 42
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_obs = []
    for file in tqdm(os.listdir(input_dir), desc="Collect cellular metadata"):
        if not file.endswith(".h5ad"): continue
        file_path = os.path.join(input_dir, file)
        adata = sc.read_h5ad(file_path, backed="r")
        df = adata.obs[["final_annotation"]].copy()
        df["source_file"] = file
        df["orig_index"] = df.index
        all_obs.append(df)
        adata.file.close()
    obs_df = pd.concat(all_obs).reset_index(drop=True)

    test_indices = []
    for ct in obs_df["final_annotation"].unique():
        ct_indices = obs_df[obs_df["final_annotation"] == ct].index
        n_test = max(1, int(len(ct_indices) * test_fraction))
        test_indices.extend(rng.choice(ct_indices, size=n_test, replace=False))
    obs_df["set"] = "train_candidate"
    obs_df.loc[test_indices, "set"] = "test"

    train_df = obs_df[obs_df["set"] == "train_candidate"]
    counts = train_df["final_annotation"].value_counts().reset_index()
    counts.columns = ["final_annotation", "count"]
    counts["adjusted"] = counts["count"] ** alpha
    adjusted_total = counts["adjusted"].sum()
    counts["initial_sample"] = (counts["adjusted"] / adjusted_total * downsample_total).astype(int)

    def final_sample(row):
        if row["count"] < rare_threshold:
            return row["count"]
        elif row["initial_sample"] > max_cells_per_class:
            return max_cells_per_class
        else:
            return row["initial_sample"]
    counts["final_sample"] = counts.apply(final_sample, axis=1)

    counts.to_csv(os.path.join(output_dir, "stratified_sampling_plan_final.csv"), index=False)

    final_train_indices = []
    for _, row in counts.iterrows():
        ct = row["final_annotation"]
        n = row["final_sample"]
        ct_indices = train_df[train_df["final_annotation"] == ct].index
        final_train_indices.extend(rng.choice(ct_indices, size=n, replace=False))

    obs_df["final_set"] = "unused"
    obs_df.loc[obs_df["set"] == "test", "final_set"] = "test"
    obs_df.loc[final_train_indices, "final_set"] = "train"

    train_adatas, test_adatas = [], []
    for file in tqdm(obs_df["source_file"].unique(), desc="Extract training and testing sets"):
        file_obs = obs_df[obs_df["source_file"] == file]
        if not any(file_obs["final_set"].isin(["train", "test"])):
            continue
        file_path = os.path.join(input_dir, file)
        adata = sc.read_h5ad(file_path)
        adata.obs["orig_index"] = adata.obs.index
        merged = adata.obs.merge(file_obs, on="orig_index", how="inner")
        adata = adata[merged.index]
        adata.obs = merged
        if "train" in merged["final_set"].values:
            train_adatas.append(adata[adata.obs["final_set"] == "train"].copy())
        if "test" in merged["final_set"].values:
            test_adatas.append(adata[adata.obs["final_set"] == "test"].copy())

    adata_train = sc.concat(train_adatas)
    adata_test = sc.concat(test_adatas)
    adata_train.write(os.path.join(output_dir, "cima_train_vector_db_final.h5ad"))
    adata_test.write(os.path.join(output_dir, "cima_test_set_final.h5ad"))

    print("CIMA data preparation completed")

if __name__ == "__main__":
    input_dir = ""
    split_output_dir = ""
    downsampling_output_dir = ""
    
    split_cima_data(os.path.join(input_dir, "cima_original.h5ad"), split_output_dir)
    
    prepare_cima_from_multiple_files_global_sampling(
        input_dir=split_output_dir,
        output_dir=downsampling_output_dir
    )