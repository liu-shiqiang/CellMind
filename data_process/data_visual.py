import os
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc

def visualize_storage_and_test(storage_path,test_path,output_dir,level):

    """
    Visualizes the distribution of cell types in the storage and test datasets.

    """

    os.makedirs(output_dir, exist_ok=True)

    adata_train = sc.read_h5ad(storage_path)
    adata_test = sc.read_h5ad(test_path)

    print(adata_train.obs.columns.tolist())

    train_counts = adata_train.obs[level].value_counts().sort_values(ascending=False)
    test_counts = adata_test.obs[level].value_counts().sort_values(ascending=False)

    df_counts = pd.DataFrame({
        "storage_count": train_counts,
        "test_count": test_counts
    }).fillna(0).astype(int)

    csv_path = os.path.join(output_dir, level + "_train_test.csv")
    df_counts.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, len(df_counts) * 0.25))
    df_counts.sort_values("storage_count", ascending=True).plot.barh(ax=ax)
    plt.title("Cell counts per celltype_l3 in Storage and Test sets")
    plt.xlabel("Number of Cells")
    plt.ylabel("Cell Type (celltype_l3)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, level + "_barplot.png"))
    plt.close()

    return None

if __name__ == "__main__":

    """
        View the distribution of data at different levels in storage and testing
    """
    # storage_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/cima_train_vector_db_final.h5ad"
    # test_path = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/cima_test_set_final.h5ad"
    # level = "celltype_l3"

    # output_dir = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/data/cell_type/CIMA_source_data/output/visualization"
    # visualize_storage_and_test(storage_path, test_path, output_dir, level)
    # print("Visualization completed successfully.")
