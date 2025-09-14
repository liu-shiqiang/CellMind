import subprocess
from pathlib import Path

def run_pyscenic(
        expr_loom: str,
        tf_list: str ='/home/share/huadjyin/home/sunyi3/Pyscenic/data/hs_hgnc_tfs.txt',
        motif_db: str ='/home/share/huadjyin/home/sunyi3/Pyscenic/data/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather',
        motif_annot="/home/share/huadjyin/home/sunyi3/Pyscenic/data/motifs-v9-fixed.tbl",
        output_dir: str ='/home/share/huadjyin/home/sunyi3/Pyscenic/data/genomix',
        num_workers: int = 20,
) -> None:
    """
    运行完整的 pyscenic 三步流程： GRN 构建 → motif context → AUC 打分。

    参数：
    -expr_loom: 表达矩阵的loom 文件路径
    -tf_list: TF列表的文件路径
    -motif_db: motif 排名文件（.feather)路径
    -motif_annot : motif 注释文件（.tbl） 路径
    -output_dir : 所有输出文件的保存目录。
    -num_workers : 使用的并行线程数。
    """

    expr_loom = Path(expr_loom).resolve()
    tf_list = Path(tf_list).resolve()
    motif_db = Path(motif_db).resolve()
    motif_annot = Path(motif_annot).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    #输出文件路径
    adjacencies_path = output_dir / "adjacencies.tsv"
    regulons_path = output_dir / "regulons.csv"
    auc_matrix_path = output_dir /"auc_matrix.csv"

    try:
        print("Step 1:GRN inference (GENIE3)...")
        subprocess.run([
            "pyscenic", "grn",
            str(expr_loom),
            str(tf_list),
            "--output",str(adjacencies_path),
            "--num_workers", "8"
        ], check=True)

        print("Step 2: Motif enrichment and context pruning...")
        subprocess.run([
            "pyscenic", "ctx",
            str(adjacencies_path),
            str(motif_db),
            "--annotations_fname", str(motif_annot),
            "--expression_mtx_fname", str(expr_loom),
            "--output", str(regulons_path),
            "--mask_dropouts",
            "--num_workers", str(num_workers)
        ], check=True)

        print("Step 3: AUCell scoring...")
        subprocess.run([
            "pyscenic", "aucell",
            str(expr_loom),
            str(regulons_path),
            "--output", str(auc_matrix_path),
            "--num_workers", str(num_workers)
        ], check=True)

        print(f"pySCENIC 全流程完成，结果已保存到: {output_dir}")



    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pySCENIC 步骤执行失败：{e}")