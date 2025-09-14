import scanpy as sc
import pandas as pd
import loompy
from pathlib import Path

def h5ad_to_loom(
    h5ad_path: str,
    loom_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    n_top_genes: int = 2000,
) -> None:
    """
    将 .h5ad 文件转换为 .loom 格式，适用于 pySCENIC 分析。

    参数：
    - h5ad_path (str): 输入 .h5ad 文件的路径。
    - loom_path (str): 输出 .loom 文件的保存路径。
    - min_genes (int): 过滤细胞的最小基因表达数，默认为 200。
    - min_cells (int): 过滤基因的最小细胞表达数，默认为 3。
    - n_top_genes (int): 选择的高变基因数量，默认为 2000。

    异常：
    - FileNotFoundError: 如果输入文件不存在或不是 .h5ad 格式。
    - ValueError: 如果输出文件路径后缀不是 .loom。
    - RuntimeError: 如果数据处理或文件写入失败。
    """
    # 规范化路径
    h5ad_path = Path(h5ad_path).expanduser().resolve()
    loom_path = Path(loom_path).expanduser().resolve()

    # 验证输入和输出文件
    if not h5ad_path.exists() or h5ad_path.suffix.lower() != ".h5ad":
        raise FileNotFoundError(f"无效或缺失的 .h5ad 文件: {h5ad_path}")
    if loom_path.suffix.lower() != ".loom":
        raise ValueError(f"输出文件路径必须以 .loom 结尾: {loom_path}")

    # 确保输出目录存在
    loom_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 读取 h5ad 文件
        adata = sc.read_h5ad(h5ad_path)

        # 保存原始数据
        adata.layers["counts"] = adata.X.copy()

        # 过滤细胞和基因
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata,  n_top_genes=n_top_genes )
        adata = adata[:, adata.var['highly_variable']]  # 保留高变基因
    

        # 过滤高变基因
        if "highly_variable" not in adata.var:
            raise RuntimeError("高变基因计算失败，未找到 'highly_variable' 列")
        print(f"保留了 {adata.shape[1]} 个高变基因")

        # 设置 pySCENIC 所需的行列属性
        gene_names = adata.var_names.values       # 原来列是基因
        cell_names = adata.obs_names.values       # 原来行是细胞

        # 保存为 loom 文件
        loompy.create(
            str(loom_path),
            adata.X.T,    # 转置后 shape = (genes, cells)
            row_attrs={"Gene": gene_names},         # 每行是基因
            col_attrs={"CellID": cell_names}        # 每列是细胞
        )

        print(f"成功将 {h5ad_path} 转换为 {loom_path}")

    except Exception as exc:
        raise RuntimeError(f"处理或转换失败: {str(exc)}")