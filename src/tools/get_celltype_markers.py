from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import anndata as ad
import scanpy as sc
import numpy as np

class GetCelltypeMarkersArgs(BaseModel):
    adata_path: str = Field(..., description="输入ADATA文件路径（需包含celltype_col列）")
    celltype_col: str = Field(default="pred_celltype", description="细胞类型注释列名")
    target_celltype: str = Field(..., description="目标细胞类型（需与注释列中的类型完全一致）")
    top_n: int = Field(default=50, description="提取Top N个Marker基因（默认50）")
    method: str = Field(default="wilcoxon", description="差异分析方法（默认wilcoxon，单细胞推荐）")

import numpy as np
import scipy.sparse as sp

def is_log_transformed(adata: ad.AnnData) -> bool:
    """
    判断adata.X是否已对数转换（兼容稀疏矩阵）。
    对数转换的特征：
    1. 无负值（log1p(counts) ≥ 0）；
    2. 最大值 ≤ 20（log1p(1e8) ≈ 18.4，单细胞计数不会超过1e8）；
    3. 小于等于0的元素比例 ≤ 10%（对数转换后，只有原计数为0的元素会保持0）。
    """
    # 步骤1：将稀疏矩阵转换为稠密矩阵（仅临时转换，不修改原adata）
    if sp.issparse(adata.X):
        X = adata.X.toarray()  # 转换为numpy数组（dense）
    else:
        X = adata.X.copy()     # 若已为dense，直接复制
    
    # 步骤2：检查数据特征
    try:
        # 特征1：无负值
        if np.min(X) < 0:
            return False
        # 特征2：最大值 ≤ 20
        if np.max(X) > 20:
            return False
        # 特征3：≤0的元素比例 ≤ 10%
        zero_ratio = np.mean(X <= 0)
        if zero_ratio > 0.1:
            return False
        # 所有特征满足，返回True（已对数转换）
        return True
    except Exception as e:
        print(f"检测数据状态失败：{str(e)}")
        return False  # 检测失败时，默认视为未对数转换

def get_celltype_markers_tool(inputs: GetCelltypeMarkersArgs) -> dict:
    try:
        adata = ad.read_h5ad(inputs.adata_path)
        
        # 验证细胞类型列
        if inputs.celltype_col not in adata.obs.columns:
            return {"status": "error", "message": f"ADATA中无{inputs.celltype_col}列"}
        if inputs.target_celltype not in adata.obs[inputs.celltype_col].cat.categories:
            return {"status": "error", "message": f"目标细胞类型{inputs.target_celltype}不在注释列中"}
        
        
        is_logged = is_log_transformed(adata)
        if is_logged:
            print("⚠️ 警告：adata.X已对数转换，跳过预处理步骤")
        else:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print("✅ 数据已完成归一化+对数转换")
       
        
        # 计算差异基因
        sc.tl.rank_genes_groups(
            adata,
            groupby=inputs.celltype_col,
            method=inputs.method,
            key_added="temp_rank_genes"
        )
        
        # 提取Marker基因
        rank_genes = adata.uns["temp_rank_genes"]
        if "names" not in rank_genes or inputs.target_celltype not in rank_genes["names"].dtype.names:
            return {"status": "error", "message": f"无{inputs.target_celltype}的差异基因结果"}
        marker_genes = rank_genes["names"][inputs.target_celltype][:inputs.top_n].tolist()
        
        del adata.uns["temp_rank_genes"]
        return {
            "status": "success",
            "target_celltype": inputs.target_celltype,
            "marker_genes": marker_genes,
            "extract_rule": f"Top {inputs.top_n}个差异基因（方法：{inputs.method}，数据状态：{'已对数转换' if is_logged else '原始计数→归一化+对数转换'}）"
        }
    except Exception as e:
        return {"status": "error", "message": f"工具执行失败：{str(e)}"}
    
    # 包装GetCelltypeMarkers工具
get_celltype_markers_langchain = StructuredTool.from_function(
    func=get_celltype_markers_tool,
    name="GetCelltypeMarkers",
    description="""
用途：从单细胞ADATA文件中获取指定细胞类型的Top N个Marker基因（自动完成数据预处理+差异基因分析）。
适用场景：当你需要获取某细胞类型的核心功能基因时使用。
输入参数：
- adata_path: 输入ADATA文件路径（需包含celltype_col列，且为原始计数或对数转换后的数据）；
- celltype_col: 细胞类型注释列名（默认'pred_celltype'）；
- target_celltype: 目标细胞类型（需与注释列中的类型完全一致，如'CD8-positive, alpha-beta cytotoxic T cell'）；
- top_n: 提取Top N个Marker基因（默认50）；
输出：成功时返回Marker基因列表、数据状态；失败时返回错误原因。
""",
    args_schema=GetCelltypeMarkersArgs
)