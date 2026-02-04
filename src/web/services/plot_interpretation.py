"""图表解读服务

提供单细胞分析图表的标题、描述和解读信息。
用于前端展示图表时的说明文字。
"""
from typing import Dict, Any, List, Optional


class PlotInterpretation:
    """图表解读信息"""

    def __init__(
        self,
        title: str,
        description: str,
        what_to_look: List[str],
        biological_meaning: str = "",
        technical_notes: str = "",
    ):
        self.title = title
        self.description = description
        self.what_to_look = what_to_look
        self.biological_meaning = biological_meaning
        self.technical_notes = technical_notes

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "description": self.description,
            "what_to_look": self.what_to_look,
            "biological_meaning": self.biological_meaning,
            "technical_notes": self.technical_notes,
        }


# 图表解读字典
PLOT_INTERPRETATIONS: Dict[str, PlotInterpretation] = {
    "qc_violin": PlotInterpretation(
        title="质控指标分布 (QC Violin Plot)",
        description="展示每个细胞的基因数、UMI数和线粒体基因比例分布，用于识别低质量细胞。",
        what_to_look=[
            "基因数和UMI数分布：大多数细胞应集中在相似范围",
            "线粒体基因比例：高比例（>20%）可能表示细胞损伤或应激状态",
            "离群细胞：考虑过滤极端值",
            "双细胞特征：异常高的基因数和UMI数可能表示双细胞",
        ],
        biological_meaning="高质量细胞应该有适中的基因数和UMI数，线粒体基因比例较低。离群细胞可能是低质量细胞或双细胞。",
        technical_notes="建议过滤线粒体基因比例>20%的细胞。基因数过低可能是空液滴，过高可能是双细胞。",
    ),

    "umap_cluster": PlotInterpretation(
        title="UMAP 聚类可视化 (UMAP Cluster Plot)",
        description="展示细胞在二维UMAP空间中的分布，相似细胞的聚集表示转录组相似性。",
        what_to_look=[
            "聚类分离度：不同cluster应有明显分离",
            "细胞类型分布：观察是否有明显的细胞类型分层",
            "批次效应：如果样本按batch聚集可能存在批次效应",
            "稀有细胞：小群体可能代表稀有细胞类型",
        ],
        biological_meaning="UMAP将高维转录组数据降维到2D，相似细胞聚集在一起。每个cluster可能代表不同的细胞类型或状态。",
        technical_notes="UMAP是非线性降维方法，主要保留局部结构。颜色代表不同cluster。",
    ),

    "umap_annotated": PlotInterpretation(
        title="UMAP 细胞类型注释 (UMAP Annotated Plot)",
        description="在UMAP图上标注细胞类型，直观展示不同细胞类型的空间分布。",
        what_to_look=[
            "注释连续性：相同细胞类型的细胞应聚集",
            "注释准确性：检查是否有明显的错误注释",
            "稀有细胞类型：小群体可能是稀有细胞类型",
            "细胞类型边界：清晰的边界表示注释质量高",
        ],
        biological_meaning="细胞类型注释基于标记基因表达，将生物学意义赋予每个cluster。",
        technical_notes="注释质量取决于标记基因分析的质量。低置信度注释应谨慎解读。",
    ),

    "marker_heatmap": PlotInterpretation(
        title="标记基因热图 (Marker Gene Heatmap)",
        description="展示各cluster的top标记基因表达模式，红色表示高表达，蓝色表示低表达。",
        what_to_look=[
            "cluster特异性标记：每个cluster应有独特的标记基因",
            "表达模式：相似表达模式的cluster可能是同一细胞类型",
            "标记基因强度：高logFC表示强marker",
            "已知标记：检查已知细胞类型标记的位置",
        ],
        biological_meaning="标记基因定义每个cluster的细胞身份。高表达基因可能是该细胞类型的功能基因。",
        technical_notes="热图显示z-score标准化后的表达值。红色=高表达，蓝色=低表达。",
    ),

    "pca_variance": PlotInterpretation(
        title="PCA 方差解释 (PCA Variance Plot)",
        description="展示各主成分解释的方差比例，帮助确定使用多少个PC进行下游分析。",
        what_to_look=[
            "肘部位置：方差解释率明显下降的点",
            "累积方差：前30-50个PC通常解释大部分方差",
            "选择PC数：建议选择累积方差>80%的PC数",
            "第1-2个PC：通常捕获最大的生物学变异",
        ],
        biological_meaning="PCA降维捕获数据中的主要变异来源。前几个PC通常与细胞类型差异相关。",
        technical_notes="选择过多PC可能引入噪声，过少可能丢失重要信号。建议使用Elbow方法确定PC数。",
    ),

    "volcano": PlotInterpretation(
        title="差异分析火山图 (Volcano Plot)",
        description="展示基因表达的log2 fold change与统计显著性关系，用于识别关键差异基因。",
        what_to_look=[
            "显著上调基因：右上角（高logFC，低p值）",
            "显著下调基因：左上角（低logFC，低p值）",
            "关键标记基因：已知细胞类型标记的位置",
            "基因功能：关注top差异基因的生物学功能",
        ],
        biological_meaning="火山图展示两组细胞间的基因表达差异。top基因定义细胞类型的特异性功能。",
        technical_notes="通常使用logFC>1和p adj<0.05作为阈值筛选显著差异基因。",
    ),

    "dot_plot": PlotInterpretation(
        title="基因表达点图 (Dot Plot)",
        description="展示基因在不同cluster中的表达比例和平均表达水平。",
        what_to_look=[
            "点的大小：表示表达该基因的细胞比例",
            "点的颜色：表示平均表达水平",
            "cluster特异性：理想标记基因在特定cluster高表达",
            "表达模式：确定基因的表达特异性",
        ],
        biological_meaning="点图同时展示基因表达的广度和强度，是验证标记基因的理想工具。",
        technical_notes="点大小=表达比例，点颜色=平均表达（对数尺度）。",
    ),

    "trajectory": PlotInterpretation(
        title="细胞轨迹图 (Trajectory Plot)",
        description="展示细胞的发育轨迹或分化过程。",
        what_to_look=[
            "起始细胞：轨迹的起点",
            "分化分支：不同的分化方向",
            "中间状态：轨迹中间的过渡细胞",
            "终点细胞：成熟的细胞类型",
        ],
        biological_meaning="轨迹分析重建细胞的分化过程，揭示发育动态。",
        technical_notes="轨迹分析基于伪时间排序，需要生物学知识验证。",
    ),
}


def get_plot_interpretation(plot_type: str) -> Dict[str, Any]:
    """获取图表解读信息

    Args:
        plot_type: 图表类型

    Returns:
        图表解读字典
    """
    interpretation = PLOT_INTERPRETATIONS.get(plot_type)
    if interpretation:
        return interpretation.to_dict()

    # 返回默认解读
    return {
        "title": plot_type.replace("_", " ").title(),
        "description": "单细胞分析结果可视化",
        "what_to_look": ["图表展示分析结果的关键特征"],
        "biological_meaning": "",
        "technical_notes": "",
    }


def get_plot_title(plot_type: str) -> str:
    """获取图表标题

    Args:
        plot_type: 图表类型

    Returns:
        图表标题
    """
    interpretation = PLOT_INTERPRETATIONS.get(plot_type)
    if interpretation:
        return interpretation.title
    return plot_type.replace("_", " ").title()


def list_all_plot_types() -> List[str]:
    """列出所有支持的图表类型

    Returns:
        图表类型列表
    """
    return list(PLOT_INTERPRETATIONS.keys())


def get_interpretations_summary() -> Dict[str, str]:
    """获取所有图表类型的标题摘要

    Returns:
        图表类型到标题的映射
    """
    return {
        plot_type: interpretation.title
        for plot_type, interpretation in PLOT_INTERPRETATIONS.items()
    }


# 导出
__all__ = [
    "PlotInterpretation",
    "PLOT_INTERPRETATIONS",
    "get_plot_interpretation",
    "get_plot_title",
    "list_all_plot_types",
    "get_interpretations_summary",
]
