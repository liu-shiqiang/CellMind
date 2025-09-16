from langchain_core.tools import tool
import gget
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from config.setting import settings
import json
import numpy as np


def plot_gsea_barplot(df: pd.DataFrame, output_path: str):
    """ barplot for GSEA result）"""
    # 1. 数据预处理（按显著性排序+计算-log10(adjP)）
    df = df.sort_values(by="adjP", ascending=True)
    df["logP"] = -np.log10(df["adjP"])
    
    # 2. 通路名称自动换行（解决过长问题）
    df["Path"] = df["Path"].apply(lambda x: textwrap.fill(x, width=30))

    # 3. 专业渐变蓝配色（增强视觉区分）
    cmap = plt.get_cmap("Blues_r")
    colors = cmap(np.linspace(0.2, 0.8, len(df)))

    # 4. 创建水平条形图（紧凑布局）
    plt.figure(figsize=(9, 6))
    bars = plt.barh(
        y=df["Path"],
        width=df["logP"],
        color=colors,
        height=0.7,
        edgecolor="white"
    )

    # 5. 添加基因计数标签（条形内部右侧）
    for bar, count in zip(bars, df["Count"]):
        plt.text(
            x=bar.get_width() - 0.2,
            y=bar.get_y() + bar.get_height()/2,
            s=str(count),
            ha="right", va="center",
            color="white", fontweight="bold", fontsize=10
        )

    # 6. 优化x轴（密集刻度+合理范围）
    max_logP = df["logP"].max()
    plt.xlim(0, max_logP + 0.5)
    plt.xticks(np.arange(0, max_logP + 0.5, 1), fontsize=10)
    plt.xlabel("-log10(Adjusted P-value)", fontsize=12, labelpad=10)
    plt.ylabel("Pathway", fontsize=12, labelpad=10)

    # 7. 美化细节
    plt.title("Top 10 Enriched Pathways", fontsize=14, fontweight="bold", pad=15)
    plt.yticks(fontsize=9)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # 8. 保存高分辨率图
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@tool
def gene_set_enrichment_analysis(
    genes: list,
    top_k: int = 10,
    database: str = "ontology",
    background_list: list = None,
    plot: bool = True,
    output_dir: str = str(settings.OUTPUT_DIR),
) -> str:
    """
    Perform Gene Set Enrichment Analysis (GSEA) for a list of genes, with optional visualization.
    
    参数说明：
    - genes: 待分析的基因列表（如细胞类型Marker基因）；
    - top_k: 返回Top N条显著通路（默认10）；
    - database: 富集数据库（默认GO_Biological_Process_2021,  
    Popular options include:
        - 'pathway'      (KEGG_2021_Human)
        - 'transcription'   (ChEA_2016)
        - 'ontology'     (GO_Biological_Process_2021)
        - 'diseases_drugs'  (GWAS_Catalog_2019)
        - 'celltypes'     (PanglaoDB_Augmented_2021)
        - 'kinase_interactions' (KEA_2015)）；
    - background_list: 背景基因集（可选，默认全基因组）；
    - plot: 是否生成条形图（默认True）；
    - output_dir: 图表保存目录（默认=settings.OUTPUT_DIR）。
    
    返回：
    - JSON字符串：包含分析步骤、图表路径、用户友好的文字结果、结构化详情。
    """
    steps_log = []
    try:
        # 1. 执行GSEA分析（本质为ORA，但统一命名）
        steps_log.append(f"Starting GSEA for genes: {', '.join(genes)}")
        df = gget.enrichr(genes, database=database, background_list=background_list)
        df = df.head(top_k)  # 取Top 10显著通路

        # 2. 数据适配（可视化用）
        gsea_df = df[["path_name", "adj_p_val", "overlapping_genes"]].rename(columns={
            "path_name": "Path",
            "adj_p_val": "adjP",
            "overlapping_genes": "Count"
        })
        gsea_df["Count"] = gsea_df["Count"].apply(len)

        # 3. 生成可视化（若plot=True）
        plot_path = None
        if plot:
            output_dir = Path(output_dir).expanduser().resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / "gsea_top10_barplot.png"
            plot_gsea_barplot(gsea_df, str(plot_path))
            steps_log.append(f"Bar plot saved to: {plot_path}")

        # 4. 生成用户友好的文字结果
        user_friendly_result = []
        for idx, row in df.iterrows():
            user_friendly_result.append(f"""
Rank {idx+1}: {row['path_name']}
- Adjusted P-value: {row['adj_p_val']:.2e}
- Overlapping Genes: {', '.join(row['overlapping_genes'])}
- Combined Score: {row['combined_score']:.6f}
----------------------------------------
            """.strip())

        # 5. 结构化返回结果
        return json.dumps({
            "status": "success",
            "steps": "\n".join(steps_log),
            "plot_path": str(plot_path) if plot_path else None,
            "user_friendly_result": "\n".join(user_friendly_result),
            "enrichment_details": df.to_dict("records")
        })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"GSEA failed: {str(e)}"
        })