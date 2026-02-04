"""智能细胞类型注释工具

根据组织类型自动选择最合适的注释方法：
- 血液/免疫组织 → CIMA markers（快速、准确）
- 脑/神经组织 → LLM + RAG（支持复杂细胞类型）
- 其他组织 → LLM注释

如果CIMA注释覆盖率 < 60%，自动使用LLM补充未注释的clusters。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import pandas as pd
import scanpy as sc
from langchain_core.tools import tool

from src.tools.base import ToolResult, create_tool_result, detect_cluster_key
from src.tools.utils.path_resolver import PathResolver
from src.tools.utils.validation import ToolValidator

logger = logging.getLogger(__name__)

# 初始化工具
path_resolver = PathResolver()
validator = ToolValidator()

# 组织类型分类
BLOOD_TISSUES = {
    'blood', 'bone_marrow', 'pbmc', 'immune', 'spleen', 'thymus',
    'peripheral_blood', 'cord_blood', 'lymph_node', 'tonsil'
}

BRAIN_TISSUES = {
    'brain', 'cerebrum', 'cerebellum', 'cortex', 'hippocampus', 'neural',
    'cerebral_cortex', 'brain_cortex', 'prefrontal_cortex', 'motor_cortex',
    'visual_cortex', 'basal_ganglia', 'substantia_nigra', 'striatum'
}

# 组织类型特异标记基因（用于检测）
TISSUE_MARKERS = {
    'blood': {
        'markers': ['PTPRC', 'CD3D', 'CD3E', 'CD19', 'MS4A1', 'CD14', 'LYZ'],
        'threshold': 3  # 至少3个标记基因高表达
    },
    'brain': {
        'markers': ['SLC17A7', 'GAD1', 'GAD2', 'GFAP', 'MBP', 'OLIG2', 'AQP4'],
        'threshold': 3
    },
}


def _resolve_input_path(file_path: str) -> Path:
    """解析输入文件路径"""
    return path_resolver.resolve_input_path(file_path, None)


def _resolve_artifact_dirs(input_path: Path):
    """解析输出目录"""
    return path_resolver.resolve_all_output_dirs(input_path)


def _detect_tissue_type(adata: sc.AnnData, file_path: str = "") -> Optional[str]:
    """从adata.metadata或基因表达模式检测组织类型

    Args:
        adata: AnnData对象
        file_path: 文件路径（用于额外线索）

    Returns:
        检测到的组织类型: 'blood', 'brain', 或 None（未知组织）
    """
    # 1. 检查metadata
    tissue = None

    # 检查 obs 列
    if 'tissue' in adata.obs.columns:
        tissue_values = adata.obs['tissue'].dropna()
        if len(tissue_values) > 0:
            tissue = str(tissue_values.iloc[0]).lower()

    # 检查 uns
    if not tissue:
        tissue = adata.uns.get('tissue_type') or adata.uns.get('tissue')
        if tissue:
            tissue = str(tissue).lower()

    # 检查文件名
    if not tissue and file_path:
        file_path_lower = file_path.lower()
        for bt in BLOOD_TISSUES:
            if bt in file_path_lower:
                tissue = 'blood'
                break
        if not tissue:
            for bt in BRAIN_TISSUES:
                if bt in file_path_lower:
                    tissue = 'brain'
                    break

    # 如果找到了明确的组织类型，直接返回
    if tissue:
        if any(bt in tissue for bt in BLOOD_TISSUES):
            return 'blood'
        if any(bt in tissue for bt in BRAIN_TISSUES):
            return 'brain'

    # 2. 基于基因表达模式检测
    try:
        # 检查是否有标准化数据
        if 'X' not in adata.layers:
            # 使用原始数据
            data = adata.X
        else:
            data = adata.layers['X']

        # 检测血液组织标记
        blood_markers = TISSUE_MARKERS['blood']['markers']
        blood_present = _check_marker_expression(adata, blood_markers, data)

        # 检测脑组织标记
        brain_markers = TISSUE_MARKERS['brain']['markers']
        brain_present = _check_marker_expression(adata, brain_markers, data)

        if blood_present >= TISSUE_MARKERS['blood']['threshold']:
            return 'blood'
        if brain_present >= TISSUE_MARKERS['brain']['threshold']:
            return 'brain'

    except Exception as e:
        logger.warning(f"基因表达检测失败: {e}")

    # 无法确定组织类型
    return None


def _check_marker_expression(adata: sc.AnnData, markers: List[str], data) -> int:
    """检查标记基因的表达情况

    Args:
        adata: AnnData对象
        markers: 标记基因列表
        data: 表达矩阵

    Returns:
        检测到的标记基因数量
    """
    detected = 0
    var_names_set = set(adata.var_names)

    for marker in markers:
        # 检查基因是否存在于数据中
        if marker in var_names_set:
            idx = adata.var_names.get_loc(marker)
            try:
                from scipy.sparse import issparse
                if issparse(data):
                    expr = data[:, idx].sum()
                else:
                    expr = data[:, idx].sum()

                # 如果该基因有表达，计数
                if expr > 0:
                    detected += 1
            except Exception:
                pass
        else:
            # 尝试大小写不敏感匹配
            for var_name in var_names_set:
                if var_name.upper() == marker.upper():
                    idx = adata.var_names.get_loc(var_name)
                    try:
                        from scipy.sparse import issparse
                        if issparse(data):
                            expr = data[:, idx].sum()
                        else:
                            expr = data[:, idx].sum()
                        if expr > 0:
                            detected += 1
                        break
                    except Exception:
                        pass

    return detected


def _calculate_coverage(annotation_map: Dict[str, str], n_clusters: int) -> float:
    """计算注释覆盖率

    Args:
        annotation_map: 注释映射 {cluster: cell_type}
        n_clusters: 总cluster数

    Returns:
        覆盖率 (0-1)
    """
    if n_clusters == 0:
        return 0.0

    annotated = sum(
        1 for ct in annotation_map.values()
        if not ct.startswith('Cluster_') and ct != 'Unknown'
    )
    return annotated / n_clusters


def _get_unannotated_clusters(annotation_map: Dict[str, str]) -> List[str]:
    """获取未注释的cluster列表

    Args:
        annotation_map: 注释映射 {cluster: cell_type}

    Returns:
        未注释的cluster ID列表
    """
    return [
        cluster for cluster, cell_type in annotation_map.items()
        if cell_type.startswith('Cluster_') or cell_type == 'Unknown'
    ]


def _llm_annotate_remaining(
    file_path: str,
    cluster_key: str,
    existing_annotations: Dict[str, str],
    tissue_context: str = "",
    species: str = "human",
    llm_model: str = "qwen3:8b",
    llm_provider: str = "ollama",
) -> Dict[str, Any]:
    """使用LLM对未注释的cluster进行补充注释

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名
        existing_annotations: 现有注释映射
        tissue_context: 组织上下文
        species: 物种
        llm_model: LLM模型名称
        llm_provider: LLM提供商

    Returns:
        补充注释结果
    """
    try:
        # 动态导入并调用LLM注释工具
        from src.tools.annotation.llm_annotate import annotate_with_llm as llm_tool

        # 获取底层函数
        llm_fn = llm_tool.func if hasattr(llm_tool, 'func') else llm_tool

        # 调用LLM注释工具（内部会处理所有clusters）
        result_json = llm_fn(
            file_path=file_path,
            cluster_key=cluster_key,
            tissue_context=tissue_context,
            species=species,
            llm_model=llm_model,
            llm_provider=llm_provider,
            save_result=False,  # 不保存，只是获取结果
        )

        result_data = json.loads(result_json)

        # 提取LLM的注释结果
        llm_annotations = result_data.get('data', {}).get('annotations', {})

        # 只保留未注释的cluster的LLM注释
        unannotated = _get_unannotated_clusters(existing_annotations)
        supplementary = {
            cluster: llm_annotations.get(cluster, f"Cluster_{cluster}")
            for cluster in unannotated
        }

        return {
            'supplementary_annotations': supplementary,
            'llm_full_result': result_data,
        }

    except Exception as e:
        logger.warning(f"LLM补充注释失败: {e}")
        return {'supplementary_annotations': {}, 'error': str(e)}


@tool("annotate_cells", return_direct=False)
def smart_annotate_cells(
    file_path: str,
    cluster_key: Optional[str] = None,
    tissue_type: Optional[str] = None,
    auto_detect: bool = True,
    fallback_threshold: float = 0.6,
    species: str = "human",
    save_result: bool = True,
    llm_model: str = "qwen3:8b",  # 默认使用本地ollama
    llm_provider: str = "ollama",  # 默认使用本地ollama
) -> str:
    """智能细胞类型注释

    根据组织类型自动选择最佳注释方法：
    - 血液/免疫组织 → CIMA markers（快速、准确）
    - 脑/神经组织 → LLM + RAG（支持复杂细胞类型）
    - 其他组织 → LLM注释

    如果CIMA注释覆盖率 < 60%，自动使用LLM补充未注释的clusters。

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        tissue_type: 组织类型（blood, brain, tumor等），不指定则自动检测
        auto_detect: 是否自动检测组织类型
        fallback_threshold: CIMA注释覆盖率阈值，低于此值使用LLM补充
        species: 物种类型 ("human" 或 "mouse")
        save_result: 是否保存结果

    Returns:
        注释结果摘要，包含每个cluster的细胞类型、置信度和推理依据

    Example:
        >>> smart_annotate_cells("data.h5ad")
        '{"annotations": {"0": "CD8_Tem", ...}, "method": "cima", ...}'
    """
    try:
        logger.info(f"开始智能细胞注释: {file_path}")

        # 验证参数
        species = validator.validate_choices(species, ['human', 'mouse'], 'species')
        fallback_threshold = float(validator.validate_positive_number(
            fallback_threshold, 'fallback_threshold', 0, 1
        ))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, plots_dir = _resolve_artifact_dirs(path)

        # 获取run_id用于API路径
        from src.web.config import settings
        runs_root = Path(settings.RUNS_DIR).resolve()
        run_id = None
        try:
            if path.is_relative_to(runs_root):
                run_id = path.relative_to(runs_root).parts[0]
        except ValueError:
            pass

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 1. 检测或使用指定的组织类型
        detected_tissue = None
        if auto_detect and not tissue_type:
            detected_tissue = _detect_tissue_type(adata, str(path))
        elif tissue_type:
            detected_tissue = tissue_type.lower()

        logger.info(f"检测到的组织类型: {detected_tissue or '未知'}")

        # 2. 根据组织类型选择注释方法
        annotation_map = {}
        annotation_details = []
        method_used = "unknown"
        cima_coverage = 0.0
        llm_supplemented = False

        if detected_tissue == 'blood':
            # 使用CIMA markers注释血液组织
            logger.info("使用 CIMA markers 注释血液组织")
            try:
                from src.tools.annotation.marker_based import annotate_with_cima_markers as cima_tool

                cima_fn = cima_tool.func if hasattr(cima_tool, 'func') else cima_tool
                cima_result_json = cima_fn(
                    file_path=str(path),
                    cluster_key=cluster_key,
                    save_result=False,  # 我们自己保存
                )
                cima_result = json.loads(cima_result_json)

                if cima_result.get('status') == 'success':
                    annotation_map = cima_result['data']['annotations']
                    annotation_details = cima_result['data'].get('annotation_details', [])
                    method_used = "cima"

                    # 计算覆盖率
                    clusters = adata.obs[cluster_key].cat.categories.tolist()
                    cima_coverage = _calculate_coverage(annotation_map, len(clusters))
                    logger.info(f"CIMA 覆盖率: {cima_coverage:.2%}")

                    # 如果覆盖率不足，使用LLM补充
                    if cima_coverage < fallback_threshold:
                        logger.info(f"CIMA 覆盖率 ({cima_coverage:.2%}) 低于阈值 ({fallback_threshold:.2%})，使用 LLM 补充")

                        supplementary = _llm_annotate_remaining(
                            file_path=str(path),
                            cluster_key=cluster_key,
                            existing_annotations=annotation_map,
                            tissue_context="blood or immune tissue",
                            species=species,
                            llm_model=llm_model,
                            llm_provider=llm_provider,
                        )

                        if supplementary.get('supplementary_annotations'):
                            annotation_map.update(supplementary['supplementary_annotations'])
                            llm_supplemented = True
                            method_used = "cima_llm_fallback"

            except Exception as e:
                logger.warning(f"CIMA 注释失败: {e}，回退到 LLM 注释")

        # 如果是脑组织或CIMA失败，使用LLM注释
        if not annotation_map or detected_tissue == 'brain' or detected_tissue is None:
            logger.info(f"使用 LLM 注释 (组织类型: {detected_tissue or '未知'})")

            from src.tools.annotation.llm_annotate import annotate_with_llm as llm_tool
            llm_fn = llm_tool.func if hasattr(llm_tool, 'func') else llm_tool

            tissue_context = ""
            if detected_tissue == 'brain':
                tissue_context = "brain or neural tissue"
            elif detected_tissue == 'blood':
                tissue_context = "blood or immune tissue"

            llm_result_json = llm_fn(
                file_path=str(path),
                cluster_key=cluster_key,
                tissue_context=tissue_context,
                species=species,
                llm_model=llm_model,
                llm_provider=llm_provider,
                save_result=False,
            )
            llm_result = json.loads(llm_result_json)

            if llm_result.get('status') == 'success':
                annotation_map = llm_result['data']['annotations']
                annotation_details = llm_result['data'].get('annotation_details', [])
                method_used = "llm"

        # 应用注释到 adata
        cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
        if 'cell_type' in adata.obs.columns:
            del adata.obs['cell_type']
        adata.obs['cell_type'] = cluster_values.astype(str)

        if 'pred_celltype' in adata.obs.columns:
            del adata.obs['pred_celltype']
        adata.obs['pred_celltype'] = adata.obs['cell_type'].astype(str)

        # 统计
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()

        # 计算最终覆盖率
        clusters = adata.obs[cluster_key].cat.categories.tolist()
        final_coverage = _calculate_coverage(annotation_map, len(clusters))

        result = ToolResult(
            status="success",
            message=f"智能注释完成，{len(annotation_map)}个clusters，覆盖率{final_coverage:.1%}",
            data={
                "cluster_key": cluster_key,
                "species": species,
                "n_annotated": len(annotation_map),
                "n_clusters": len(clusters),
                "annotations": annotation_map,
                "cell_type_counts": cell_type_counts,
                "annotation_details": annotation_details,
                "method": method_used,
                "detected_tissue": detected_tissue,
                "coverage": final_coverage,
                "llm_supplemented": llm_supplemented,
                "cima_coverage": cima_coverage if method_used.startswith("cima") else None,
            }
        )

        # 保存结果
        if save_result:
            from src.tools.base import _save_result
            result_path = _save_result(adata, "annotated_smart", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 保存注释详情
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_path = tables_dir / f"annotation_details_smart_{timestamp}.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(annotation_details, f, ensure_ascii=False, indent=2)
            result.artifacts["detail_path"] = str(detail_path)

            # 生成注释后的UMAP图
            if 'X_umap' in adata.obsm:
                try:
                    from src.tools.single_cell_core import _save_umap_plot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    plot_metadata = _save_umap_plot(adata, plots_dir, timestamp, 'cell_type', run_id)
                    if plot_metadata:
                        result.artifacts["annotated_umap_plot"] = plot_metadata
                        result.data["annotated_umap_plot"] = plot_metadata
                        if run_id:
                            plot_metadata["run_id"] = run_id
                except Exception as e:
                    logger.warning(f"生成注释UMAP图失败: {e}")

        logger.info(f"智能注释完成: {len(annotation_map)} clusters, 覆盖率 {final_coverage:.1%}")

        return result.to_json()

    except Exception as e:
        logger.error(f"智能注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"智能注释失败: {str(e)}",
            error=str(e)
        )


# 导出工具
__all__ = [
    "smart_annotate_cells",
]
