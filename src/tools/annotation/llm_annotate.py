"""基于LLM+RAG的细胞类型智能注释工具

使用大语言模型结合RAG检索的文献知识进行细胞类型注释。
适用于脑组织等复杂组织的细胞注释。
"""
from __future__ import annotations

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import scanpy as sc
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.tools.base import ToolResult, create_tool_result, detect_cluster_key
from src.tools.utils.path_resolver import PathResolver
from src.tools.utils.validation import ToolValidator
from src.utils.llm_manager import get_llm_manager

logger = logging.getLogger(__name__)

# 初始化工具
path_resolver = PathResolver()
validator = ToolValidator()


class CellAnnotationResult(BaseModel):
    """细胞类型注释结果"""
    cell_type: str = Field(description="细胞类型名称")
    confidence: float = Field(description="置信度评分 (0-1)", ge=0, le=1)
    reasoning: str = Field(description="推理过程")
    key_markers: List[str] = Field(description="关键标记基因", default_factory=list)
    alternative_types: List[str] = Field(description="备选细胞类型", default_factory=list)
    tissue_context: Optional[str] = Field(default=None, description="组织上下文")


def _resolve_input_path(file_path: str) -> Path:
    """解析输入文件路径"""
    return path_resolver.resolve_input_path(file_path, None)


def _resolve_artifact_dirs(input_path: Path):
    """解析输出目录"""
    return path_resolver.resolve_all_output_dirs(input_path)


def _build_annotation_prompt(
    marker_genes: List[str],
    cluster_id: str,
    rag_context: str = "",
    tissue_context: str = "",
    species: str = "human"
) -> str:
    """构建细胞类型注释的prompt

    Args:
        marker_genes: 标记基因列表
        cluster_id: 聚类ID
        rag_context: RAG检索的相关文献知识
        tissue_context: 组织上下文
        species: 物种

    Returns:
        构建的prompt
    """
    # 只使用前20个标记基因
    top_markers = marker_genes[:20] if len(marker_genes) > 20 else marker_genes
    markers_str = ", ".join(top_markers)

    rag_section = ""
    if rag_context:
        rag_section = f"""
## 相关文献知识
{rag_context}
"""

    tissue_section = ""
    if tissue_context:
        tissue_section = f"""
## 组织类型
{tissue_context}
"""

    prompt = f"""你是一个专业的单细胞RNA测序分析专家。请根据以下信息为细胞群体分配最合适的细胞类型。

## 聚类标识
Cluster {cluster_id}

## Top 差异表达基因 (共{len(top_markers)}个)
{markers_str}
{tissue_section}
{rag_section}
## 任务
1. 分析这些标记基因的生物学功能和表达模式
2. 结合文献知识，确定最可能的细胞类型
3. 提供置信度评分（0-1之间的小数，1表示最高置信度）
4. 详细解释推理过程
5. 列出支持该注释的关键标记基因
6. 提供1-3个备选细胞类型

## 输出格式
请以JSON格式返回结果，必须严格遵循以下格式：
```json
{{
    "cell_type": "细胞类型名称",
    "confidence": 0.85,
    "reasoning": "详细的推理过程，包括标记基因分析和文献支持",
    "key_markers": ["基因1", "基因2", "基因3"],
    "alternative_types": ["备选类型1", "备选类型2"],
    "tissue_context": "组织上下文（如果适用）"
}}
```

## 注释建议
- 如果是神经元，请区分兴奋性神经元和抑制性神经元
- 如果是胶质细胞，请区分星形胶质细胞、少突胶质细胞、小胶质细胞等
- 置信度应该基于标记基因的特异性和文献支持程度
- 如果没有足够的证据支持特定细胞类型，请使用更广泛的类别（如"神经元细胞"、"胶质细胞"）
"""
    return prompt


def _extract_marker_genes(
    adata: sc.AnnData,
    cluster_id: str,
    cluster_key: str,
    top_k: int = 50
) -> List[str]:
    """提取聚类的标记基因

    Args:
        adata: AnnData对象
        cluster_id: 聚类ID
        cluster_key: 聚类键名
        top_k: 提取的标记基因数量

    Returns:
        标记基因列表
    """
    try:
        # 新版 Scanpy 不支持 n_genes 参数，先获取全部再切片
        genes_df = sc.get.rank_genes_groups_df(adata, group=str(cluster_id))
        marker_genes = genes_df['names'].head(top_k).tolist()
        return marker_genes
    except Exception as e:
        logger.warning(f"提取cluster {cluster_id} 的标记基因失败: {e}")
        return []


async def _query_rag_for_markers(
    marker_genes: List[str],
    rag_client: Optional[Any] = None,
    top_k: int = 3
) -> str:
    """使用RAG检索标记基因相关的文献知识

    Args:
        marker_genes: 标记基因列表
        rag_client: RAG客户端（BioKnowledgeRag实例）
        top_k: 检索的文档数量

    Returns:
        检索到的文献上下文
    """
    if not rag_client:
        return ""

    try:
        # 使用top标记基因构建查询
        query_genes = marker_genes[:5] if len(marker_genes) > 5 else marker_genes
        query = f"cell type markers {', '.join(query_genes)} single cell RNA sequencing"

        # 同步方式查询RAG
        docs = rag_client.query(query, top_k=top_k)

        if not docs:
            return ""

        # 提取文档内容
        context_parts = []
        for i, doc in enumerate(docs[:top_k], 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_parts.append(f"[文献{i}] {content[:500]}")  # 限制每篇文献的长度

        return "\n\n".join(context_parts)

    except Exception as e:
        logger.warning(f"RAG查询失败: {e}")
        return ""


def _parse_llm_response(response_text: str) -> Optional[CellAnnotationResult]:
    """解析LLM返回的JSON响应

    Args:
        response_text: LLM返回的文本

    Returns:
        解析后的注释结果，失败返回None
    """
    try:
        # 尝试提取JSON部分
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            return CellAnnotationResult(**data)
        else:
            logger.warning(f"未能从响应中提取JSON: {response_text[:200]}")
            return None

    except Exception as e:
        logger.warning(f"解析LLM响应失败: {e}, 响应内容: {response_text[:200]}")
        return None


def _fallback_annotation(
    marker_genes: List[str],
    cluster_id: str
) -> CellAnnotationResult:
    """回退注释策略：基于已知标记基因进行简单注释

    Args:
        marker_genes: 标记基因列表
        cluster_id: 聚类ID

    Returns:
        基于规则的注释结果
    """
    marker_set = set(g.upper() for g in marker_genes[:10])

    # 神经细胞标记 (通用)
    neuron_markers = {'RBFOX3', 'NEUN', 'MAP2', 'SLC17A7', 'VGLUT1', 'SLC17A6', 'SYN1', 'SYT1'}
    # 兴奋性神经元 - 添加更多脑组织特异性标记
    excitatory_markers = {'SLC17A7', 'VGLUT1', 'SLC17A6', 'CAMK2A', 'RORB', 'SATB2', 'CUX1', 'FEZF2'}
    # 抑制性神经元
    inhibitory_markers = {'GAD1', 'GAD2', 'SLC32A1', 'VGAT', 'SLC6A1', 'PVALB', 'SST', 'LHX6'}
    # 星形胶质细胞
    astrocyte_markers = {'GFAP', 'AQP4', 'SLC1A2', 'GLT1', 'ALDH1L1', 'SLC1A3', 'GLUL', 'APOE'}
    # 少突胶质细胞
    oligodendrocyte_markers = {'MBP', 'PLP1', 'MOG', 'OLIG1', 'OLIG2', 'SOX10', 'PDGFRA', 'CSPG4'}
    # 小胶质细胞
    microglia_markers = {'CX3CR1', 'P2RY12', 'AIF1', 'IBA1', 'P2RY12'}
    # 内皮细胞
    endothelial_markers = {'VWF', 'PECAM1', 'CD31', 'CLDN5', 'FLT1', 'ESAM'}
    # 小脑细胞
    purkinje_markers = {'CALB1', 'PCP2', 'CAR8', 'ZIC1'}
    # 室管膜细胞
    ependymal_markers = {'EVI1', 'FOXJ1', 'TMEM212'}

    # 检测细胞类型
    if excitatory_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Excitatory Neuron",
            confidence=0.6,
            reasoning=f"检测到兴奋性神经元标记基因: {', '.join(list(excitatory_markers & marker_set)[:3])}",
            key_markers=list(excitatory_markers & marker_set),
            alternative_types=["Inhibitory Neuron", "Neuron"]
        )
    elif inhibitory_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Inhibitory Neuron",
            confidence=0.6,
            reasoning=f"检测到抑制性神经元标记基因: {', '.join(list(inhibitory_markers & marker_set)[:3])}",
            key_markers=list(inhibitory_markers & marker_set),
            alternative_types=["Excitatory Neuron", "Neuron"]
        )
    elif astrocyte_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Astrocyte",
            confidence=0.6,
            reasoning=f"检测到星形胶质细胞标记基因: {', '.join(list(astrocyte_markers & marker_set)[:3])}",
            key_markers=list(astrocyte_markers & marker_set),
            alternative_types=["Glial Cell"]
        )
    elif oligodendrocyte_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Oligodendrocyte",
            confidence=0.6,
            reasoning=f"检测到少突胶质细胞标记基因: {', '.join(list(oligodendrocyte_markers & marker_set)[:3])}",
            key_markers=list(oligodendrocyte_markers & marker_set),
            alternative_types=["Glial Cell", "OPC"]
        )
    elif microglia_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Microglia",
            confidence=0.6,
            reasoning=f"检测到小胶质细胞标记基因: {', '.join(list(microglia_markers & marker_set)[:3])}",
            key_markers=list(microglia_markers & marker_set),
            alternative_types=["Immune Cell"]
        )
    elif endothelial_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Endothelial Cell",
            confidence=0.6,
            reasoning=f"检测到内皮细胞标记基因: {', '.join(list(endothelial_markers & marker_set)[:3])}",
            key_markers=list(endothelial_markers & marker_set),
            alternative_types=["Vascular Cell"]
        )
    elif purkinje_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Purkinje Cell",
            confidence=0.6,
            reasoning=f"检测到浦肯野细胞标记基因: {', '.join(list(purkinje_markers & marker_set)[:3])}",
            key_markers=list(purkinje_markers & marker_set),
            alternative_types=["Neuron", "Cerebellar Neuron"]
        )
    elif neuron_markers & marker_set:
        return CellAnnotationResult(
            cell_type="Neuron",
            confidence=0.5,
            reasoning=f"检测到神经元标记基因: {', '.join(list(neuron_markers & marker_set)[:3])}",
            key_markers=list(neuron_markers & marker_set),
            alternative_types=["Neural Cell"]
        )
    else:
        return CellAnnotationResult(
            cell_type=f"Cluster_{cluster_id}",
            confidence=0.3,
            reasoning=f"未能识别明确的细胞类型标记，Top标记基因: {', '.join(marker_genes[:5])}",
            key_markers=marker_genes[:5],
            alternative_types=["Unknown"]
        )


@tool("annotate_with_llm", return_direct=False)
def annotate_with_llm(
    file_path: str,
    cluster_key: Optional[str] = None,
    top_k_genes: int = 50,
    use_rag: bool = False,  # 默认关闭RAG，因为scgpt可能未安装
    llm_provider: str = "ollama",  # 默认使用本地ollama
    llm_model: str = "qwen3:8b",  # 默认使用本地模型
    tissue_context: str = "",
    species: str = "human",
    save_result: bool = True,
) -> str:
    """使用大语言模型进行细胞类型注释

    基于聚类的top-k差异基因，结合RAG检索的文献知识，
    使用LLM生成细胞类型标签和解释。适用于脑组织等复杂组织的注释。

    Args:
        file_path: .h5ad 文件路径
        cluster_key: 聚类键名（如 'leiden'），默认自动检测
        top_k_genes: 使用的top差异基因数量
        use_rag: 是否使用RAG增强（需要配置RAG系统）
        llm_provider: LLM提供商（openai, ollama, claude等）
        llm_model: 模型名称
        tissue_context: 组织类型上下文（如"脑组织"、"小脑"等）
        species: 物种类型 ("human" 或 "mouse")
        save_result: 是否保存结果

    Returns:
        注释结果摘要，包含每个cluster的细胞类型、置信度和推理依据

    Example:
        >>> annotate_with_llm("data.h5ad", tissue_context="大脑皮层")
        '{"annotations": {"0": {"cell_type": "Excitatory Neuron", ...}}, ...}'
    """
    try:
        logger.info(f"开始LLM+RAG细胞注释: {file_path}")

        # 验证参数
        species = validator.validate_choices(species, ['human', 'mouse'], 'species')
        top_k_genes = int(validator.validate_positive_number(top_k_genes, 'top_k_genes', 10, 200))

        # 解析路径并加载数据
        path = _resolve_input_path(file_path)
        adata = sc.read_h5ad(path)
        data_dir, tables_dir, _ = _resolve_artifact_dirs(path)

        # 检测聚类键名
        cluster_key = validator.validate_cluster_key(adata, cluster_key)

        # 检查是否有标记基因分析结果
        if "rank_genes_groups" not in adata.uns:
            # 如果没有标记基因，先计算
            sc.tl.rank_genes_groups(
                adata,
                groupby=cluster_key,
                method='wilcoxon',
                n_genes=top_k_genes
            )
            logger.info("已自动运行标记基因分析")

        # 初始化RAG客户端
        rag_client = None
        if use_rag:
            try:
                from src.scripts.rag import BioKnowledgeRag
                from src.web.config import settings

                # RAG向量库配置: data/chroma_data, collection名称: lit_rag
                vector_store_path = Path(settings.DATA_DIR) / "chroma_data"
                if vector_store_path.exists():
                    rag_client = BioKnowledgeRag(
                        vector_store_path=str(vector_store_path),
                        top_k=3
                    )
                    rag_client.init_vector_store("lit_rag")
                    logger.info("RAG系统初始化成功 (chroma_data/lit_rag)")
                else:
                    logger.warning(f"RAG向量存储不存在: {vector_store_path}")
            except Exception as e:
                logger.warning(f"RAG系统初始化失败: {e}，将不使用RAG增强")
                rag_client = None

        # 获取LLM实例
        llm_manager = get_llm_manager()
        try:
            llm = llm_manager.get_llm(llm_model)
        except Exception as e:
            logger.warning(f"使用模型 {llm_model} 失败: {e}，使用默认模型")
            llm = llm_manager.get_llm()

        # 获取聚类列表
        clusters = adata.obs[cluster_key].cat.categories.tolist()
        annotation_map = {}
        annotation_details = []

        logger.info(f"开始为 {len(clusters)} 个聚类进行LLM注释...")

        # 为每个聚类进行注释
        for cluster in clusters:
            try:
                logger.info(f"正在注释 cluster {cluster}...")

                # 提取标记基因
                marker_genes = _extract_marker_genes(adata, str(cluster), cluster_key, top_k_genes)

                if not marker_genes:
                    logger.warning(f"Cluster {cluster} 没有标记基因，跳过")
                    annotation_map[cluster] = f"Cluster_{cluster}"
                    continue

                # RAG增强
                rag_context = ""
                if use_rag and rag_client:
                    try:
                        rag_context = asyncio.run(_query_rag_for_markers(
                            marker_genes, rag_client, top_k=3
                        ))
                        if rag_context:
                            logger.info(f"Cluster {cluster} RAG查询成功，获得相关文献")
                    except Exception as e:
                        logger.warning(f"Cluster {cluster} RAG查询失败: {e}")

                # 构建prompt
                prompt = _build_annotation_prompt(
                    marker_genes=marker_genes,
                    cluster_id=str(cluster),
                    rag_context=rag_context,
                    tissue_context=tissue_context,
                    species=species
                )

                # 调用LLM
                try:
                    response = llm.invoke(prompt)
                    response_text = str(response.content) if hasattr(response, 'content') else str(response)

                    # 解析响应
                    result = _parse_llm_response(response_text)

                    if result:
                        annotation_map[cluster] = result.cell_type
                        annotation_details.append({
                            "cluster": str(cluster),
                            "cell_type": result.cell_type,
                            "confidence": result.confidence,
                            "reasoning": result.reasoning,
                            "key_markers": result.key_markers,
                            "alternative_types": result.alternative_types,
                            "method": "llm_rag" if rag_context else "llm",
                        })
                        logger.info(f"Cluster {cluster} 注释为: {result.cell_type} (置信度: {result.confidence})")
                    else:
                        # 使用回退策略
                        logger.warning(f"Cluster {cluster} LLM响应解析失败，使用回退策略")
                        fallback_result = _fallback_annotation(marker_genes, str(cluster))
                        annotation_map[cluster] = fallback_result.cell_type
                        annotation_details.append({
                            "cluster": str(cluster),
                            "cell_type": fallback_result.cell_type,
                            "confidence": fallback_result.confidence,
                            "reasoning": fallback_result.reasoning,
                            "key_markers": fallback_result.key_markers,
                            "alternative_types": fallback_result.alternative_types,
                            "method": "fallback",
                        })

                except Exception as e:
                    logger.error(f"Cluster {cluster} LLM调用失败: {e}，使用回退策略")
                    marker_genes = _extract_marker_genes(adata, str(cluster), cluster_key, top_k_genes)
                    fallback_result = _fallback_annotation(marker_genes, str(cluster))
                    annotation_map[cluster] = fallback_result.cell_type
                    annotation_details.append({
                        "cluster": str(cluster),
                        "cell_type": fallback_result.cell_type,
                        "confidence": fallback_result.confidence,
                        "reasoning": fallback_result.reasoning,
                        "key_markers": fallback_result.key_markers,
                        "alternative_types": fallback_result.alternative_types,
                        "method": "fallback",
                    })

            except Exception as e:
                logger.error(f"注释cluster {cluster} 失败: {e}")
                annotation_map[cluster] = f"Cluster_{cluster}"

        # 应用注释
        cluster_values = adata.obs[cluster_key].astype(str).map(annotation_map).fillna('Unknown')
        if 'cell_type' in adata.obs.columns:
            del adata.obs['cell_type']
        adata.obs['cell_type'] = cluster_values.astype(str)

        if 'pred_celltype' in adata.obs.columns:
            del adata.obs['pred_celltype']
        adata.obs['pred_celltype'] = adata.obs['cell_type'].astype(str)

        # 计算平均置信度
        valid_confidences = [d["confidence"] for d in annotation_details if isinstance(d.get("confidence"), (int, float))]
        avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0

        # 统计
        cell_type_counts = adata.obs['cell_type'].value_counts().to_dict()
        high_confidence_count = sum(1 for d in annotation_details if d.get("confidence", 0) >= 0.7)

        result = ToolResult(
            status="success",
            message=f"LLM注释完成，{len(annotation_map)}个clusters，平均置信度{avg_confidence:.2f}",
            data={
                "cluster_key": cluster_key,
                "species": species,
                "n_annotated": len(annotation_map),
                "annotations": annotation_map,
                "cell_type_counts": cell_type_counts,
                "annotation_details": annotation_details,
                "method": "llm_rag" if (use_rag and rag_client) else "llm",
                "average_confidence": avg_confidence,
                "high_confidence_count": high_confidence_count,
                "tissue_context": tissue_context,
            }
        )

        # 保存结果
        if save_result:
            from src.tools.base import _save_result
            result_path = _save_result(adata, "annotated_llm", "cell_annotation", output_dir=data_dir)
            result.artifacts["result_path"] = result_path

            # 保存注释详情
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detail_path = tables_dir / f"annotation_details_llm_{timestamp}.json"
            with open(detail_path, "w", encoding="utf-8") as f:
                json.dump(annotation_details, f, ensure_ascii=False, indent=2)
            result.artifacts["detail_path"] = str(detail_path)

        logger.info(f"LLM注释完成: {len(annotation_map)} clusters")

        return result.to_json()

    except Exception as e:
        logger.error(f"LLM注释失败: {e}")
        return create_tool_result(
            status="error",
            message=f"LLM注释失败: {str(e)}",
            error=str(e)
        )


# 导出工具
__all__ = [
    "annotate_with_llm",
    "CellAnnotationResult",
]
