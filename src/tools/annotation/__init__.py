"""细胞类型注释工具

提供多种注释方法：
- 简单标记基因匹配 (annotate_with_simple_markers)
- CIMA 参考文件匹配 (annotate_with_cima_markers)
- 血液细胞标记匹配 (annotate_with_blood_markers)
- LLM + RAG 智能注释 (annotate_with_llm)
"""
try:
    from src.tools.annotation.marker_based import (
        annotate_with_simple_markers,
        annotate_with_cima_markers,
        annotate_with_blood_markers,
    )
    _has_marker_based_tools = True
except ImportError as e:
    annotate_with_simple_markers = None
    annotate_with_cima_markers = None
    annotate_with_blood_markers = None
    _has_marker_based_tools = False

try:
    from src.tools.annotation.llm_annotate import (
        annotate_with_llm,
    )
    _has_llm_tool = True
except ImportError as e:
    annotate_with_llm = None
    _has_llm_tool = False

__all__ = [
    "annotate_with_simple_markers",
    "annotate_with_cima_markers",
    "annotate_with_blood_markers",
    "annotate_with_llm",
]
