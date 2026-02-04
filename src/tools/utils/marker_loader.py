"""细胞标记基因加载器

统一加载各种参考数据源：
- common_markers.json: 常见细胞类型标记基因
- CIMA_l3.csv / CIMA_l4.csv: CIMA 数据库标记基因
- blood.csv: 血液细胞标记基因
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from src.web.config import settings

logger = logging.getLogger(__name__)

# 参考数据目录
REFERENCE_DIR = Path(settings.DATA_DIR) / "references" / "cell_markers"


class MarkerLoader:
    """细胞标记基因加载器

    提供统一的接口加载不同来源的标记基因数据
    """

    def __init__(self, reference_dir: Optional[Path] = None):
        """初始化加载器

        Args:
            reference_dir: 参考数据目录，默认使用配置中的目录
        """
        self.reference_dir = Path(reference_dir) if reference_dir else REFERENCE_DIR
        self._common_markers: Optional[Dict] = None
        self._cima_markers: Optional[Dict] = None
        self._blood_markers: Optional[pd.DataFrame] = None

    @property
    def common_markers_file(self) -> Path:
        """常见标记基因文件路径"""
        return self.reference_dir / "common_markers.json"

    @property
    def cima_l3_file(self) -> Path:
        """CIMA L3 标记基因文件路径"""
        return self.reference_dir / "CIMA_l3.csv"

    @property
    def cima_l4_file(self) -> Path:
        """CIMA L4 标记基因文件路径"""
        return self.reference_dir / "CIMA_l4.csv"

    @property
    def blood_markers_file(self) -> Path:
        """血液细胞标记基因文件路径"""
        return self.reference_dir / "blood.csv"

    def load_common_markers(self, species: str = "human") -> Dict[str, List[str]]:
        """加载常见标记基因

        Args:
            species: 物种类型 ("human" 或 "mouse")

        Returns:
            字典形式的标记基因，格式: {cell_type: [gene1, gene2, ...]}
        """
        if self._common_markers is None:
            file_path = self.common_markers_file
            if not file_path.exists():
                logger.warning(f"常见标记基因文件不存在: {file_path}")
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._common_markers = data.get(species, {})

            # 合并免疫细胞和非免疫细胞
            result = {}
            if "immune_cells" in self._common_markers:
                result.update(self._common_markers["immune_cells"])
            if "non_immune_cells" in self._common_markers:
                result.update(self._common_markers["non_immune_cells"])

            self._common_markers = result
            logger.info(f"加载常见标记基因: {len(result)} 种细胞类型")

        return self._common_markers

    def load_cima_markers(
        self,
        level: int = 3
    ) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """加载 CIMA 标记基因

        Args:
            level: 层级级别 (3 或 4)

        Returns:
            字典形式的标记基因，格式:
            {
                cell_type: {
                    "ontology_id": "CL:xxxxx",
                    "markers": [gene1, gene2, ...]
                }
            }
        """
        cache_key = f"cima_l{level}"

        if self._cima_markers is None:
            self._cima_markers = {}

        if cache_key not in self._cima_markers:
            file_path = self.cima_l3_file if level == 3 else self.cima_l4_file
            if not file_path.exists():
                logger.warning(f"CIMA L{level} 标记基因文件不存在: {file_path}")
                self._cima_markers[cache_key] = {}
                return {}

            df = pd.read_csv(file_path)

            result = {}
            for _, row in df.iterrows():
                cell_type = row.iloc[0]  # 第一列是细胞类型名称
                ontology_id = row.get("cell_type_ontology_term_id", "")
                markers_str = row.get("markergene", "")

                # 解析标记基因字符串
                markers = [m.strip() for m in str(markers_str).split(",") if m.strip()]

                result[cell_type] = {
                    "ontology_id": ontology_id,
                    "markers": markers
                }

            self._cima_markers[cache_key] = result
            logger.info(f"加载 CIMA L{level} 标记基因: {len(result)} 种细胞类型")

        return self._cima_markers[cache_key]

    def load_blood_markers(self) -> pd.DataFrame:
        """加载血液细胞标记基因

        Returns:
            血液细胞标记基因 DataFrame
        """
        if self._blood_markers is None:
            file_path = self.blood_markers_file
            if not file_path.exists():
                logger.warning(f"血液细胞标记基因文件不存在: {file_path}")
                self._blood_markers = pd.DataFrame()
                return self._blood_markers

            self._blood_markers = pd.read_csv(file_path)
            logger.info(f"加载血液细胞标记基因: {len(self._blood_markers)} 种细胞类型")

        return self._blood_markers

    def get_marker_dict(
        self,
        cell_type: str,
        source: str = "common",
        species: str = "human",
        level: int = 3
    ) -> List[str]:
        """获取指定细胞类型的标记基因

        Args:
            cell_type: 细胞类型名称
            source: 数据源 ("common", "cima", "blood")
            species: 物种类型（仅用于 source="common"）
            level: CIMA 层级（仅用于 source="cima"）

        Returns:
            标记基因列表
        """
        if source == "common":
            markers = self.load_common_markers(species)
            return markers.get(cell_type, [])
        elif source == "cima":
            markers = self.load_cima_markers(level)
            cell_data = markers.get(cell_type, {})
            return cell_data.get("markers", [])
        elif source == "blood":
            df = self.load_blood_markers()
            if df.empty:
                return []
            row = df[df.iloc[:, 0] == cell_type]
            if not row.empty:
                markers_str = row.iloc[0].get("markergene", "")
                return [m.strip() for m in str(markers_str).split(",") if m.strip()]
            return []
        else:
            logger.warning(f"未知的数据源: {source}")
            return []

    def get_all_cell_types(self, source: str = "common") -> List[str]:
        """获取所有可用的细胞类型

        Args:
            source: 数据源 ("common", "cima", "blood")

        Returns:
            细胞类型列表
        """
        if source == "common":
            markers = self.load_common_markers()
            return list(markers.keys())
        elif source == "cima":
            markers = self.load_cima_markers()
            return list(markers.keys())
        elif source == "blood":
            df = self.load_blood_markers()
            return df.iloc[:, 0].tolist() if not df.empty else []
        else:
            return []

    def calculate_overlap_score(
        self,
        gene_set: Set[str],
        cell_type: str,
        source: str = "common",
        species: str = "human",
        level: int = 3
    ) -> float:
        """计算基因集合与细胞类型标记基因的重叠分数

        Args:
            gene_set: 待比较的基因集合
            cell_type: 细胞类型名称
            source: 数据源
            species: 物种类型
            level: CIMA 层级

        Returns:
            重叠分数 (0-1)
        """
        markers = self.get_marker_dict(cell_type, source, species, level)
        marker_set = set(markers)

        if not marker_set:
            return 0.0

        overlap = gene_set & marker_set
        return len(overlap) / len(marker_set)

    def find_best_match(
        self,
        gene_set: Set[str],
        source: str = "common",
        species: str = "human",
        level: int = 3,
        top_n: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """查找与基因集合最佳匹配的细胞类型

        Args:
            gene_set: 待比较的基因集合
            source: 数据源
            species: 物种类型
            level: CIMA 层级
            top_n: 返回前 N 个最佳匹配

        Returns:
            最佳匹配列表，格式: [{"cell_type": str, "score": float, "markers": List[str]}, ...]
        """
        cell_types = self.get_all_cell_types(source)

        results = []
        for cell_type in cell_types:
            score = self.calculate_overlap_score(gene_set, cell_type, source, species, level)
            if score > 0:
                markers = self.get_marker_dict(cell_type, source, species, level)
                results.append({
                    "cell_type": cell_type,
                    "score": score,
                    "markers": markers
                })

        # 按分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]


# 全局单例
_loader: Optional[MarkerLoader] = None


def get_marker_loader() -> MarkerLoader:
    """获取全局标记基因加载器单例"""
    global _loader
    if _loader is None:
        _loader = MarkerLoader()
    return _loader


# 便捷函数
def load_common_markers(species: str = "human") -> Dict[str, List[str]]:
    """加载常见标记基因"""
    return get_marker_loader().load_common_markers(species)


def load_cima_markers(level: int = 3) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """加载 CIMA 标记基因"""
    return get_marker_loader().load_cima_markers(level)


def load_blood_markers() -> pd.DataFrame:
    """加载血液细胞标记基因"""
    return get_marker_loader().load_blood_markers()


def get_marker_dict(
    cell_type: str,
    source: str = "common",
    species: str = "human",
    level: int = 3
) -> List[str]:
    """获取指定细胞类型的标记基因"""
    return get_marker_loader().get_marker_dict(cell_type, source, species, level)


# 导出
__all__ = [
    "MarkerLoader",
    "get_marker_loader",
    "load_common_markers",
    "load_cima_markers",
    "load_blood_markers",
    "get_marker_dict",
]
