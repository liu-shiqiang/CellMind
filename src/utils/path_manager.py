#!/usr/bin/env python3
"""
统一路径管理器
提供路径解析、验证、工作目录管理等功能
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PathInfo:
    """路径信息数据类"""
    original_path: str
    resolved_path: Path
    file_name: str
    file_stem: str
    file_suffix: str
    parent_dir: Path
    exists: bool
    is_file: bool
    is_dir: bool
    size: Optional[int] = None

class PathManager:
    """统一路径管理器"""
    
    def __init__(self, base_output_dir: str = "/home/share/huadjyin/home/liushiqiang/Projects/genomix-agent/output/test_output"):
        self.base_output_dir = Path(base_output_dir).expanduser().resolve()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 路径提取模式
        self.path_patterns = [
            r'(?:file_path|data_path|input_file|adata_path)\s*:\s*["\']?([^"\s]+)["\']?',
            r'(?:file_path|data_path|input_file|adata_path)\s*=\s*["\']?([^"\s]+)["\']?',
            r'([a-zA-Z]:\\[^\s]+\.h5ad)',  # Windows绝对路径
            r'(/[^\s]+\.h5ad)',  # Unix绝对路径
            r'(\./[^\s]+\.h5ad)',  # 相对路径
        ]
        
        # 支持的文件格式
        self.supported_formats = {
            '.h5ad': 'AnnData文件',
            '.csv': 'CSV文件',
            '.tsv': 'TSV文件',
            '.txt': '文本文件',
            '.json': 'JSON文件',
            '.png': 'PNG图像',
            '.jpg': 'JPEG图像',
            '.pdf': 'PDF文档'
        }
    
    def extract_paths_from_text(self, text: str) -> List[str]:
        """
        从文本中提取所有可能的文件路径
        
        Args:
            text: 包含路径的文本
            
        Returns:
            提取到的路径列表
        """
        paths = []
        for pattern in self.path_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            paths.extend(matches)
        
        # 去重并过滤
        unique_paths = list(set(paths))
        valid_paths = [p for p in unique_paths if self._is_valid_path(p)]
        
        logger.info(f"从文本中提取到 {len(valid_paths)} 个有效路径: {valid_paths}")
        return valid_paths
    
    def parse_path(self, path_str: str) -> PathInfo:
        """
        解析路径并返回详细信息
        
        Args:
            path_str: 路径字符串
            
        Returns:
            PathInfo对象
        """
        try:
            # 处理路径字符串
            path_str = path_str.strip().strip('"\'')
            resolved_path = Path(path_str).expanduser().resolve()
            
            # 获取文件信息
            path_info = PathInfo(
                original_path=path_str,
                resolved_path=resolved_path,
                file_name=resolved_path.name,
                file_stem=resolved_path.stem,
                file_suffix=resolved_path.suffix.lower(),
                parent_dir=resolved_path.parent,
                exists=resolved_path.exists(),
                is_file=resolved_path.is_file() if resolved_path.exists() else False,
                is_dir=resolved_path.is_dir() if resolved_path.exists() else False,
                size=resolved_path.stat().st_size if resolved_path.exists() else None
            )
            
            logger.debug(f"解析路径: {path_str} -> {path_info}")
            return path_info
            
        except Exception as e:
            logger.error(f"路径解析失败: {path_str}, 错误: {e}")
            raise ValueError(f"无效的路径: {path_str}")
    
    def validate_input_file(self, path_str: str, expected_formats: Optional[List[str]] = None) -> PathInfo:
        """
        验证输入文件
        
        Args:
            path_str: 文件路径
            expected_formats: 期望的文件格式列表
            
        Returns:
            验证通过的PathInfo对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        path_info = self.parse_path(path_str)
        
        if not path_info.exists:
            raise FileNotFoundError(f"文件不存在: {path_str}")
        
        if not path_info.is_file:
            raise ValueError(f"路径不是文件: {path_str}")
        
        if expected_formats:
            if path_info.file_suffix not in expected_formats:
                raise ValueError(f"不支持的文件格式: {path_info.file_suffix}, 期望: {expected_formats}")
        
        logger.info(f"文件验证通过: {path_str}")
        return path_info
    
    def create_work_dir(self, base_name: str, task_type: str = "analysis") -> Path:
        """
        创建工作目录
        
        Args:
            base_name: 基础名称（通常来自输入文件名）
            task_type: 任务类型
            
        Returns:
            工作目录路径
        """
        # 清理基础名称
        safe_base_name = self._sanitize_filename(base_name)
        
        # 创建工作目录
        work_dir = self.base_output_dir / task_type / safe_base_name
        work_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建工作目录: {work_dir}")
        return work_dir
    
    def generate_output_path(self, work_dir: Path, file_name: str, suffix: str = "") -> Path:
        """
        生成输出文件路径
        
        Args:
            work_dir: 工作目录
            file_name: 文件名
            suffix: 后缀（可选）
            
        Returns:
            输出文件路径
        """
        safe_file_name = self._sanitize_filename(file_name)
        if suffix:
            safe_file_name = f"{safe_file_name}_{suffix}"
        
        output_path = work_dir / safe_file_name
        logger.debug(f"生成输出路径: {output_path}")
        return output_path
    
    def get_relative_path(self, path: Path, base_dir: Optional[Path] = None) -> str:
        """
        获取相对路径
        
        Args:
            path: 目标路径
            base_dir: 基准目录（默认为当前工作目录）
            
        Returns:
            相对路径字符串
        """
        if base_dir is None:
            base_dir = Path.cwd()
        
        try:
            relative_path = path.relative_to(base_dir)
            return str(relative_path)
        except ValueError:
            return str(path)
    
    def ensure_dir_exists(self, dir_path: Union[str, Path]) -> Path:
        """
        确保目录存在
        
        Args:
            dir_path: 目录路径
            
        Returns:
            目录路径对象
        """
        dir_path = Path(dir_path).expanduser().resolve()
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    def copy_file_to_work_dir(self, source_path: Path, work_dir: Path, new_name: Optional[str] = None) -> Path:
        """
        复制文件到工作目录
        
        Args:
            source_path: 源文件路径
            work_dir: 工作目录
            new_name: 新文件名（可选）
            
        Returns:
            目标文件路径
        """
        if new_name is None:
            new_name = source_path.name
        
        target_path = work_dir / new_name
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)
        
        logger.info(f"复制文件: {source_path} -> {target_path}")
        return target_path
    
    def _is_valid_path(self, path_str: str) -> bool:
        """检查路径是否有效"""
        try:
            path = Path(path_str)
            return len(path_str) > 0 and not path_str.isspace()
        except:
            return False
    
    def _sanitize_filename(self, filename: str) -> str:
        """清理文件名，移除不安全的字符"""
        # 移除或替换不安全的字符
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # 移除多余的空格和点
        filename = filename.strip(' .')
        
        # 限制长度
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename
    
    def get_file_info_summary(self, path_info: PathInfo) -> Dict:
        """获取文件信息摘要"""
        return {
            "file_name": path_info.file_name,
            "file_size": self._format_file_size(path_info.size) if path_info.size else "Unknown",
            "file_type": self.supported_formats.get(path_info.file_suffix, "Unknown"),
            "absolute_path": str(path_info.resolved_path),
            "relative_path": self.get_relative_path(path_info.resolved_path)
        }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes is None:
            return "Unknown"
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

# 全局路径管理器实例
path_manager = PathManager()

# 便捷函数
def extract_paths_from_objective(objective: str) -> List[str]:
    """从目标描述中提取路径"""
    return path_manager.extract_paths_from_text(objective)

def validate_h5ad_file(file_path: str) -> PathInfo:
    """验证h5ad文件"""
    return path_manager.validate_input_file(file_path, ['.h5ad'])

def create_analysis_work_dir(base_name: str) -> Path:
    """创建分析工作目录"""
    return path_manager.create_work_dir(base_name, "analysis") 