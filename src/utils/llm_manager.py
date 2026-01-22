#!/usr/bin/env python3
"""
统一LLM管理器
集中管理所有LLM实例，支持多种模型和配置
"""

import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from src.web.config import settings

logger = logging.getLogger(__name__)

class LLMManager:
    """统一LLM管理器"""
    
    def __init__(self):
        self._llm_instances: Dict[str, BaseLanguageModel] = {}
        self._default_model = settings.LLM_MODEL
        self._tool_model = getattr(settings, "LLM_TOOL_MODEL", None)
        self._base_url = settings.LLM_BASE_URL
        self._temperature = settings.LLM_TEMPERATURE
        
    def get_llm(self, model_name: Optional[str] = None) -> BaseLanguageModel:
        """
        获取LLM实例
        
        Args:
            model_name: 模型名称，如果为None则使用默认模型
            
        Returns:
            LLM实例
        """
        model_name = model_name or self._default_model
        
        if model_name not in self._llm_instances:
            self._llm_instances[model_name] = self._create_llm(model_name)
            
        return self._llm_instances[model_name]
    
    def get_tool_llm(self) -> BaseLanguageModel:
        """
        获取工具调用专用LLM
        
        Returns:
            工具调用LLM实例
        """
        tool_model = self._tool_model or self._default_model
        return self.get_llm(tool_model)
    
    def _create_llm(self, model_name: str) -> BaseLanguageModel:
        """
        创建LLM实例
        
        Args:
            model_name: 模型名称
            
        Returns:
            LLM实例
        """
        try:
            # 根据模型名称判断类型
            if model_name.startswith(("gpt-", "claude-", "gemini-")):
                # OpenAI/Anthropic/Google模型
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=self._temperature,
                    base_url=self._base_url if self._base_url != "http://localhost:11434" else None
                )
            else:
                # Ollama模型
                llm = ChatOllama(
                    model=model_name,
                    temperature=self._temperature,
                    base_url=self._base_url
                )
            
            logger.info(f"创建LLM实例: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"创建LLM实例失败: {model_name}, 错误: {e}")
            # 降级到默认Ollama模型
            return ChatOllama(
                model="qwen3:8b",
                temperature=self._temperature,
                base_url="http://localhost:11434"
            )
    
    def get_llm_with_tools(self, tools: list) -> BaseLanguageModel:
        """
        获取绑定工具的LLM
        
        Args:
            tools: 工具列表
            
        Returns:
            绑定工具的LLM实例
        """
        tool_llm = self.get_tool_llm()
        return tool_llm.bind_tools(tools)
    
    def get_structured_llm(self, model_name: Optional[str] = None, output_schema: Any = None):
        """
        获取结构化输出LLM
        
        Args:
            model_name: 模型名称
            output_schema: 输出模式
            
        Returns:
            结构化输出LLM实例
        """
        llm = self.get_llm(model_name)
        if output_schema:
            return llm.with_structured_output(output_schema)
        return llm
    
    def clear_cache(self):
        """清除LLM实例缓存"""
        self._llm_instances.clear()
        logger.info("清除LLM实例缓存")
    
    def list_available_models(self) -> list:
        """
        列出可用的模型
        
        Returns:
            可用模型列表
        """
        return list(self._llm_instances.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        LLM健康检查
        
        Returns:
            健康状态信息
        """
        health_status = {
            "status": "healthy",
            "models": {},
            "errors": []
        }
        
        for model_name, llm in self._llm_instances.items():
            try:
                # 简单测试
                response = llm.invoke("Hello")
                health_status["models"][model_name] = {
                    "status": "healthy",
                    "response_length": len(str(response))
                }
            except Exception as e:
                health_status["models"][model_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["errors"].append(f"{model_name}: {e}")
                health_status["status"] = "unhealthy"
        
        return health_status

# 全局LLM管理器实例
@lru_cache()
def get_llm_manager() -> LLMManager:
    """获取全局LLM管理器实例"""
    return LLMManager()

# 便捷函数
def get_llm(model_name: Optional[str] = None) -> BaseLanguageModel:
    """获取LLM实例"""
    return get_llm_manager().get_llm(model_name)

def get_tool_llm() -> BaseLanguageModel:
    """获取工具调用LLM"""
    return get_llm_manager().get_tool_llm()

def get_llm_with_tools(tools: list) -> BaseLanguageModel:
    """获取绑定工具的LLM"""
    return get_llm_manager().get_llm_with_tools(tools)

def get_structured_llm(model_name: Optional[str] = None, output_schema: Any = None):
    """获取结构化输出LLM"""
    return get_llm_manager().get_structured_llm(model_name, output_schema) 