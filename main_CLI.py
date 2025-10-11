# coding=utf-8

import os 
import re
import asyncio
import scanpy as sc
import argparse
from pathlib import Path
from typing import List, Optional

from src.scripts.utils import get_data_path, read_scrna_data
from src.agent.agent_new import build_graph, run_objective
from src.scripts.llm_loader import ModelLoader
from config.setting import settings
from src.utils.path_manager import validate_h5ad_file

from rich import print

def parse_args():
    parser = argparse.ArgumentParser(description="Genomix-Agent: AI-Powered Multi-Omics Analysis Platform", add_help=False)
    parser.add_argument('--file', type=str, help='Path to .h5ad file')
    parser.add_argument('--files',nargs='+', help = 'Mutiple .h5ad files')
    parser.add_argument('--model',
                        help='name of model engine, e.g. gpt-4, please refer to model zoo',
                        default='ollama_deepseek-r1:14b')
    parser.add_argument('--gui_mode',
                        default=False,
                        type=bool)
    parser.add_argument('--rag',
                        help='Use RAG or not',
                        default=False,
                        type=bool)
    return parser.parse_args()



def validate_input_files(file_paths: List[str]) -> List[str]:
    """
    验证输入文件列表
    
    Args:
        file_paths: 文件路径列表
        
    Returns:
        验证通过的文件路径列表
        
    Raises:
        FileNotFoundError: 文件不存在或格式不正确
    """
    validated_files = []
    
    for file_path in file_paths:
        try:
            path_info = validate_h5ad_file(file_path)
            print(path_info)
            validated_files.append(str(path_info.resolved_path))
            print(f"Validation Documentation: {path_info.file_name} ({path_info.size})")
        except Exception as e:
            print(f"File verification failed: {file_path} - {e}")
            raise FileNotFoundError(f"File verification failed: {file_path}")
    
    return validated_files

def get_files_interactively() -> List[str]:
    """
    交互式获取文件路径
    
    Returns:
        文件路径列表
    """
    files = []
    print("Please enter the file path (enter 'done' to complete, 'q' to exit):")
    
    while True:
        file_path = input(f"file {len(files) + 1}: ").strip()
        
        if file_path.lower() == 'q':
            print("User cancels operation")
            return []
        
        if file_path.lower() == 'done':
            if not files:
                print("Please enter at least one file path")
                continue
            break
        
        if file_path:
            try:
                path_info = validate_h5ad_file(file_path)
                files.append(str(path_info.resolved_path))
                print(f"add file: {path_info.file_name}")
            except Exception as e:
                print(f"Invalid file: {e}")
                print("Please re-enter or enter 'q' to exit")
    
    return files



def main():
    args = parse_args()
    
    # 获取用户任务描述
    print("\n Please enter the analysis task description (e.g. 'Cell Type Annotation', 'Pathway Enrichment Analysis',' Regulatory Network Inference '):")
    user_task = input("task: ").strip()
    
    if not user_task:
        print("Task description cannot be empty")
        return
    
    # 获取输入文件
    input_files = []
    
    # 方式1: 命令行参数
    if args.file:
        input_files = [args.file]
    elif args.files:
        input_files = args.files
    
    # 方式2: 交互式输入
    if not input_files:
        print("\nPlease choose a file input method:")
        print("1. Interactive input file path")
        print("2. Use default test files")
        print("3. Skip file input (for testing purposes only)")
        
        choice = input("choose (1/2/3): ").strip()
        
        if choice == "1":
            input_files = get_files_interactively()
        elif choice == "2":
            # 使用默认测试文件（需要根据实际情况调整）
            default_file = "./data/test.h5ad"
            if os.path.exists(default_file):
                input_files = [default_file]
                print(f"Use default file: {default_file}")
            else:
                print(f"The default file does not exist: {default_file}")
                input_files = get_files_interactively()
        elif choice == "3":
            print("Skip file input, only for testing purposes")
        else:
            print("Invalid selection, using interactive input")
            input_files = get_files_interactively()
    
    # 验证文件
    if input_files:
        try:
            validated_files = validate_input_files(input_files)
            print(f"\nInput File ({len(validated_files)} 个):")
            for i, file_path in enumerate(validated_files, 1):
                print(f"  {i}. {file_path}")
        except FileNotFoundError as e:
            print(f"{e}")
            return
    else:
        validated_files = []
    
    # 执行任务
    print(f"\nStart executing the task: {user_task}")
    if validated_files:
        print(f"User files: {len(validated_files)} ")
    
    try:
        result = asyncio.run(run_objective(user_task, validated_files))
        print(f"\nMission accomplished!")
        print(f"📊 结果: {result}")
    except Exception as e:
        print(f"\nTask execution failed: {e}")


if __name__ == '__main__':

    print("\n" + "=" * 70)
    print(" 🎉  Welcome to Genomix-Agent: AI-Powered Multi-Omics Analysis Platform 🎉 ")
    print("=" * 70)
    print(r"""
   ██████╗ ███████╗███╗   ██╗ ██████╗ ███╗   ███╗██╗██╗  ██╗
  ██╔════╝ ██╔════╝████╗  ██║██╔═══██╗████╗ ████║██║╚██╗██╔╝
  ██║  ███╗█████╗  ██╔██╗ ██║██║   ██║██╔████╔██║██║ ╚███╔╝ 
  ██║   ██║██╔══╝  ██║╚██╗██║██║   ██║██║╚██╔╝██║██║ ██╔██╗ 
  ╚██████╔╝███████╗██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║██╔╝ ██╗
   ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝
    """)
    print("=" * 80 + "\n")
    while True:
        main()
        if input("Do you want to continue using Genomix-Agent?(y/n)\n").strip().lower() != 'y':
            break
        print("\n" + "=" * 70)
    
    print("👋 Thank you for using Genomix-Agent!")


