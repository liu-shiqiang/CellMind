import os 

import scanpy as sc
import argparse

from src.scripts.utils import get_data_path, read_scrna_data
from src.agent.planner_executor import Agent
from src.scripts.llm_loader import ModelLoader
from config.setting import settings

def main(model_name, openai, chroma_path, gui_mode, cpu, rag):
    
    user_task = input("Please enter your scRNA seq analysis task (e.g. 'cell type annotation')\n").strip()
    agent = Agent(model_name, openai, chroma_path, gui_mode, cpu, rag)
    result = agent.invoke(user_task)
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Genomix-Agent: AI-Powered Multi-Omics Analysis Platform", add_help=False)
    parser.add_argument('--openai',
                        help='openai api',
                        default='SET_YOUR_OPENAI_API')
    parser.add_argument('--model',
                        help='name of model engine, e.g. gpt-4, please refer to model zoo',
                        default='ollama_deepseek-r1:14b')
    parser.add_argument('--chroma_path',
                        help='Persist the data path of ChromaDB',
                        default='/home/share/huadjyin/home/liushiqiang/Projects/bio-llm-platform/backend/persist_chroma',
                        type=str)
    parser.add_argument('--gui_mode',
                        default=False,
                        type=bool)
    parser.add_argument('--cpu',
                        default=False,
                        type=bool)
    parser.add_argument('--rag',
                        help='Use RAG or not',
                        default=False,
                        type=bool)
    args = parser.parse_args()
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
        main(args.model, args.openai, args.chroma_path, args.gui_mode, args.cpu, args.rag)
        if input("Do you want to continue using Genomix-Agent?(y/n)\n").strip().lower() != 'y':
            break
        print("\n" + "=" * 70)
        print("👋 Thank you for using Genomix-Agent!")


