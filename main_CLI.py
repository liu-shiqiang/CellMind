import os 
import re
import asyncio
import scanpy as sc
import argparse
from pathlib import Path

from src.scripts.utils import get_data_path, read_scrna_data
from src.agent.agent1 import build_graph
from src.scripts.llm_loader import ModelLoader
from config.setting import settings

from rich import print

def parse_args():
    parser = argparse.ArgumentParser(description="Genomix-Agent: AI-Powered Multi-Omics Analysis Platform", add_help=False)
    parser.add_argument('--file', type=str, help='Path to .h5ad file')
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

_PATH_PATTERN = re.compile(r"(?:file_path|data_path)\s*:\s*([^\s]+)")
async def run_agent(objective:str):
    graph = build_graph()
    print(f"Running Objective: {objective}")
    m = _PATH_PATTERN.search(objective)
    path_hint = m.group(1) if m else None

    async for event in graph.astream(
        {"input": objective,
         "plan": [],
         "past_steps": [],
         "last_step_result":path_hint,
         "response":""
         },
        config = {
            "recursion_limit":50,"configurable":{"thread_id":"CLI"},

        }
    ):
        for k,v in event.items():
            if k != "__end__":
                print(f"[bold green]{k}[/bold green] → {v}")


def main():

    args = parse_args()
    user_task = input("Please enter your scRNA seq analysis task (e.g. 'cell type annotation')\n").strip()

    if args.file:
        file_path = Path(args.file).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        full_objective = f"{user_task},data_path: {file_path}"
    else:
        full_objective = user_task

    asyncio.run(run_agent(full_objective))


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


