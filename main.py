import os 

import scanpy as sc
import argparse

from src.utils import get_data_path, read_scrna_data
from src.planner import Planner
from src.executor import Executor
from src.llm_loader import ModelLoader
from config.setting import settings

def main(model_name, openai, chroma_path, gui_mode, cpu, rag):
    
    
    user_task = input("Please enter your scRNA seq analysis task (e.g. 'cell type annotation')\n").strip()
    data_path = get_data_path()
    if data_path != None:
        adata = read_scrna_data(data_path)
        data_representation = str(adata)
        print(data_representation)
        planner = Planner(model_name,user_task, data_path, data_representation)
        user_task = planner.plan()
        data_basename_head = os.path.splitext(os.path.basename(data_path))[0]
        output_path = os.path.join(settings.OUTPUT_DIR, data_basename_head)
        executor = Executor(model_name,user_task,output_path)
        executor.task_excute()
        
    else:
        llm = ModelLoader(model_name).load_model()
        print(llm.invoke(user_task))


    



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


