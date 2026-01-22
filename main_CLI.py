# coding=utf-8

import os
import json
import asyncio
import argparse
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import BaseMessage

from src.scripts.utils import get_data_path
from src.utils.path_manager import validate_h5ad_file
from src.utils.langgraph_stream import run_agent_stream, serialize_message
from src.web.config import settings

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
    parser.add_argument('--stream',
                        help='Stream planner/executor events in real time',
                        action='store_true')
    parser.add_argument('--thread_id',
                        help='Optional conversation thread identifier',
                        default=None)
    return parser.parse_args()


def main(persistent_thread_id: Optional[str] = None) -> Optional[str]:
    args = parse_args()

    # 获取用户任务描述
    print("\n Please enter the analysis task description (e.g. 'Cell Type Annotation', 'Pathway Enrichment Analysis',' Regulatory Network Inference '):")
    # print('stdin encoding:', sys.stdin.encoding)

    user_task = input("task: ").strip()
    print("Received:", repr(user_task))

    if not user_task:
        print("Task description cannot be empty")
        return persistent_thread_id
    
    # 方式1: 命令行参数
    if args.file:
        input_file = args.file

    # 方式2: 交互式输入
    if not input_file:
        print("\nPlease choose a file input method:")
        print("1. Interactive input file path")
        print("2. Skip file input (only want to chat with the agent)")
        choice = input("choose (1/2): ").strip()
        
        if choice == "1":
            input_file = get_data_path()
        elif choice == "2":
            print("Skip file input, only for chatting with the agent")
        else:
            print("Invalid selection, using interactive input")
            input_file = get_data_path()
    
    # 验证文件
    if input_file:
        try:
            validated_file = validate_h5ad_file(input_file)
            print(f"\nInput File: {validated_file}")
        except FileNotFoundError as e:
            print(f"{e}")
            return persistent_thread_id
    else:
        validated_file = None

    resolved_thread_id = (args.thread_id or persistent_thread_id or str(uuid4())).strip()

    # 执行任务
    print(f"\nStart executing the task: {user_task}")
    raw_input_path = None
    if validated_file:
        raw_input_path = str(validated_file.resolved_path)
        print(f"User file: {validated_file} ")

    def _prepare_input_files(run_id: str) -> Optional[List[str]]:
        if not raw_input_path:
            return None
        run_upload_dir = Path(settings.RUNS_DIR) / run_id / "uploads"
        run_upload_dir.mkdir(parents=True, exist_ok=True)
        target_path = run_upload_dir / Path(raw_input_path).name
        shutil.copy2(raw_input_path, target_path)
        return [str(target_path)]

    if args.stream:
        print("\n" + "=" * 70)
        print("📡 Streaming mode enabled")
        print("=" * 70 + "\n")

        async def _stream_objective() -> Tuple[Optional[BaseMessage], Optional[Any]]:
            run_id = str(uuid4())
            input_files = _prepare_input_files(run_id)

            # Enhanced event handler for better UX
            async def _printer(event: dict) -> None:
                event_type = event.get("type", "unknown")

                # Format events for better readability
                if event_type == "start":
                    payload = event.get("payload", {})
                    print(f"🚀 {payload.get('message', 'Starting...')}")
                    print(f"   Objective: {payload.get('objective', 'N/A')}")

                elif event_type == "node_enter":
                    payload = event.get("payload", {})
                    node = event.get("node", "unknown")
                    msg = payload.get("message", f"Executing: {node}")
                    print(f"⚙️  {msg}")

                elif event_type == "plan_update":
                    payload = event.get("payload", {})
                    msg = payload.get("message", "Plan updated")
                    plan = payload.get("plan", [])
                    print(f"📋 {msg}")
                    for i, step in enumerate(plan, 1):
                        print(f"   {i}. {step}")

                elif event_type == "tool_call":
                    payload = event.get("payload", {})
                    msg = payload.get("message", "Calling tools")
                    print(f"🔧 {msg}")

                elif event_type == "tool_result":
                    payload = event.get("payload", {})
                    msg = payload.get("message", "Tool result received")
                    print(f"✅ {msg}")

                elif event_type == "token":
                    # Real-time LLM token streaming
                    payload = event.get("payload", {})
                    token = payload.get("token", "")
                    # Print tokens without newline for streaming effect
                    print(token, end="", flush=True)

                elif event_type == "end":
                    payload = event.get("payload", {})
                    msg = payload.get("message", "Complete")
                    print(f"\n{msg}")

                elif event_type == "error":
                    payload = event.get("payload", {})
                    detail = payload.get("detail", "Unknown error")
                    print(f"\n❌ Error: {detail}")

                else:
                    # Fallback: print full event JSON for debugging
                    print(json.dumps(event, ensure_ascii=False, indent=2))

            final_message, error_info = await run_agent_stream(
                objective=user_task,
                input_files=input_files,
                thread_id=resolved_thread_id,
                run_id=run_id,
                event_handler=_printer,
                stream_mode="updates",  # Use updates mode for progress tracking
            )
            return final_message, error_info

        try:
            final_message, error_info = asyncio.run(_stream_objective())
        except Exception as e:
            print(f"\n❌ Task execution failed: {e}")
            return persistent_thread_id or resolved_thread_id

        if final_message is not None:
            print("\n" + "=" * 70)
            print("✅ Mission accomplished!")
            print("=" * 70)
            print("\n📊 Final Result:")
            print(json.dumps(serialize_message(final_message), ensure_ascii=False, indent=2))
        elif error_info is not None:
            print("\n" + "=" * 70)
            print("❌ Task execution failed")
            print("=" * 70)
            print(json.dumps(error_info, ensure_ascii=False, indent=2))
        else:
            print("\n⚠️  Mission completed without a final response message.")
    else:
        async def _run_objective() -> Tuple[Optional[BaseMessage], Optional[Any]]:
            run_id = str(uuid4())
            input_files = _prepare_input_files(run_id)
            async def _noop(_: dict) -> None:
                return None

            return await run_agent_stream(
                objective=user_task,
                input_files=input_files,
                thread_id=resolved_thread_id,
                run_id=run_id,
                event_handler=_noop,
                stream_mode="updates",
            )

        try:
            final_message, error_info = asyncio.run(_run_objective())
        except Exception as e:
            print(f"\nTask execution failed: {e}")
            return persistent_thread_id or resolved_thread_id

        if final_message is not None:
            print(f"\nMission accomplished!")
            print(json.dumps(serialize_message(final_message), ensure_ascii=False, indent=2))
        elif error_info is not None:
            print(f"\nTask execution failed")
            print(json.dumps(error_info, ensure_ascii=False, indent=2))

    return resolved_thread_id

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
    persistent_thread_id: Optional[str] = None
    while True:
        result_thread_id = main(persistent_thread_id)
        if result_thread_id:
            persistent_thread_id = result_thread_id
        if input("Do you want to continue using Genomix-Agent?(y/n)\n").strip().lower() != 'y':
            break
        print("\n" + "=" * 70)

    print("👋 Thank you for using Genomix-Agent!")
