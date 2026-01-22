#!/bin/bash
# CellMind 后端启动脚本

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:."
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-test}"

echo "启动 CellMind 后端服务..."

# 启动服务
uvicorn src.web.main:app --host 0.0.0.0 --port 8000 --reload
