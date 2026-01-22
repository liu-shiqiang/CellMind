#!/bin/bash
# CellMind 前端启动脚本

echo "启动 CellMind 前端服务..."

# 安装依赖（如果需要）
if [ ! -d "node_modules" ]; then
    echo "安装前端依赖..."
    npm install
fi

# 启动开发服务器
npm run dev
