# 🚀 Genomix Agent 微服务启动指南

## 架构说明

本项目采用微服务架构,将 scGPT 功能独立为单独的服务:

- **主服务 (Port 8000)**: FastAPI + LangGraph + LangChain
- **scGPT 服务 (Port 8001)**: 独立的 scGPT embeddings 提取服务

---

## 📦 本地开发启动

### 方式 1: 手动启动 (开发推荐)

#### 1. 启动 scGPT 服务
```bash
# 创建独立的 conda 环境 (可选)
conda create -n genomix-scgpt python=3.10
conda activate genomix-scgpt

# 安装 scGPT 服务依赖
pip install -r requirements-scgpt.txt

# 启动服务
uvicorn src.scgpt_service.main:app --reload --host 0.0.0.0 --port 8001
```

#### 2. 启动主服务 (新终端)
```bash
# 使用主环境
conda activate genomix

# 安装主服务依赖
pip install -r requirements-main.txt

# 启动服务
uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. 启动 Streamlit UI (可选,新终端)
```bash
conda activate genomix
streamlit run ui/app.py
```

---

### 方式 2: Docker Compose (生产推荐)

```bash
# 构建并启动所有服务
docker-compose up --build

# 后台运行
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 🔍 验证服务状态

### 检查主服务
```bash
curl http://localhost:8000/docs
```

### 检查 scGPT 服务
```bash
curl http://localhost:8001/health
```

**预期响应**:
```json
{
  "status": "healthy",
  "service": "scgpt",
  "version": "1.0.0",
  "scgpt_available": true
}
```

---

## 🔧 配置说明

### 环境变量

**主服务 (.env)**:
```bash
# scGPT 服务地址
SCGPT_SERVICE_URL=http://localhost:8001

# API 地址
GENOMIX_AGENT_API=http://localhost:8000
```

**scGPT 服务 (.env.scgpt)**:
```bash
# 服务端口
SCGPT_SERVICE_PORT=8001
```

---

## 📝 使用示例

### API 调用示例

#### 1. 提取 scGPT embeddings

```bash
curl -X POST "http://localhost:8001/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/data.h5ad",
    "model_name": "scgpt"
  }'
```

#### 2. 使用主服务 (自动调用 scGPT)

```bash
curl -X POST "http://localhost:8000/api/v2/agent/run" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "提取细胞 embeddings",
    "input_files": ["/path/to/data.h5ad"]
  }'
```

---

## 🐛 故障排查

### 问题 1: scGPT 服务连接失败
```
错误: 无法连接到 scGPT 服务
解决:
1. 检查 scGPT 服务是否启动: curl http://localhost:8001/health
2. 检查环境变量: echo $SCGPT_SERVICE_URL
3. 查看 scGPT 服务日志
```

### 问题 2: torch/torchtext 版本冲突
```
错误: Symbol not found: __ZN3c105ErrorC1E...
解决:
1. 主服务: pip install -r requirements-main.txt
2. scGPT 服务: pip install -r requirements-scgpt.txt
3. 确保使用独立的 conda 环境
```

### 问题 3: Docker 内存不足
```bash
# 增加 Docker 内存限制
docker-compose down
# 在 Docker 设置中增加内存到至少 8GB
docker-compose up
```

---

## 📊 服务监控

### 主服务 API 文档
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### scGPT 服务 API 文档
- Swagger UI: http://localhost:8001/docs

---

## 🎯 优势

✅ **依赖隔离**: 主服务和 scGPT 服务使用不同的 PyTorch 版本
✅ **独立扩展**: scGPT 服务可以单独扩展资源
✅ **故障隔离**: scGPT 服务崩溃不影响主服务
✅ **开发效率**: 主服务启动不再受 scGPT 依赖影响
✅ **部署灵活**: 可以将 scGPT 服务部署到 GPU 服务器

---

## 📚 更多信息

- LangGraph 文档: https://docs.langchain.com/langgraph
- scGPT 仓库: https://github.com/bowang-lab/scGPT
