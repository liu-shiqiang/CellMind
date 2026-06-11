# 🧬 CellMind: AI-Powered Single-Cell Multi-Omics Analysis Platform

[![Python](https://img.shields.io/badge/Python-76.4%25-blue)]()
[![TypeScript](https://img.shields.io/badge/TypeScript-23.5%25-green)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Orchestration-FF6B6B)]()

**CellMind** is an intelligent platform for single-cell multi-omics analysis that leverages Large Language Models (LLMs) and agentic AI to automate complex biological data analysis workflows. It combines scGPT embeddings with LangGraph-based reasoning agents to provide intuitive, AI-driven insights from single-cell RNA-seq and related genomic data.

---

## ✨ Key Features

- 🤖 **AI-Powered Agent System**: LLM-driven multi-step reasoning agents orchestrated via LangGraph
- 🧪 **Multi-Omics Analysis**: Support for single-cell RNA-seq (scRNA-seq) analysis with scGPT embeddings
- 🔌 **Microservices Architecture**: Independent scGPT embedding service for scalability
- 🚀 **Real-time Streaming**: Stream analysis progress and LLM outputs in real-time
- 📊 **Data Visualization**: Interactive plots and results using Plotly and Streamlit
- 🔍 **RAG Integration**: Retrieval-Augmented Generation with Chroma vector store
- 🐳 **Docker Support**: Production-ready Docker Compose setup
- 🗣️ **Conversational Interface**: Multi-turn conversations with persistent thread support

---

## 🏗️ Architecture

CellMind uses a **microservices architecture** to isolate dependencies and enable independent scaling:

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │
│  │ CLI (Python) │  │ Web UI (API) │  │ Streamlit App  │    │
│  └──────────────┘  └──────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   Main Service (Port 8000)            │
        │  FastAPI + LangGraph + LangChain      │
        │  ┌─────────────────────────────────┐  │
        │  │ LangGraph Agent Orchestrator    │  │
        │  │ ├─ Planner                      │  │
        │  │ ├─ Executor                     │  │
        │  │ └─ Reasoner                     │  │
        │  └─────────────────────────────────┘  │
        │  ┌─────────────────────────────────┐  │
        │  │ LLM Integration                 │  │
        │  │ ├─ OpenAI (GPT-4/3.5)           │  │
        │  │ ├─ Ollama (Local Models)        │  │
        │  │ └─ HuggingFace Models           │  │
        │  └─────────────────────────────────┘  │
        └────────────────┬──────────────────────┘
                         ↓
        ┌────────────────────────────────┐
        │ scGPT Service (Port 8001)      │
        │ Single-Cell Embedding Engine   │
        │ ├─ scGPT Model                 │
        │ ├─ scANpy Integration          │
        │ └─ H5AD Processing             │
        └────────────────────────────────┘
```

### Components

| Service | Port | Purpose | Framework |
|---------|------|---------|-----------|
| **Main Service** | 8000 | API & Agent Orchestration | FastAPI + LangGraph |
| **scGPT Service** | 8001 | Cell Embedding Generation | FastAPI + scGPT |

**Why Microservices?**
- 🔓 **Dependency Isolation**: Different PyTorch versions (Main: 2.5.x, scGPT: 2.1.x)
- 📈 **Independent Scaling**: Scale scGPT service separately for GPU-bound workloads
- 🛡️ **Fault Isolation**: Service failure doesn't cascade
- ⚡ **Faster Development**: Main service startup unaffected by scGPT dependencies

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 18+** (for TypeScript components)
- **Docker & Docker Compose** (optional, for production)
- **CUDA 11.8+** (optional, for GPU acceleration)

### Option 1: Local Development (Recommended)

#### Step 1: Clone & Setup Environment

```bash
git clone https://github.com/liu-shiqiang/CellMind.git
cd CellMind

# Create main environment
conda create -n cellmind python=3.10
conda activate cellmind

# Create scGPT environment (optional, for development)
conda create -n cellmind-scgpt python=3.10
```

#### Step 2: Start scGPT Service (In separate terminal)

```bash
conda activate cellmind-scgpt

# Install scGPT dependencies
pip install -r requirements-scgpt.txt

# Start the service
uvicorn src.scgpt_service.main:app --reload --host 0.0.0.0 --port 8001
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete
```

#### Step 3: Start Main Service (In new terminal)

```bash
conda activate cellmind

# Install main dependencies
pip install -r requirements-main.txt

# Start the main service
uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
```

#### Step 4: Run CLI or Web UI

**CLI Mode (Interactive)**:
```bash
python main_CLI.py --stream

# Follow prompts:
# 1. Enter analysis task (e.g., "Cell Type Annotation")
# 2. Provide H5AD file or skip
# 3. Watch real-time progress with streaming events
```

**Web UI (Optional)**:
```bash
streamlit run ui/app.py
# Opens at http://localhost:8501
```

### Option 2: Docker Compose (Production)

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📝 Usage

### CLI Examples

#### Basic Single-Cell Analysis

```bash
# Start interactive CLI
python main_CLI.py --stream

# Enter task: "Annotate cell types"
# Upload file: /path/to/data.h5ad
# View real-time analysis progress
```

#### Specify Model

```bash
python main_CLI.py \
  --file /path/to/data.h5ad \
  --model gpt-4 \
  --stream

# Alternative models:
# - gpt-4 (OpenAI)
# - gpt-3.5-turbo
# - ollama_deepseek-r1:14b (local)
# - mistral (local via Ollama)
```

#### With Persistence

```bash
python main_CLI.py --thread_id my-session-id --stream

# Same thread_id maintains conversation history
# Useful for iterative analysis
```

### API Examples

#### Embed Cells via scGPT

```bash
curl -X POST "http://localhost:8001/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/app/data/sample.h5ad",
    "model_name": "scgpt"
  }'
```

#### Run Analysis via Main API

```bash
curl -X POST "http://localhost:8000/api/v2/agent/run" \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Identify marker genes for each cell type",
    "input_files": ["/app/data/sample.h5ad"],
    "model": "gpt-4"
  }'
```

#### Check Service Health

```bash
# Main service
curl http://localhost:8000/docs

# scGPT service
curl http://localhost:8001/health
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# LLM Configuration
OPENAI_API_KEY=your-api-key-here
OLLAMA_BASE_URL=http://localhost:11434

# Service URLs
SCGPT_SERVICE_URL=http://localhost:8001
GENOMIX_AGENT_API=http://localhost:8000

# Data Paths
DATA_DIR=./data
RUNS_DIR=./runs

# Database
DATABASE_URL=sqlite:///cellmind.db

# RAG
CHROMA_PATH=./chroma_db
```

### Supported LLM Backends

| Backend | Configuration | Notes |
|---------|---------------|-------|
| **OpenAI** | `OPENAI_API_KEY` | GPT-4, GPT-3.5-turbo |
| **Ollama** | `OLLAMA_BASE_URL` | Local models (Deepseek, Mistral, etc.) |
| **HuggingFace** | `HF_TOKEN` | Open-source models |

---

## 📊 Analysis Capabilities

CellMind supports various single-cell analysis tasks:

- 🏷️ **Cell Type Annotation** - Identify and label cell types
- 🧬 **Gene Expression Analysis** - Analyze expression patterns
- 🛣️ **Pathway Enrichment** - Identify enriched biological pathways
- 🕸️ **Gene Regulatory Networks** - Infer GRN structure
- 📈 **Differential Expression** - Compare cell states
- 🔗 **Cell-Cell Interactions** - Predict intercellular communication
- 📍 **Spatial Analysis** - Integrate spatial transcriptomics data

---

## 📂 Project Structure

```
CellMind/
├── src/
│   ├── web/                    # FastAPI main service
│   │   ├── main.py            # Application entry point
│   │   ├── config.py          # Configuration
│   │   └── routes/            # API endpoints
│   │
│   ├── scgpt_service/         # scGPT embedding service
│   │   └── main.py            # Embedding service entry point
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── path_manager.py    # File path handling
│   │   └── langgraph_stream.py # Event streaming
│   │
│   └── scripts/                # Helper scripts
│       └── utils.py
│
├── cellmind/                   # Frontend TypeScript components
│   └── README.md
│
├── data/                       # Sample data directory
├── data_process/              # Data processing scripts
├── config/                    # Configuration files
│
├── main_CLI.py                # CLI entry point
├── test_llm_annotation.py     # Test suite
│
├── Dockerfile.main            # Main service container
├── Dockerfile.scgpt           # scGPT service container
├── docker-compose.yml         # Multi-service orchestration
│
├── requirements-main.txt      # Main service dependencies
├── requirements-scgpt.txt     # scGPT service dependencies
│
├── thesis_chapter4.md         # Documentation
├── thesis_chapter5.md         # Documentation
└── README.md                  # This file
```

---

## 🔍 Troubleshooting

### Problem: scGPT Service Connection Failed

```
Error: Failed to connect to scGPT service at http://localhost:8001
```

**Solution:**
```bash
# 1. Verify scGPT service is running
curl http://localhost:8001/health

# 2. Check environment variable
echo $SCGPT_SERVICE_URL

# 3. View scGPT service logs
docker logs genomix-scgpt  # if using Docker
# or check terminal output
```

### Problem: PyTorch/Torch Version Conflict

```
Error: Symbol not found: __ZN3c105ErrorC1E...
```

**Solution:**
```bash
# Use separate conda environments
conda activate cellmind-scgpt
pip install -r requirements-scgpt.txt

conda activate cellmind
pip install -r requirements-main.txt
```

### Problem: Out of Memory (GPU/RAM)

```
Error: CUDA out of memory
or RuntimeError: unable to allocate X.XX GiB
```

**Solution:**
```bash
# For Docker: Increase memory allocation
docker update --memory 16g genomix-main
docker update --memory 16g genomix-scgpt

# For local: Check available resources
nvidia-smi  # GPU memory
free -h     # System memory
```

### Problem: H5AD File Not Found

```
Error: File not found: /path/to/data.h5ad
```

**Solution:**
```bash
# Verify file path
ls -lh /path/to/data.h5ad

# Check file format
file /path/to/data.h5ad
# Should show: HDF5 file

# Run with absolute path
python main_CLI.py --file /absolute/path/to/data.h5ad
```

---

## 📚 API Documentation

### Interactive Docs

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v2/agent/run` | POST | Run analysis task |
| `/api/v2/health` | GET | Service health check |
| `/embeddings` | POST | Generate scGPT embeddings |
| `/docs` | GET | Interactive API documentation |

---

## 🧪 Testing

```bash
# Run test suite
pytest test_llm_annotation.py -v

# Test with specific model
python test_llm_annotation.py --model gpt-4

# Generate test report
pytest test_llm_annotation.py --html=report.html
```

---

## 🛠️ Development

### Install Development Dependencies

```bash
pip install -r requirements-main.txt
pip install pytest pytest-asyncio black flake8 mypy
```

### Code Quality

```bash
# Format code
black src/ main_CLI.py

# Lint
flake8 src/ main_CLI.py

# Type check
mypy src/ main_CLI.py
```

### Build Docker Images

```bash
# Build main service
docker build -f Dockerfile.main -t cellmind:latest .

# Build scGPT service
docker build -f Dockerfile.scgpt -t cellmind-scgpt:latest .
```

---

## 📖 Background & Theory

For detailed information about the algorithms and methodology, see:
- `thesis_chapter4.md` - Theoretical foundations
- `thesis_chapter5.md` - Implementation details

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See LICENSE file for details.

---

## 🙏 Acknowledgments

- **scGPT**: Cell foundation model from [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT)
- **LangGraph**: Multi-agent orchestration from [LangChain](https://docs.langchain.com/langgraph)
- **ScanPy**: Single-cell analysis toolkit
- **FastAPI**: Modern Python web framework

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/liu-shiqiang/CellMind/issues)
- **Documentation**: See markdown files in project root
- **API Docs**: Run service and visit `/docs` endpoint

---

**Last Updated**: June 2026
**Version**: 1.0.0
