# 系统功能测试指南

本指南用于验证 Agent Mode 的端到端流程：上传数据 → 自动分析 → 结果解读 → 产物下载。

## 1. 环境准备
1. Python 3.10+，依赖至少包含 `scanpy`, `anndata`, `langchain`, `chromadb`。
2. 配置检查：`src/web/config.py` 中的 LLM/RAG/路径配置可用。
3. 准备 `.h5ad` 文件用于测试。

## 2. CLI 端到端测试
```bash
python main_CLI.py --file /path/to/sample.h5ad --stream
```
期望：
- 终端输出 node/plan/token 流式事件。
- `runs/{job_id}/artifacts/` 生成 `data/`, `tables/`, `reports/` 等产物目录。

## 3. 产物校验
- 报告：`runs/{job_id}/artifacts/reports/analysis_report_*.md`
- UMAP 坐标：`runs/{job_id}/artifacts/tables/umap_coords_*.csv`
- Marker 表：`runs/{job_id}/artifacts/tables/marker_genes_*.csv`
- 差异表达：`runs/{job_id}/artifacts/tables/diff_expression_*.csv`

## 4. API 校验
```bash
# UMAP
curl http://localhost:8000/api/visualization/{job_id}/umap

# Marker 基因
curl http://localhost:8000/api/visualization/{job_id}/markers

# 差异表达
curl http://localhost:8000/api/visualization/{job_id}/diff_expression
```

## 5. 前端校验
- Agent Mode 执行后，ReportCard 可下载真实产物文件。
- UMAP/markers/DE 可视化数据来自 `runs/{job_id}/artifacts`。
