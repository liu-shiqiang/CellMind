# 系统功能测试指南

本文档说明如何在本项目中验证端到端的代理分析流程，包括结果图的产出与最终的解读报告。请在执行前确认已经具备单细胞分析的输入数据以及必要的知识库资源。

## 1. 环境与资源准备
1. **Python 环境**：推荐使用 Python 3.10 及以上版本，并安装 Scanpy、Matplotlib、LangChain、Chromadb、Neo4j 驱动等依赖。例如：
   ```bash
   pip install scanpy matplotlib langchain chromadb neo4j
   ```
2. **配置路径**：根据本地环境修改 `config/setting.py` 中的路径与模型配置，使其指向可访问的模型目录、Chroma 持久化目录及文献知识库。该文件同时定义了默认的 LLM 访问地址。  
3. **数据准备**：保证待分析的 `.h5ad` 文件和上游聚类/marker/富集结果位于同一工作目录。代理流程会在该目录下生成图像和解读文件。若需要使用细胞级 RAG 或 PubMed RAG，需提前构建/下载对应的向量库与 API 访问凭据。

## 2. 端到端代理测试
1. 通过 CLI 启动代理流程（支持实时事件流）：
   ```bash
   python main_CLI.py --file /path/to/sample.h5ad --stream
   ```
   - 根据提示输入任务描述、核验数据文件。  
   - 运行结束后，终端会输出最终消息，同时在工作目录写入分析产物。
2. 若需批量文件或关闭事件流，可使用 `--files` 或去掉 `--stream` 参数。CLI 逻辑位于 `main_CLI.py`，可用于调试输入校验与整体流程。  

## 3. 核验结果图像
代理在执行 `cluster_and_rank_markers` 与细胞注释工具时会生成关键图像文件：
- `work_dir/<sample>_umap_scgpt_clustered.png`：聚类 UMAP 可视化。由 `src/tools/clustering_marker.py` 在完成 Leiden 聚类后调用 `scanpy.pl.umap` 自动保存。【F:src/tools/clustering_marker.py†L33-L80】
- `work_dir/<sample>_llm_celltypes.png`：LLM 注释后的细胞类型 UMAP 图，由 `src/tools/cell_anno_zilin.py` 保存。【F:src/tools/cell_anno_zilin.py†L412-L459】
- `work_dir/<sample>_optimized_dotplot.pdf`：精选 marker 基因的 dotplot，可用于核实 marker 表达模式。【F:src/tools/cell_anno_zilin.py†L529-L558】
- 若启用了 CellPhoneDB 等通讯分析，还会在对应子目录下生成热图或 chord 图（参见 `src/tools/cellphoneDB.py` 中的 `plt.savefig` 调用）。【F:src/tools/cellphoneDB.py†L380-L436】

完成一次流程后，请人工检查上述图片是否生成、内容是否与预期一致，从而确认聚类与注释结果的可视化环节正常。

## 4. 验证双层解读产物
聚类级与细胞类型级解读均在 `interpretation/` 子目录内生成：
- `dataset_context.json`：聚类列表、Top 基因、富集与全局信号汇总，由 `build_dataset_context` 构建。【F:src/scripts/dataset_interpretation.py†L119-L190】
- `dataset_interpretation_report.md`：数据集整体报告，由 `generate_dataset_report` 写入。【F:src/scripts/dataset_interpretation.py†L192-L236】
- `celltype_context.json` 与 `celltype_interpretation_report.md`：细胞类型聚合上下文与叙述报告，由 `generate_celltype_report` 在同一目录输出。触发逻辑可在 `interpret_cluster_results` 中查看。【F:src/tools/interpret_cluster_results.py†L96-L177】
- 若只需单独刷新细胞类型报告，可直接调用 `interpret_celltype_results` 工具，它会复用已有聚类结果并重新生成细胞类型层级文档。【F:src/tools/interpret_celltype_results.py†L16-L74】

请检查 Markdown 报告是否包含概述、功能通路、通讯网络等段落，并对照 JSON 上下文确认数值与聚类统计吻合。

## 5. 独立调试关键工具
在端到端流程外，也可手动调用核心工具以复现特定阶段：
```bash
python - <<'PY'
from src.tools.interpret_cluster_results import interpret_cluster_results
print(interpret_cluster_results(
    work_dir="/path/to/work_dir",
    enable_cell_rag=True,
    cell_rag_results=5
))
PY
```
- 输出 JSON 会列出聚类解读文件、双层报告路径以及使用的知识库集合。

若需要单独刷新细胞类型层级报告：
```bash
python - <<'PY'
from src.tools.interpret_celltype_results import interpret_celltype_results
print(interpret_celltype_results(
    work_dir="/path/to/work_dir"
))
PY
```

## 6. 解读问答功能验证
在解读文件准备就绪后，可以测试问答工具：
```bash
python - <<'PY'
from src.tools.dataset_qa import dataset_bio_qa
print(dataset_bio_qa(
    work_dir="/path/to/work_dir",
    question="激活态T细胞的关键信号通路有哪些？",
    top_k_local=3,
    top_k_pubmed=3
))
PY
```
该工具会读取双层报告、结合本地/在线文献检索生成答案，并把交互日志写入 `interpretation/qa_history.jsonl`，便于回放历史问答。【F:src/tools/dataset_qa.py†L1-L248】

通过以上步骤，可系统性地验证：
- 原始数据加载与聚类分析是否成功；
- 各类结果图像是否完整生成；
- 聚类级与细胞类型级解读报告内容是否合理；
- 基于报告与文献检索的问答功能是否正常工作。
