# 实验自动化代码设计说明

本文档概述 `src/experiments` 目录下新增模块的职责与关键设计决策，以便研究人员快速理解并扩展实验流水线。

## 1. 模块结构

```
src/experiments/
├── __init__.py
├── analysis_pipeline.py
├── metrics.py
├── runner.py
├── tasks.py
├── visualization.py
└── workflows.py
```

- **`analysis_pipeline.py`**：封装单细胞分析核心工具的调用顺序，提供统一的 `SingleCellAnalysisPipeline` 类，用于在实验中按需执行数据加载、嵌入提取、聚类、注释、富集分析以及知识检索。该模块还实现了 `ToolFailureInjector`，用于在鲁棒性实验中注入可配置的工具故障。
- **`tasks.py`**：定义任务与数据集的元数据结构，并根据给定 `.h5ad` 数据集自动生成包含简单、复合与模糊意图的任务组合，同时提供默认的知识问答问题库。
- **`workflows.py`**：实现单代理基线与多代理规划执行器。`MultiAgentWorkflow` 模拟规划、重规划、记忆与知识检索的协同流程，而 `SingleAgentWorkflow` 则代表无规划、直接调用工具的基线。
- **`metrics.py`**：聚合 `WorkflowRun` 的指标，输出任务成功率、运行时间、工具调用次数、计划编辑距离等统计量，并可按数据集或任务维度生成表格。
- **`visualization.py`**：针对实验设计生成标准化图表，包括分组条形图、小提琴图、生存曲线、雷达图、散点图及混淆矩阵，统一使用 `matplotlib` 与 `seaborn` 风格。
- **`runner.py`**：提供 `ExperimentSuite` 主入口，串联六个实验的执行逻辑、结果收集与可视化生成，并通过 `typer` 暴露 CLI 接口 `python -m src.experiments.runner`。

## 2. 关键设计选择

### 2.1 工具调用与缓存
`SingleCellAnalysisPipeline` 对工具返回的工件进行追踪，并在启用缓存时复用中间结果，以便在多次运行时避免重复计算。记忆相关实验复用同一个 `ConversationMemoryStore`，确保跨运行的上下文一致性。

### 2.2 多代理特性模拟
`MultiAgentWorkflow` 通过显式的计划生成与步骤执行，模拟真实系统中的规划、执行、重规划流程。每一步调用 `SingleCellAnalysisPipeline`，并根据工具调用结果推导计划编辑距离、失败恢复率等指标。

### 2.3 实验维度覆盖
`ExperimentSuite` 将实验拆分为独立函数，分别控制规划消融、重规划器故障注入、记忆开关、知识检索开关等变量。每个实验返回结构化的 `WorkflowRun` 列表，便于统一聚合与可视化。

### 2.4 图表与表格输出
`_generate_outputs` 方法集中生成论文所需的图表（图 3A–3F）以及综合指标表（表 2），所有文件默认写入 `results/figures/` 与 `results/table_2_metrics.csv`。

## 3. 使用指南

1. 在本地准备 `.h5ad` 数据集后执行：
   ```bash
   python -m src.experiments.runner --dataset /path/to/sample.h5ad --output-dir results --runs-per-task 20
   ```
2. 运行结束后，`results/` 目录包含：
   - `figures/figure_3A_*` 等图像文件；
   - `table_2_metrics.csv` 综合指标；
   - `conversation_memory.json` 长期记忆快照。
3. 若需调整任务或指标，可修改 `tasks.py` 中的任务生成逻辑，或在 `metrics.py`/`visualization.py` 中添加新的统计与图表函数。

## 4. 后续扩展建议

- 接入真实的意图识别模型，替换 `RuleBasedIntentClassifier` 以获取更精确的混淆矩阵。
- 在 `SingleCellAnalysisPipeline` 中增加更多工具节点（如伪时间分析、细胞互作），以覆盖实验设计中提及的全部任务类型。
- 将图表生成函数与报告模板对接，实现一键导出 PDF 或 Markdown 形式的实验报告。

