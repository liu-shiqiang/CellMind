# 实验自动化代码设计说明

本文档概述 `src/experiments` 目录下的实验自动化实现，帮助研究人员理解如何基于 `agent_new.py` 的多智能体架构复现实验 1–6 并生成论文所需的指标与图表。

## 1. 模块结构

```
src/experiments/
├── __init__.py              # 暴露 ExperimentSuite
├── agent_runner.py          # 单次任务执行与运行结果封装
├── experiments.py           # 六大实验的 orchestration
├── metrics.py               # 指标聚合与统计
├── plots.py                 # 图 3A–3F 的绘制函数
├── runner.py                # Typer CLI 入口
└── task_library.py          # 任务库与意图基准数据
```

- **`task_library.py`**：定义超过 10 个覆盖细胞注释、差异分析、富集、记忆/状态查询等场景的 `TaskSpec`，并提供 `memory_task_sequences()` 与 `knowledge_task_ids()` 便于选择特定实验任务。`build_intent_benchmark()` 生成 200 条标注意图样本用于实验 6。
- **`agent_runner.py`**：封装对 `agent_new.run_objective` 的调用，尊重 `AgentRuntimeConfig` 中的规划器、记忆、RAG、故障注入等开关，返回 `AgentRunResult`（包含耗时、工具调用数、重规划次数、关键词命中率等）。
- **`experiments.py`**：`ExperimentSuite` 串联六个实验，针对不同实验构造对照配置（如线性基线、planner disabled、failure injection、memory off），执行任务并调用 `metrics` / `plots` 生成表格与图像。实验 6 直接调用 `run_objective(..., return_diagnostics=True)` 读取 `recognized_intents`，计算意图混淆矩阵。
- **`metrics.py`**：将 `AgentRunResult` 转为 `DataFrame`，提供按配置/任务聚合的成功率、耗时、工具调用、关键词召回等统计；`failure_recovery_table`、`memory_scores`、`knowledge_accuracy` 分别支撑实验 3/4/5 的专用指标；`build_confusion_matrix` 和 `classification_report` 输出实验 6 的评估。
- **`plots.py`**：使用 `matplotlib` 绘制实验 1–6 对应的图 3A–3F，包括分组条形图、小提琴图、生存曲线、雷达图、准确率-延迟散点和混淆矩阵。所有函数会在输出路径缺失时自动创建目录。
- **`runner.py`**：提供命令行入口 `python -m src.experiments.runner`，接收 `--dataset`、`--output-dir`、`--runs-per-task`、`--seed` 参数，内部创建 `ExperimentSuite` 并运行全部实验。

## 2. 与 `agent_new.py` 的集成方式

- 所有实验均通过 `AgentRuntimeConfig` 调节 `agent_new` 的规划器、重规划器、记忆、RAG、工具执行以及故障注入等特性，从而直接在真实多智能体实现上进行对比与消融。
- 线性基线、无规划、禁用记忆/RAG 等情形通过 `planner_mode`、`enable_replanner`、`enable_memory`、`enable_dataset_qa` 等开关控制，避免额外构造独立工作流。
- 实验 3 的鲁棒性评估利用 `FailureInjectionConfig` 在指定工具上以 30% 概率注入故障，并比较启用/禁用重规划器的恢复情况。
- 实验 4 将同一 `thread_id` 连续运行记忆写入与提问任务，衡量多轮对话记忆召回准确率；禁用记忆的配置使用不同 thread 以确保无持久化上下文。
- 实验 6 通过 `return_diagnostics=True` 直接读取 `recognized_intents` 并归一化标签，构建 200 条样本的混淆矩阵与分类报告。

## 3. 输出物

运行 `python -m src.experiments.runner --dataset /path/to/data.h5ad --output-dir results --runs-per-task 20` 后，将生成：

- `aggregated_results.csv`：所有任务的原始运行记录与指标。
- `summary_by_config.csv` / `summary_by_task.csv`：用于表 2 的综合统计。
- `fig_3a_grouped_bars.png` 至 `fig_3f_confusion.png`：对应论文图 3A–3F。
- `intent_predictions.csv` 与 `intent_classification_report.csv`：实验 6 的详细预测与分类报告。
- `metadata.json`：记录运行参数（随机种子、任务轮数、数据集路径等）。

若某些实验在当前配置下缺乏有效数据（例如未注入故障导致生存曲线为空），绘图函数会自动跳过并保持其他图表正常输出。

## 4. 扩展建议

- 如需新增任务或调整意图覆盖，可直接修改 `task_library.py` 中的 `TaskSpec` 列表或 `build_intent_benchmark` 模板。
- 如果要引入新的评估指标，可在 `metrics.py` 添加聚合函数，并在 `experiments.py` 中调用后输出相应表格或图像。
- 通过扩展 `AgentRuntimeConfig` 可继续探索更多组件的消融（例如自适应重规划策略），只需在 `agent_new.py` 中新增对应开关并在实验中传递配置即可。
