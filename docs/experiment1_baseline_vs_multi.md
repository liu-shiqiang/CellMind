# 实验 1：多智能体 vs 线性基线（baseline_vs_multi）

本文档整理 `ExperimentSuite` 中实验 1 的设计细节，便于研究人员核对运行配置、任务覆盖与预期输出。运行入口为 `src/experiments/exp1_baseline_vs_multi.py`，也可通过 `ExperimentSuite.run_experiment1()` 触发批量运行。

## 目标
- 量化多智能体配置（规划、重规划、记忆、RAG、数据集 QA 全部开启）相较于简化的线性基线在同一任务集合上的表现差异。
- 评估是否在任务成功率、工具调用效率和响应关键字覆盖度上取得提升。

## 任务集合
实验 1 针对所有需要数据集的任务执行对比（共 10 个）。任务由 `build_task_suite()` 定义，`dataset_task_ids()` 选择 `requires_dataset=True` 的项，包括：

1. `dataset_overview_basic`：报告细胞/基因计数与批次元数据【F:src/experiments/task_library.py†L34-L41】【F:src/experiments/task_library.py†L146-L148】
2. `cell_annotation_basic`：基础注释与代表性标记基因【F:src/experiments/task_library.py†L42-L48】
3. `marker_gene_panel`：免疫相关簇的标记基因盘点【F:src/experiments/task_library.py†L49-L55】
4. `differential_expression_pair`：两个主要聚类的差异表达与功能解读【F:src/experiments/task_library.py†L56-L62】
5. `ssgsea_enrichment_tcell`：T 细胞富集通路的 ssGSEA【F:src/experiments/task_library.py†L63-L69】
6. `cellphone_communication_core`：免疫簇间配体-受体通信【F:src/experiments/task_library.py†L70-L76】
7. `pseudotime_brief`：祖细胞/未分化细胞的伪时间轨迹【F:src/experiments/task_library.py†L77-L83】
8. `knowledge_validation_rag`：基于 RAG 的 T 细胞衰竭/活化知识总结【F:src/experiments/task_library.py†L84-L90】
9. `dataset_consistency_check`：批次或样本分层检测与可视化建议【F:src/experiments/task_library.py†L91-L97】
10. `plan_next_steps`：基于已完成的聚类/标注提出下一步分析建议【F:src/experiments/task_library.py†L98-L104】

## 配置对比
在每个任务上对比两套运行时配置，均使用相同的随机种子与线程前缀以便配对分析：

- **multi_agent（默认）**：开启规划器、重规划器、记忆、RAG 与数据集 QA。【F:src/experiments/experiments.py†L210-L235】
- **linear_baseline**：线性规划模式，禁用重规划、记忆、RAG 与数据集 QA。【F:src/experiments/experiments.py†L213-L244】

`runs_per_task` 控制重复次数；每个任务会按顺序运行 multi_agent 与 baseline 各一次（或多次），总运行次数为 `len(dataset_tasks) * runs_per_task * 2`。【F:src/experiments/experiments.py†L221-L245】

## 运行输出
所有产物写入 `--output_dir` 下的 `experiment1_baseline_vs_multi/` 子目录，核心文件包括：

- `runs.csv`：逐任务逐配置的成功标记、耗时、工具调用数和关键字命中率。
- `summary_by_config.csv`：按配置聚合的成功率与均值/方差指标，用于 Fig 3A 绘制。
- `task_level_summary.csv`：按任务拆分的对比，便于检查异常任务。
- `run_logs/experiment1_baseline_vs_multi.jsonl`：每次调用的原始记录，方便在实验中断时恢复。

若数据为空，绘图会跳过并在日志中提示；否则会生成成功率/运行时间/工具调用的分组柱状图（对应论文 Fig 3A）。

## 成功判定与诊断
- 任务成功取决于 `AgentRunResult.success` 标记，来源于 `run_objective` 的完成信号和关键字匹配。【F:src/experiments/experiments.py†L152-L179】
- 关键异常会写入 `failure_report.json`，单次失败仍会被记录在 JSONL 日志中，避免长时间运行中断造成结果缺失。

研究人员可据此快速核对实验 1 的设置是否符合论文方案，并对输出文件与图表进行验证。
