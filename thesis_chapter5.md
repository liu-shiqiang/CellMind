# 第5章 单细胞数据智能分析系统的设计与实现

## 5.1 引言

随着单细胞测序技术的快速发展，单细胞数据分析已成为生命科学研究领域的重要工具。然而，现有的单细胞数据分析工具普遍存在以下问题：（1）需要专业编程知识，门槛较高；（2）分析流程缺乏自动化，需要大量人工干预；（3）缺乏智能化决策支持，分析结果依赖于用户经验；（4）难以整合多种分析方法和知识资源。

为解决上述问题，本研究在前期理论与方法研究的基础上，设计并实现了一个面向单细胞数据的智能分析系统（Genomix-Agent）。该系统融合了大语言模型技术、智能体编排框架和生物信息学分析工具，旨在为研究人员提供一个自动化、智能化、交互式的单细胞数据分析平台。

本章将详细阐述该系统的需求分析、总体设计、具体实现与系统测试过程。通过本章的介绍，读者可以全面了解本系统如何将第4章提出的理论方法与模型转化为实际可用的软件系统，以及系统各模块的实现细节与技术特点。

## 5.2 系统需求分析

### 5.2.1 业务流程分析

本系统的核心用户群体是生物信息学研究人员和实验科学家。通过对用户工作流程的分析，本系统设计的典型业务流程如图5-1所示。

```
[用户] → [上传数据] → [描述分析目标] → [智能规划] → [自动执行]
                                              ↓
[查看结果] ← [生成报告] ← [结果整合] ← [进度反馈]
```

**业务流程说明**：

1. **数据上传阶段**：用户通过Web界面上传单细胞测序数据文件（如.h5ad格式），系统自动进行格式验证和元数据提取。

2. **目标描述阶段**：用户使用自然语言描述分析目标（如"对这批PBMC数据进行聚类分析并识别细胞类型"），系统无需用户选择具体的工具或参数。

3. **智能规划阶段**：系统自动理解用户意图，生成结构化的分析计划，包括数据预处理、聚类分析、差异表达等步骤的序列。

4. **自动执行阶段**：系统按照计划依次调用分析工具，支持工具间的数据流转和错误处理。

5. **进度反馈阶段**：用户可实时查看分析进度和中间结果，确保分析过程透明可控。

6. **结果整合阶段**：系统将所有分析结果整合为统一的报告，包括可视化图表、统计摘要和解释说明。

### 5.2.2 用例需求分析

通过用例建模，识别出系统的主要参与者和核心用例。图5-2展示了系统的用例图。

**主要参与者（Actors）**：
- **访客用户**：未登录用户，可浏览系统功能但不能执行分析
- **注册用户**：已登录用户，可创建分析项目、上传数据、执行分析
- **管理员**：具有系统管理权限的用户

**核心用例列表**：

| 用例编号 | 用例名称 | 参与者 | 描述 |
|----------|----------|--------|------|
| UC-01 | 用户注册/登录 | 注册用户 | 身份验证与权限管理 |
| UC-02 | 上传单细胞数据 | 注册用户 | 支持.h5ad等格式的数据上传 |
| UC-03 | 创建分析会话 | 注册用户 | 创建独立的分析工作空间 |
| UC-04 | 自然语言交互 | 注册用户 | 通过对话描述分析目标 |
| UC-05 | 自动分析执行 | 系统 | 智能规划和执行分析流程 |
| UC-06 | 查看分析进度 | 注册用户 | 实时监控分析任务状态 |
| UC-07 | 查看分析结果 | 注册用户 | 浏览可视化结果和统计数据 |
| UC-08 | 下载分析报告 | 注册用户 | 导出PDF/HTML格式的分析报告 |
| UC-09 | 管理历史会话 | 注册用户 | 查看和恢复历史分析项目 |
| UC-10 | 系统管理 | 管理员 | 用户管理、资源监控等 |

**关键用例详细描述**：

**UC-04：自然语言交互**
- **用例名称**：自然语言交互
- **参与者**：注册用户
- **前置条件**：用户已登录，已有数据上传
- **后置条件**：系统理解用户意图并生成分析计划
- **基本事件流**：
  1. 用户在聊天界面输入分析目标（如"对数据进行聚类分析"）
  2. 系统调用意图识别模块分析用户输入
  3. 系统识别出核心意图（聚类分析）和依赖意图（数据加载、质控）
  4. 系统生成结构化的执行计划
  5. 系统向用户展示计划并请求确认
- **扩展事件流**：
  - 2a. 用户输入不明确：系统请求澄清
  - 4a. 缺少必要数据：系统提示用户先上传数据

### 5.2.3 功能需求分析

根据业务流程和用例分析，系统功能需求可划分为以下核心模块：

**（1）用户管理模块**
- 用户注册与登录
- 基于角色的权限控制
- 用户会话管理

**（2）数据管理模块**
- 单细胞数据文件上传（支持.h5ad、.csv、.mtx等格式）
- 数据格式验证与元数据提取
- 数据存储与版本管理
- 数据集关联与项目组织

**（3）智能分析核心模块（系统核心）**
- **意图识别**：将自然语言转化为结构化的分析意图
- **自动规划**：根据意图生成多步骤分析计划
- **工具调度**：执行各类单细胞分析工具
- **智能重规划**：分析失败时自动调整策略
- **长期记忆**：跨会话保持分析上下文

**（4）分析工具模块**
- 数据质量控制（QC）
- 数据标准化与高变基因识别
- 降维分析（PCA、UMAP、t-SNE）
- 细胞聚类（Leiden、K-Means）
- 标记基因识别
- 细胞类型注释
- 差异表达分析
- 伪时序分析
- 细胞通讯分析
- 通路富集分析
- scGPT嵌入提取

**（5）可视化与交互模块**
- UMAP/t-SNE散点图
- 细胞类型分布图
- 基因表达小提琴图
- 热图可视化
- 交互式图表筛选

**（6）结果导出与报告模块**
- 分析结果JSON导出
- Markdown格式报告生成
- 可视化图表导出

**（7）会话与记忆管理模块**
- 会话状态持久化
- 长期记忆存储与检索
- 项目状态恢复

### 5.2.4 可行性分析

**（1）技术可行性**

本系统采用的技术栈均为成熟且广泛应用的开源技术：

| 技术领域 | 选型 | 成熟度 |
|----------|------|--------|
| 后端框架 | FastAPI | 高，生产级Web框架 |
| 前端框架 | Streamlit | 高，快速数据应用开发 |
| 智能体编排 | LangGraph | 高，LangChain生态核心组件 |
| 单细胞分析 | Scanpy | 高，领域标准工具 |
| 数据库 | SQLite | 高，轻量级关系数据库 |
| 容器化 | Docker | 高，行业标准 |

scGPT微服务的独立部署解决了PyTorch版本冲突问题，体现了架构设计的前瞻性。

**（2）操作可行性**

系统采用自然语言交互界面，用户无需编写代码即可完成复杂的单细胞分析流程。交互式进度反馈和结果展示降低了使用门槛，使非计算机专业的生物学家也能有效使用。

**（3）经济可行性**

系统基于完全开源的技术栈构建，无需支付商业软件许可费用。Docker容器化部署降低了部署和维护成本，支持在常规服务器上运行。

## 5.3 系统设计

### 5.3.1 系统整体架构设计

本系统（CellMind）采用**微服务架构**与**分层架构**相结合的设计模式，实现模块间的松耦合和高内聚。系统由React前端、FastAPI后端、scGPT微服务和多种数据存储组件构成。图5-3展示了系统的整体架构。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              前端表现层 (Frontend)                            │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                          React + TypeScript                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │  │
│  │  │ ChatBox  │  │Sidebar   │  │AgentStatus│ │ UmapVisualization    │   │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘   │  │
│  │  ┌──────────────────────────────────────────────────────────────────┐  │  │
│  │  │                 Zustand 状态管理 + Axios HTTP客户端               │  │  │
│  │  └──────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────┬─────────────────────────────────┘
                                              │ HTTP/REST + SSE
┌─────────────────────────────────────────────▼─────────────────────────────────┐
│                              API网关层 (API Gateway)                           │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                           FastAPI Router                               │  │
│  │                    (Port 8000) /api/*                                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────┬──────────────┬──────────────┬──────────────┬──────────────────────┘
           │              │              │              │
┌──────────▼──────┐ ┌────▼────────┐ ┌───▼──────────┐ ┌─▼──────────────┐
│ /api/auth       │ │ /api/chat   │ │ /api/agent   │ │ /api/upload     │
│ 登录/注册/密码   │ │ 聊天/会话   │ │ 智能体执行   │ │ 文件上传/验证   │
└─────────────────┘ └─────────────┘ └───┬──────────┘ └─────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                           业务逻辑层 (Business Logic)                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          LangGraph 状态机                             │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │   意图   │→│   规划   │→│   执行   │→│     重规划       │    │  │
│  │  │  识别    │  │   器    │  │   器    │  │                  │    │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │               AgentState 状态管理                               │  │
│  │  │  - objective, messages, intents, plan                          │  │
│  │  │  - execution_status, tool_history, project_state               │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        对话记忆服务 (RAG)                             │  │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │  │
│  │  │上下文保存       │───▶│ ChromaDB向量库  │◀───│上下文检索       │  │  │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
┌──────────────────────────────────────────────▼───────────────────────────────┐
│                          分析工具层 (Analysis Tools)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        单细胞核心工具 (Scanpy)                          │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │  │
│  │  │数据  │ │质控  │ │标准化│ │PCA   │ │聚类  │ │标记  │ │注释  │     │  │
│  │  │加载  │ │      │ │HVG   │ │      │ │UMAP  │ │基因  │ │      │     │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │  │
│  │  │差异表达  │ │伪时序    │ │细胞通讯  │ │通路富集  │                  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
┌──────────────────────────────────────────────▼───────────────────────────────┐
│                          数据访问层 (Data Access)                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│  │  SQLAlchemy     │    │   文件系统       │    │   ChromaDB      │          │
│  │     ORM         │    │   FileSystem     │    │  向量数据库      │          │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘          │
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
┌──────────────────────────────────────────────▼───────────────────────────────┐
│                          数据存储层 (Data Storage)                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐  │
│  │   SQLite     │    │uploaded_data/│    │   runs/      │    │  ChromaDB  │  │
│  │   数据库      │    │  用户上传    │    │  分析产物    │    │  向量存储   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘

                                ═════════════════════
┌────────────────────────────────────────────────────────────────────────────────┐
│                          scGPT 微服务 (独立容器)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                     scGPT Embedding Service                             │  │
│  │                            (Port 8001)                                  │  │
│  │  ┌─────────────────┐    ┌─────────────────┐                            │  │
│  │  │  PyTorch 2.0+   │    │   scGPT Model   │                            │  │
│  │  └─────────────────┘    └─────────────────┘                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

**技术流程说明**：

1. **前端交互**：React前端通过Zustand管理应用状态，使用Axios与后端API通信。SSE（Server-Sent Events）用于接收Agent执行的实时进度推送。

2. **API网关**：FastAPI提供统一的RESTful API入口，包括认证、聊天、Agent执行、文件上传等模块。

3. **智能体工作流**：基于LangGraph构建的状态机，实现意图识别→任务规划→工具执行→结果响应的完整流程。支持执行失败后的智能重规划。

4. **工具调用**：单细胞分析工具基于Scanpy实现，包括数据加载、质控、标准化、降维、聚类、注释等功能。

5. **数据持久化**：SQLite存储用户、会话、消息等元数据；ChromaDB存储对话上下文的向量表示；文件系统存储原始数据和分析产物。

6. **微服务隔离**：scGPT作为独立容器部署，通过HTTP与主服务通信，解决PyTorch版本冲突问题。

**技术栈一览**：

| 层次 | 技术选型 | 版本 | 用途 |
|------|----------|------|------|
| 前端 | React | 19.2.3 | UI框架 |
| 前端 | TypeScript | 5.x | 类型安全 |
| 前端 | Zustand | 5.0.10 | 状态管理 |
| 前端 | Vite | 6.2.0 | 构建工具 |
| 前端 | Tailwind CSS | 4.1.18 | 样式框架 |
| 前端 | D3.js | 7.9.0 | 数据可视化 |
| 后端 | FastAPI | 0.104.0 | Web框架 |
| 后端 | LangGraph | 0.2.0 | 智能体编排 |
| 后端 | SQLAlchemy | 2.0.0 | ORM框架 |
| 分析 | Scanpy | 1.9.0 | 单细胞分析 |
| 存储 | SQLite | 3.x | 关系数据库 |
| 存储 | ChromaDB | 0.4.0 | 向量数据库 |
| 容器 | Docker | 24.0+ | 容器化部署 |

### 5.3.2 主要功能模块设计

#### （1）智能体工作流模块

智能体工作流是系统的核心，采用**有限状态机（FSM）**模式设计。图5-4展示了智能体的状态转换图，该流程基于LangGraph框架实现。

```
                    ┌─────────────────────────────────────────────────────┐
                    │                    Agent 工作流状态机                 │
                    └─────────────────────────────────────────────────────┘

                               ┌─────────────┐
                               │    START    │
                               └──────┬──────┘
                                      │
                                      ▼
                        ┌─────────────────────────────┐
                        │    intent_recognition       │
                        │       意图识别节点           │
                        │  - 解析用户输入             │
                        │  - 识别分析意图             │
                        │  - 判断是否需要规划         │
                        └──────────┬──────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼ 需要分析                     ▼ 简单问答
          ┌───────────────────┐          ┌───────────────────┐
          │      planner      │          │     response      │
          │      规划器       │          │     响应节点       │
          │  - 生成执行计划   │          │  - 格式化回复     │
          │  - 确定工具序列   │          │  - 保存上下文     │
          └─────────┬─────────┘          └─────────┬─────────┘
                    │                              │
                    ▼                              │
          ┌───────────────────┐                  │
          │     executor      │◄─────────────────┘
          │      执行器        │
          │  - 调用分析工具   │
          │  - 更新项目状态   │
          └─────────┬─────────┘
                    │
      ┌─────────────┼─────────────┐
      │             │             │
      ▼ 继续执行    ▼ 执行失败    ▼ 执行完成
┌───────────┐  ┌───────────┐  ┌───────────┐
│ executor  │  │replanner  │  │ response  │
│ (下一工具)│  │  重规划器  │  │  响应     │
└───────────┘  └─────┬─────┘  └─────┬─────┘
                    │              │
        ┌───────────┴───────────┐  │
        │                       │  ▼
        ▼ 可恢复                ▼  END
┌─────────────┐          ┌─────────────┐
│  executor   │          │  response   │
│  (重试)     │          │  (放弃)     │
└─────────────┘          └──────┬──────┘
                                │
                                ▼
                            ┌───────┐
                            │  END  │
                            └───────┘
```

**Agent状态定义（AgentState）**：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| objective | str | 用户分析目标 |
| messages | List[BaseMessage] | 对话消息历史 |
| input_files | List[str] | 输入文件路径列表 |
| intents | List[Intent] | 识别到的意图列表 |
| plan | List[str] | 执行计划步骤 |
| next_step | Optional[str] | 下一步操作 |
| execution_status | str | 执行状态：in_progress/completed/failed |
| replan_attempts | int | 重规划次数 |
| work_dir | Optional[str] | 工作目录 |
| tool_history | List[Dict] | 工具调用历史 |
| project_state | Dict | 项目状态（数据集、已完成步骤等） |
| thread_id | str | 线程ID |
| session_id | str | 会话ID |
| run_id | str | 运行ID |

**意图类型定义**：

| 意图标签 | 说明 | 依赖意图 |
|----------|------|----------|
| load_data | 加载数据文件 | 无 |
| quality_control | 质量控制分析 | load_data |
| normalize | 数据标准化与高变基因 | quality_control |
| dimensionality_reduction | 降维分析（PCA/UMAP） | normalize |
| clustering | 细胞聚类 | dimensionality_reduction |
| marker_genes | 标记基因识别 | clustering |
| annotation | 细胞类型注释 | marker_genes |
| differential_expression | 差异表达分析 | annotation |
| trajectory_analysis | 伪时序分析 | clustering |
| cell_communication | 细胞通讯分析 | annotation |
| pathway_enrichment | 通路富集分析 | annotation |

#### （2）文件上传与验证模块

文件上传模块负责接收用户上传的单细胞数据文件，进行格式验证和元数据提取。图5-5展示了文件上传的完整流程。

```
                        ┌─────────────────────────────────────┐
                        │           文件上传流程               │
                        └─────────────────────────────────────┘

     用户                 React组件                   FastAPI后端              文件系统
      │                      │                           │                      │
      │  选择文件            │                           │                      │
      ├─────────────────────►│                           │                      │
      │                      │                           │                      │
      │                      │  POST /api/upload          │                      │
      │                      ├──────────────────────────►│                      │
      │                      │  - multipart/form-data     │                      │
      │                      │                           │                      │
      │                      │                           │  1. 文件类型验证      │
      │                      │                           │  2. 文件大小检查      │
      │                      │                           │  3. 保存到磁盘        │
      │                      │                           ├─────────────────────►│
      │                      │                           │  uploaded_data/...    │
      │                      │                           │                      │
      │                      │                           │  4. 创建File记录      │
      │                      │                           │  5. H5AD格式验证      │
      │                      │  返回文件元数据            │  6. 提取数据摘要      │
      │                      │◄──────────────────────────┤                      │
      │  文件信息卡片        │                           │                      │
      │◄─────────────────────┤                           │                      │
      │                      │                           │                      │
      │  [文件名]            │                           │                      │
      │  [细胞数: XXX]       │                           │                      │
      │  [基因数: XXX]       │                           │                      │
      │                      │                           │                      │
```

**文件验证逻辑**：

1. **格式验证**：仅支持.h5ad格式文件
2. **大小限制**：默认最大2GB
3. **数据完整性**：使用Scanpy读取文件，确保数据格式正确
4. **元数据提取**：提取细胞数、基因数、是否有聚类结果等信息

#### （3）会话与记忆管理模块

会话管理模块负责维护用户的分析会话状态，支持跨会话的项目状态恢复。图5-6展示了会话状态管理的流程。

```
                    ┌──────────────────────────────────────────┐
                    │           会话与记忆管理流程              │
                    └──────────────────────────────────────────┘

     用户                    前端状态                   后端服务
      │                        │                          │
      │  登录                  │                          │
      ├───────────────────────►│                          │
      │                        │  POST /api/auth/login     │
      │                        ├─────────────────────────►│
      │                        │  返回JWT Token           │
      │                        │◄─────────────────────────┤
      │  Token + 用户信息      │                          │
      │◄───────────────────────┤                          │
      │                        │                          │
      │  加载会话列表          │                          │
      ├───────────────────────►│                          │
      │                        │  GET /api/sessions        │
      │                        ├─────────────────────────►│
      │                        │  查询SQLite              │
      │                        │◄─────────────────────────┤
      │  会话列表              │                          │
      │◄───────────────────────┤                          │
      │                        │                          │
      │  选择/创建会话         │                          │
      ├───────────────────────►│                          │
      │                        │  POST /api/sessions       │
      │                        ├─────────────────────────►│
      │                        │  创建Session记录          │
      │                        │  加载project_state        │
      │                        │◄─────────────────────────┤
      │  会话详情              │                          │
      │◄───────────────────────┤                          │
      │                        │                          │
      │  发送分析请求          │                          │
      ├───────────────────────►│                          │
      │                        │  POST /api/agent/run      │
      │                        ├─────────────────────────►│
      │                        │  加载记忆上下文           │
      │                        │  恢复project_state       │
      │                        │  执行Agent分析           │
      │                        │                          │
      │  ═══════════════       │                          │
      │  SSE进度推送           │═══════════════════════════│
      │◄═══════════════════════╛═══════════════════════════╛
      │                        │                          │
      │  分析完成              │                          │
      │  - 更新project_state   │                          │
      │  - 保存对话记忆        │                          │
      │                        │                          │
```

**记忆存储结构**：

```json
{
  "thread_id": "conv_12345",
  "summary": "用户对PBMC数据进行了聚类分析和细胞注释",
  "records": [
    {
      "timestamp": "2024-01-15T10:00:00",
      "objective": "对数据进行聚类分析",
      "result_summary": "识别到12个细胞cluster",
      "key_findings": ["主要细胞类型：T细胞、B细胞、单核细胞"]
    }
  ],
  "project_state": {
    "active_dataset": "pbmc_3k",
    "datasets": {
      "pbmc_3k": {
        "work_dir": "runs/job_123/artifacts",
        "loaded_path": "data/pbmc_3k.h5ad",
        "qc_path": "data/qc_filtered.h5ad",
        "normalized_path": "data/normalized.h5ad",
        "pca_path": "data/pca.h5ad",
        "clustered_path": "data/clustered.h5ad",
        "markers_path": "tables/markers.csv",
        "annotated_path": "data/annotated.h5ad",
        "completed_steps": ["load", "qc", "normalize", "pca", "cluster", "annotate"]
      }
    }
  }
}
```

#### （4）数据可视化模块

数据可视化模块提供多种图表展示分析结果。图5-7展示了可视化组件的结构。

```
                    ┌──────────────────────────────────────────┐
                    │            数据可视化模块                 │
                    └──────────────────────────────────────────┘

                        ┌──────────────────────────┐
                        │    UmapVisualization      │
                        │     UMAP散点图组件        │
                        │  - D3.js渲染             │
                        │  - 交互式筛选            │
                        │  - cluster着色           │
                        └──────────────────────────┘

                        ┌──────────────────────────┐
                        │      VolcanoPlot         │
                        │      火山图组件          │
                        │  - DEG结果展示           │
                        │  - 基因标注              │
                        └──────────────────────────┘

                        ┌──────────────────────────┐
                        │    ClusterHeatmap        │
                        │     聚类热图组件         │
                        │  - 标记基因热图          │
                        │  - 颜色映射              │
                        └──────────────────────────┘

                        ┌──────────────────────────┐
                        │        DotPlot           │
                        │       点图组件           │
                        │  - 基因表达点图          │
                        │  - 分组展示              │
                        └──────────────────────────┘

                        ┌──────────────────────────┐
                        │     GoEnrichment         │
                        │     富集分析组件         │
                        │  - GO条目展示            │
                        │  - 柱状图/气泡图         │
                        └──────────────────────────┘
```

### 5.3.3 数据库设计

系统采用SQLite作为主数据库，通过SQLAlchemy ORM进行数据访问。数据库包含五个核心表，分别存储用户、会话、消息、文件和Agent运行记录。

#### （1）数据库ER图

```
┌─────────────────────┐
│      User           │
│─────────────────────│
│ id (PK)             │◄────┐
│ username            │ 1:N │
│ email               │     │
│ hashed_password     │     │
│ full_name           │     │
│ is_active           │     │  ┌─────────────────────┐
│ is_verified         │     └──│      Session         │
│ created_at          │        │─────────────────────│
│ updated_at          │        │ id (PK)             │
│ last_login_at       │        │ user_id (FK)        │
└─────────────────────┘        │ title               │
                               │ agent_mode          │
                               │ project_state (JSON)│
                               │ created_at          │
                               │ updated_at          │
                               └────────┬────────────┘
                                        │
                   ┌────────────────────┼────────────────────┐
                   │                    │                    │
          ┌────────▼────────┐  ┌───────▼──────┐  ┌──────────▼─────────┐
          │     Message     │  │     File     │  │     AgentRun       │
          │─────────────────│  │──────────────│  │─────────────────────│
          │ id (PK)         │  │ id (PK)      │  │ id (PK)             │
          │ session_id (FK) │  │ session_id   │  │ session_id (FK)     │
          │ role            │  │ filename     │  │ objective           │
          │ content         │  │ filepath     │  │ status              │
          │ user_metadata   │  │ file_size    │  │ steps (JSON)        │
          │ timestamp       │  │ upload_time  │  │ result              │
          └─────────────────┘  └──────────────┘  │ error_message       │
                                                 │ started_at          │
                                                 │ completed_at        │
                                                 └─────────────────────┘
```

#### （2）数据表设计

**表5-3 用户表（users）**

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| id | VARCHAR | PRIMARY KEY | 用户唯一标识 |
| username | VARCHAR(50) | UNIQUE, NOT NULL, INDEX | 用户名，唯一 |
| email | VARCHAR(100) | UNIQUE, NOT NULL, INDEX | 邮箱，唯一 |
| hashed_password | VARCHAR(255) | NOT NULL | 加密后的密码（bcrypt） |
| full_name | VARCHAR(100) | NULLABLE | 用户全名 |
| is_active | BOOLEAN | DEFAULT TRUE | 账户是否激活 |
| is_verified | BOOLEAN | DEFAULT FALSE | 邮箱是否验证 |
| created_at | DATETIME | NOT NULL | 创建时间 |
| updated_at | DATETIME | NOT NULL | 更新时间 |
| last_login_at | DATETIME | NULLABLE | 最后登录时间 |

用户表（users）是系统用户管理的核心数据结构，用于存储注册用户的基本信息和账户状态。该表采用用户名和邮箱双唯一性约束，确保用户身份的唯一性。密码字段使用bcrypt算法进行单向加密存储，最大长度为255个字符，以增强系统安全性。is_active和is_verified字段分别控制账户的激活状态和邮箱验证状态，支持用户注册后的邮箱验证流程。created_at、updated_at和last_login_at三个时间戳字段完整记录了用户账户的生命周期，为用户行为分析和账户审计提供数据支持。

**表5-4 会话表（sessions）**

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| id | VARCHAR | PRIMARY KEY | 会话唯一标识 |
| user_id | VARCHAR | FOREIGN KEY(users.id) | 关联用户ID |
| title | VARCHAR | NOT NULL | 会话标题 |
| agent_mode | BOOLEAN | DEFAULT FALSE | 是否为Agent模式 |
| project_state | JSON | NULLABLE | 项目状态（核心字段） |
| created_at | DATETIME | NOT NULL | 创建时间 |
| updated_at | DATETIME | NOT NULL | 更新时间 |

会话表（sessions）用于维护用户的分析会话状态，是系统实现多会话管理和长期记忆功能的基础。每个会话通过user_id外键关联到特定用户，支持用户创建和管理多个独立的分析项目。agent_mode布尔字段标识会话是否启用智能体模式，用于区分普通对话和自动化分析场景。project_state字段是本表的核心设计，采用JSON类型存储跨分析任务的项目状态，包括数据集信息、已完成步骤、分析产物路径等结构化数据，实现了会话状态的高效序列化和反序列化，为项目状态恢复提供了灵活的数据结构支持。

**表5-5 消息表（messages）**

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| id | VARCHAR | PRIMARY KEY | 消息唯一标识 |
| session_id | VARCHAR | FOREIGN KEY(sessions.id) | 关联会话ID |
| role | VARCHAR | NOT NULL | 消息角色：user/assistant/system |
| content | TEXT | NOT NULL | 消息内容 |
| user_metadata | JSON | NULLABLE | 用户自定义元数据 |
| timestamp | DATETIME | NOT NULL | 消息时间戳 |

消息表（messages）负责存储会话中的所有对话消息，构建完整的对话历史记录。role字段采用枚举类型约束，支持user（用户消息）、assistant（系统响应）和system（系统提示）三种消息角色，符合主流对话系统的消息模型。content字段使用TEXT类型存储消息正文，支持较长的文本内容。user_metadata字段设计为JSON类型，允许存储与消息相关的扩展信息，如消息的情感标签、意图分类结果或工具调用参数等，为消息的智能化处理提供了灵活的元数据管理能力。

**表5-6 文件表（files）**

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| id | VARCHAR | PRIMARY KEY | 文件唯一标识 |
| session_id | VARCHAR | FOREIGN KEY(sessions.id) | 关联会话ID |
| filename | VARCHAR | NOT NULL | 原始文件名 |
| filepath | VARCHAR | NOT NULL | 文件存储路径 |
| file_size | INTEGER | NULLABLE | 文件大小（字节） |
| upload_time | DATETIME | NOT NULL | 上传时间 |

文件表（files）管理用户上传的单细胞数据文件及相关元数据，是系统数据管理模块的核心数据结构。每个文件记录通过session_id外键关联到所属会话，实现文件与会话的绑定管理。filename字段保留用户上传文件的原始名称，filepath字段存储文件在服务器文件系统中的绝对路径。file_size字段记录文件大小（以字节为单位），用于存储空间统计和配额管理。该设计使得系统能够追踪文件的完整生命周期，支持文件的版本管理、重复检测和清理策略。

**表5-7 Agent运行记录表（agent_runs）**

| 字段名 | 数据类型 | 约束 | 说明 |
|--------|----------|------|------|
| id | VARCHAR | PRIMARY KEY | 运行记录唯一标识 |
| session_id | VARCHAR | FOREIGN KEY(sessions.id) | 关联会话ID |
| objective | TEXT | NOT NULL | 分析目标描述 |
| status | VARCHAR | DEFAULT 'pending' | 运行状态：pending/running/completed/failed |
| steps | JSON | NULLABLE | 执行步骤记录 |
| result | TEXT | NULLABLE | 执行结果摘要 |
| error_message | TEXT | NULLABLE | 错误信息 |
| started_at | DATETIME | NOT NULL | 开始时间 |
| completed_at | DATETIME | NULLABLE | 完成时间 |

Agent运行记录表（agent_runs）用于记录智能体分析任务的完整执行过程，是系统实现任务追踪和执行历史管理的关键数据结构。objective字段以自然语言形式记录用户的分析目标。status字段采用有限状态机模式，支持pending（待执行）、running（执行中）、completed（已完成）和failed（执行失败）四种状态转换。steps字段以JSON格式存储结构化的执行步骤记录，包括每一步的输入、输出、执行时间和状态等详细信息。result和error_message字段分别存储任务执行结果摘要和错误堆栈信息，支持失败任务的诊断和重试。started_at和completed_at时间戳字段共同记录任务的执行时长，为性能分析提供数据支持。

#### （3）project_state字段结构

`project_state`是会话表中的核心JSON字段，用于存储跨分析任务的项目状态：

```json
{
  "active_dataset": "pbmc_3k",
  "last_dataset": "pbmc_3k",
  "datasets": {
    "pbmc_3k": {
      "input_files": ["pbmc_3k.h5ad"],
      "work_dir": "runs/job_123/artifacts",
      "loaded_path": "data/pbmc_3k.h5ad",
      "qc_path": "data/qc_filtered.h5ad",
      "normalized_path": "data/normalized.h5ad",
      "pca_path": "data/pca.h5ad",
      "clustered_path": "data/clustered.h5ad",
      "markers_path": "tables/markers.csv",
      "annotated_path": "data/annotated.h5ad",
      "de_path": "tables/de_results.csv",
      "embeddings_path": "embeddings/scgpt.npy",
      "completed_steps": ["load", "qc", "normalize", "pca", "cluster", "annotate"],
      "reports": {
        "analysis_report": "reports/analysis_report.md"
      },
      "enrichment": {
        "ssgsea": {
          "result_paths": ["tables/ssgsea_results.csv"]
        }
      }
    }
  }
}
```

#### （4）文件系统组织

系统采用分层目录结构组织分析数据：

```
genomix-agent/
├── data/                          # 参考数据和知识库
│   └── references/                # 参考基因组、标记基因库
├── uploaded_data/                 # 用户上传数据
│   └── {user_id}/
│       └── {file_id}.h5ad
├── runs/                          # Job工作空间
│   └── {job_id}/                  # 每个Job独立目录
│       ├── request.json           # 请求参数
│       ├── state.json             # 实时状态
│       ├── plan.json              # 执行计划
│       ├── events.ndjson          # 事件日志
│       ├── artifacts/             # 分析产物
│       │   ├── data/              # 数据文件
│       │   ├── tables/            # 表格文件
│       │   ├── plots/             # 图表文件
│       │   └── reports/           # 报告文件
│       ├── uploads/               # Job关联的上传文件
│       └── logs/                  # 执行日志
└── chroma_data/                   # 向量数据库
```

系统采用分层目录结构组织分析数据，实现了数据分类存储和隔离管理。data目录存放参考基因组和标记基因库等静态知识资源，支持分析工具的参数配置和结果解释。uploaded_data目录按用户ID和文件ID建立两级目录结构，实现用户数据的隔离存储和快速定位。runs目录是系统的核心工作区，每个分析任务（Job）分配独立的子目录，包含请求参数、实时状态、执行计划和事件日志等元数据文件，以及artifacts子目录用于组织数据文件、表格、图表和报告等分析产物。chroma_data目录存储ChromaDB向量数据库的持久化数据，支持对话历史的语义检索和上下文恢复。

#### （5）数据表关系与约束

数据库表之间通过外键约束建立了严格的引用关系，形成完整的数据一致性保障。User表与Session表之间建立一对多关系，一个用户可创建多个会话。Session表作为中心实体，与Message、File和AgentRun表均建立一对多关系，确保所有操作记录和资源文件都能准确关联到所属会话。这种设计支持会话级别的数据隔离和批量清理操作。级联删除策略配置为：删除会话时自动级联删除关联的消息、文件和运行记录，避免孤儿数据的产生。
    Executor --> Response: 执行完成
    Replanner --> Executor: 重新规划
    Replanner --> Response: 无法恢复
    Response --> [*]
```

**状态定义**（AgentState）：

```python
class AgentState(TypedDict):
    # 基础信息
    objective: str              # 用户目标
    messages: List[BaseMessage] # 消息历史
    input_files: List[str]      # 输入文件

    # 意图与计划
    intents: List[Intent]       # 识别到的意图
    plan: List[str]            # 执行计划

    # 执行控制
    next_step: Optional[str]    # 下一步操作
    execution_status: str       # 执行状态
    replan_attempts: int        # 重规划次数

    # 上下文管理
    work_dir: Optional[str]     # 工作目录
    project_state: Dict[str, Any]  # 项目状态

    # 会话标识
    thread_id: str              # 线程ID
    session_id: str             # 会话ID
    run_id: str                 # 运行ID
```

**节点功能说明**：

| 节点名称 | 功能描述 | 输入 | 输出 |
|----------|----------|------|------|
| intent_recognition | 解析用户意图 | objective, messages | intents, next_step |
| planner | 生成执行计划 | intents, project_state | plan, next_step |
| executor | 执行分析工具 | plan, work_dir | tool_history, results |
| replanner | 处理执行失败 | error, tool_history | new_plan, next_step |
| response | 格式化响应 | all state | response_message |

#### （2）工具注册表模块

工具注册表采用**注册表模式**，实现分析工具的统一管理和动态调用。

```python
class ToolRegistry:
    """分析工具注册表"""

    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, name: str, tool: BaseTool):
        """注册工具"""

    @classmethod
    def get(cls, name: str) -> BaseTool:
        """获取工具"""

    @classmethod
    def list_tools(cls) -> List[str]:
        """列出所有工具"""
```

**工具分类**：

| 分类 | 工具名称 | 功能 |
|------|----------|------|
| 基础分析 | load_h5ad_data | 加载.h5ad格式数据 |
| 基础分析 | calculate_qc_metrics | 质量控制 |
| 基础分析 | normalize_and_hvg | 标准化与高变基因 |
| 基础分析 | pca_reduction | PCA降维 |
| 聚类注释 | cluster_and_umap | 聚类与UMAP |
| 聚类注释 | find_marker_genes | 标记基因识别 |
| 聚类注释 | annotate_cells | 细胞类型注释 |
| 差异分析 | differential_expression | 差异表达分析 |
| 高级分析 | pseudotime_analysis | 伪时序分析 |
| 高级分析 | cellphone_db | 细胞通讯分析 |
| 高级分析 | ora, ssgsea | 通路富集分析 |
| 嵌入提取 | extract_embeddings | scGPT嵌入 |

#### （3）长期记忆模块

长期记忆模块实现跨会话的上下文保持，采用**RAG（检索增强生成）**模式。

```python
class ConversationMemoryStore:
    """长期对话记忆存储"""

    def save_context(
        self,
        thread_id: str,
        objective: str,
        response: str,
        project_state: dict
    ):
        """保存上下文"""

    def load_context(
        self,
        thread_id: str,
        objective: str
    ) -> MemoryContext:
        """加载相关上下文"""

    def build_context_messages(
        self,
        context: MemoryContext
    ) -> List[BaseMessage]:
        """构建上下文消息"""
```

**记忆存储结构**：

```json
{
  "thread_id": "conv_12345",
  "summary": "用户对PBMC数据进行了聚类分析",
  "records": [
    {
      "timestamp": "2024-01-15T10:00:00",
      "objective": "对数据进行聚类",
      "result_summary": "识别到12个细胞cluster",
      "key_findings": [...]
    }
  ],
  "project_state": {
    "active_dataset": "pbmc_3k",
    "datasets": {
      "pbmc_3k": {
        "work_dir": "runs/job_123",
        "completed_steps": ["qc", "hvg", "pca", "cluster"],
        "clustered_path": "runs/job_123/artifacts/data/clustered.h5ad"
      }
    }
  }
}
```

### 5.3.3 数据与知识存储设计

#### （1）数据库设计

系统采用SQLite作为主数据库，通过SQLAlchemy ORM进行访问。图5-5展示了数据库ER图。

```
┌─────────────┐       ┌─────────────┐
│    User     │       │   Session   │
│─────────────│       │─────────────│
│ id (PK)     │◄──────│ id (PK)     │
│ username    │  1:N  │ user_id (FK)│
│ email       │       │ title       │
│ password    │       │ agent_mode  │
│ created_at  │       │ project_state│
└─────────────┘       │ created_at  │
                      └──────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼────────┐ ┌───▼────┐ ┌──────▼──────┐
     │    Message      │ │  File  │ │  AgentRun   │
     │─────────────────│ │────────│ │─────────────│
     │ id (PK)         │ │ id (PK)│ │ id (PK)     │
     │ session_id (FK) │ │session │ │session_id(FK)│
     │ role            │ │_id (FK)│ │ objective   │
     │ content         │ │filepath│ │ status      │
     │ timestamp       │ │uploaded│ │ result      │
     └─────────────────┘ │_at     │ │ started_at  │
                          └────────┘ │ completed_at│
                                     └─────────────┘
```

**表结构说明**：

**用户表（users）**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | String | 主键 |
| username | String(50) | 用户名，唯一 |
| email | String(100) | 邮箱，唯一 |
| hashed_password | String(255) | 加密密码 |
| is_active | Boolean | 账户状态 |
| created_at | DateTime | 创建时间 |

用户表存储系统用户的基本认证信息。id字段作为主键采用UUID格式确保全局唯一性。username和email字段设置唯一约束，防止重复注册。hashed_password字段存储经过bcrypt加密的密码哈希值，原始密码不落地存储以保障安全性。is_active字段控制账户的启用状态，支持管理员禁用违规账户。

**会话表（sessions）**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | String | 主键 |
| user_id | String | 外键，关联用户 |
| title | String | 会话标题 |
| project_state | JSON | 项目状态（核心字段） |
| created_at | DateTime | 创建时间 |

会话表是系统实现多任务管理的核心数据结构。每个会话通过user_id外键关联到创建用户，支持一个用户创建多个独立的分析项目。project_state字段采用JSON类型存储项目状态，包括数据集路径、已完成步骤列表和分析产物位置等结构化信息。这种设计使得会话可以完整保存分析进度，支持中断恢复和历史会话重载。

**Agent运行记录表（agent_runs）**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | String | 主键 |
| session_id | String | 外键，关联会话 |
| objective | Text | 分析目标 |
| status | String | 运行状态 |
| steps | JSON | 执行步骤 |
| result | Text | 执行结果 |
| started_at | DateTime | 开始时间 |
| completed_at | DateTime | 完成时间 |

Agent运行记录表用于追踪智能体任务的执行过程。objective字段记录用户的自然语言分析目标。status字段采用状态枚举，包括pending（待执行）、running（执行中）、completed（成功完成）和failed（执行失败）四种状态，支持任务状态的精确追踪。steps字段以JSON格式存储详细的执行步骤序列，每步包含工具名称、参数、执行时间和结果状态。result字段存储执行结果的文本摘要，error_message字段在任务失败时记录错误堆栈信息用于问题诊断。started_at和completed_at字段共同记录任务的执行耗时。

#### （2）文件系统组织

系统采用分层目录结构组织分析数据：

```
genomix-agent/
├── data/                          # 参考数据和知识库
│   ├── references/                # 参考基因组、标记基因库
│   └── knowledge/                 # 领域知识
├── uploaded_data/                 # 用户上传数据
│   └── {user_id}/
│       └── {file_id}.h5ad
├── runs/                          # Job工作空间
│   └── {job_id}/                  # 每个Job独立目录
│       ├── request.json           # 请求参数
│       ├── state.json             # 实时状态
│       ├── plan.json              # 执行计划
│       ├── events.ndjson          # 事件日志
│       ├── artifacts/             # 分析产物
│       │   ├── data/              # 数据文件
│       │   ├── tables/            # 表格文件
│       │   ├── plots/             # 图表文件
│       │   └── reports/           # 报告文件
│       ├── uploads/               # Job关联的上传文件
│       └── logs/                  # 执行日志
├── output/                        # 通用输出
├── logs/                          # 系统日志
└── chroma_data/                   # 向量数据库
```

系统文件系统采用分层组织结构，实现不同类型数据的隔离存储。data目录存放参考基因组序列和细胞类型标记基因库等静态知识资源，为分析工具提供领域知识支持。uploaded_data目录采用"用户ID/文件ID"的两级目录结构，实现用户数据的逻辑隔离和权限控制。runs目录为每个分析任务创建独立的工作空间，包含请求参数、实时状态、执行计划和事件日志等元数据文件。artifacts子目录按照data、tables、plots和reports四个子类目组织分析产物，便于结果的定位和导出。chroma_data目录存储ChromaDB向量数据库的持久化文件，支持对话历史的语义检索。

#### （3）Job隔离机制

每个分析任务（Job）在独立的工作空间中运行，确保系统的稳定性和可追溯性。Job隔离机制的核心设计原则如下：

1. **数据隔离**：不同Job的数据互不干扰，避免并发执行时的资源冲突。每个Job拥有独立的artifacts目录，中间结果和最终产物隔离存储，防止文件覆盖和路径混淆。

2. **状态追踪**：完整记录执行过程和中间结果。state.json文件实时更新Job的执行状态，events.ndjson文件按时间顺序记录所有执行事件，支持任务执行的细粒度监控和问题定位。

3. **错误恢复**：失败后可从检查点恢复。系统在每个工具执行前后保存检查点，当任务失败时可通过重规划机制调整执行策略，从失败的步骤恢复执行，避免全量重跑带来的时间开销。

4. **结果管理**：分析产物结构化存储。artifacts目录按照数据类型（data、tables、plots、reports）分类组织，每个产物文件通过相对路径在project_state中注册，支持前端界面的快速定位和渲染。

## 5.4 系统开发与实现

### 5.4.1 开发工具与环境

本系统的开发与运行环境如表5-1所示。

**表5-1 开发环境配置**

| 类别 | 工具/框架 | 版本 | 用途 |
|------|-----------|------|------|
| 操作系统 | Ubuntu/Linux | 20.04+ | 服务器环境 |
| 编程语言 | Python | 3.10+ | 主要开发语言 |
| 后端框架 | FastAPI | 0.100+ | Web API |
| 前端框架 | Streamlit | 1.28+ | 用户界面 |
| 智能体编排 | LangGraph | 0.0.20+ | 工作流编排 |
| ORM框架 | SQLAlchemy | 2.0+ | 数据库访问 |
| 单细胞分析 | Scanpy | 1.10+ | 核心分析库 |
| 数据处理 | Pandas/NumPy | 2.0+/1.24+ | 数据操作 |
| 可视化 | Matplotlib/Plotly | 3.7+/2.18+ | 图表生成 |
| 容器化 | Docker | 24.0+ | 服务部署 |
| 数据库 | SQLite | 3.x | 数据持久化 |

### 5.4.2 主要功能模块实现

#### （1）智能体工作流实现

智能体工作流基于LangGraph框架实现，核心代码结构如下：

```python
def build_graph():
    """构建Agent状态图"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("intent_recognition", intent_recognition)
    graph.add_node("planner", general_planner)
    graph.add_node("executor", general_executor)
    graph.add_node("replanner", intelligent_replanner)
    graph.add_node("response", response_node)

    # 构建状态转换
    graph.add_edge(START, "intent_recognition")
    graph.add_conditional_edges(
        "intent_recognition",
        route_after_intent,
        {"planner": "planner", "response": "response"}
    )
    # ... 其他转换

    # 编译（带检查点支持）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
```

**关键特性实现**：

1. **状态检查点**：使用MemorySaver实现状态持久化，支持中断恢复

2. **条件路由**：根据执行状态动态选择下一步操作

3. **流式输出**：通过SSE（Server-Sent Events）实时推送进度

#### （2）意图识别实现

意图识别模块采用结构化提示词引导LLM进行分析：

```python
INTENT_RECOGNITION_PROMPT = """
你是一个单细胞数据分析专家。请分析用户的目标，识别需要的分析意图。

可用意图类型：
- load_data: 加载数据文件
- quality_control: 质量控制分析
- normalize: 数据标准化
- dimensionality_reduction: 降维分析（PCA/UMAP）
- clustering: 细胞聚类
- annotation: 细胞类型注释
- marker_genes: 标记基因识别
- differential_expression: 差异表达分析
- trajectory_analysis: 伪时序分析
- cell_communication: 细胞通讯分析
- pathway_enrichment: 通路富集分析

用户目标：{objective}

请以JSON格式返回识别结果。
"""
```

**意图模型定义**：

```python
class Intent(BaseModel):
    """意图识别模式"""
    intent: str                    # 标准化意图标签
    description: str               # 任务详细描述
    confidence: float              # 置信度 (0-1)
    dependencies: List[str]        # 前置依赖意图
    justification: str             # 选择理由
```

#### （3）核心分析工具实现

系统实现了完整的单细胞分析工具链。以下展示核心工具的实现要点。

**数据加载工具**：

```python
@tool("load_h5ad_data")
def load_h5ad_data(
    file_path: Optional[str] = None,
    cache: bool = True,
) -> str:
    """加载.h5ad格式的单细胞数据文件"""
    # 路径解析与验证
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(settings.UPLOAD_DIR) / path.name

    # 数据加载
    adata = sc.read_h5ad(path)

    # 提取基础信息
    result = {
        "status": "success",
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "has_clustering": any(col in adata.obs.columns
                             for col in ['leiden', 'louvain']),
        "has_embedding": 'X_umap' in adata.obsm
    }

    return json.dumps(result, ensure_ascii=False)
```

**质量控制工具**：

```python
@tool("calculate_qc_metrics")
def calculate_qc_metrics(
    file_path: str,
    min_genes: int = 200,
    min_cells: int = 3,
    mt_prefix: str = "MT-",
) -> str:
    """计算单细胞数据的质控指标"""
    adata = sc.read_h5ad(path)

    # 计算质控指标
    adata.var['mt'] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

    # 过滤低质量细胞
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 统计信息
    qc_stats = {
        "n_cells_before": n_cells_before,
        "n_cells_after": adata.n_obs,
        "mean_genes_per_cell": float(adata.obs['n_genes_by_counts'].mean()),
        "mean_mt_percent": float(adata.obs['pct_counts_mt'].mean()),
    }

    return json.dumps(qc_stats, ensure_ascii=False)
```

**聚类与UMAP工具**：

```python
@tool("cluster_and_umap")
def cluster_and_umap(
    file_path: str,
    resolution: float = 0.5,
    n_neighbors: int = 15,
) -> str:
    """聚类分析和UMAP降维"""
    adata = sc.read_h5ad(path)

    # 计算邻接图
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=40)

    # Leiden聚类
    sc.tl.leiden(adata, resolution=resolution)

    # UMAP降维
    sc.tl.umap(adata)

    # 导出UMAP坐标
    umap_df = pd.DataFrame(adata.obsm['X_umap'],
                          columns=['UMAP_1', 'UMAP_2'])
    umap_df['cluster'] = adata.obs['leiden'].values
    umap_path = tables_dir / f"umap_coords_{timestamp}.csv"
    umap_df.to_csv(umap_path, index=False)

    return json.dumps({
        "n_clusters": len(adata.obs['leiden'].unique()),
        "umap_coords_path": str(umap_path)
    }, ensure_ascii=False)
```

#### （4）API接口实现

系统采用FastAPI实现RESTful API，关键端点如下：

```python
# Agent执行端点
@app.post("/api/agent/run")
async def run_agent(request: AgentRunRequest):
    """启动Agent分析任务"""
    state = create_initial_state(
        objective=request.objective,
        input_files=request.input_files,
        thread_id=request.thread_id
    )

    job_id = job_manager.create_job(state)
    # 异步执行
    asyncio.create_task(execute_agent_graph(job_id))

    return {"job_id": job_id, "status": "started"}

# 流式进度端点
@app.get("/api/agent/stream/{job_id}")
async def stream_agent_events(job_id: str):
    """流式推送Agent执行事件"""
    async def event_generator():
        async for event in job_manager.stream_events(job_id):
            yield f"data: {event.json()}\n\n"

    return EventSourceResponse(event_generator())
```

#### （5）前端界面实现

前端采用Streamlit构建，提供交互式分析界面：

```python
import streamlit as st

# 会话管理
if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session()

# 数据上传
uploaded_file = st.file_uploader(
    "上传单细胞数据",
    type=["h5ad"],
    help="支持.h5ad格式的单细胞数据"
)

# 目标输入
objective = st.text_area(
    "描述你的分析目标",
    placeholder="例如：对这批PBMC数据进行聚类分析，识别主要的细胞类型..."
)

# 执行分析
if st.button("开始分析") and objective:
    with st.spinner("正在执行分析..."):
        result = run_agent_analysis(objective, uploaded_file)
    st.success("分析完成！")
```

### 5.4.3 scGPT微服务实现

scGPT（单细胞基础模型）作为独立微服务部署，避免PyTorch版本冲突。

**服务端实现**：

```python
# scGPT服务主程序
from fastapi import FastAPI
import torch
from scgpt import GeneEmbedding

app = FastAPI()
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = GeneEmbedding.from_pretrained("scGPT")

@app.post("/embeddings")
async def extract_embeddings(request: EmbeddingRequest):
    """提取scGPT嵌入"""
    embeddings = model.encode(request.data)
    return {"embeddings": embeddings.tolist()}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Docker配置**：

```dockerfile
# Dockerfile.scgpt
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

RUN pip install scgpt==0.1.0
COPY src/scgpt_service/ /app/

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

### 5.4.4 Docker部署配置

系统使用Docker Compose进行服务编排：

```yaml
version: '3.8'

services:
  genomix-main:
    build:
      dockerfile: Dockerfile.main
    ports:
      - "8000:8000"
    environment:
      - SCGPT_SERVICE_URL=http://genomix-scgpt:8001
    volumes:
      - ./data:/app/data
      - ./runs:/app/runs
    depends_on:
      - genomix-scgpt

  genomix-scgpt:
    build:
      dockerfile: Dockerfile.scgpt
    ports:
      - "8001:8001"
    volumes:
      - ./data:/app/data
```

## 5.5 系统测试

### 5.5.1 测试环境

测试环境配置如下：

| 项目 | 配置 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| CPU | Intel Xeon 8核 |
| 内存 | 32 GB |
| 存储 | 500 GB SSD |
| 浏览器 | Chrome 120+, Firefox 120+ |

### 5.5.2 功能测试

#### （1）核心功能测试用例

**表5-2 功能测试用例**

| 编号 | 测试功能 | 输入/操作 | 预期结果 | 实际结果 | 通过 |
|------|----------|-----------|----------|----------|------|
| TC-01 | 用户登录 | 用户名/密码 | 登录成功，跳转主页 | 登录成功 | ✓ |
| TC-02 | 数据上传 | 上传.h5ad文件 | 文件保存，元数据提取成功 | 成功 | ✓ |
| TC-03 | 意图识别-简单 | "对数据进行聚类" | 识别clustering意图 | 正确识别 | ✓ |
| TC-04 | 意图识别-复杂 | "分析PBMC数据，找出T细胞和B细胞的差异基因" | 识别DEG意图 | 正确识别 | ✓ |
| TC-05 | 自动规划 | 聚类分析意图 | 生成完整计划（QC→HVG→PCA→聚类） | 计划完整 | ✓ |
| TC-06 | QC分析 | PBMC数据 | 质控指标正确计算 | 结果正确 | ✓ |
| TC-07 | 聚类分析 | 已质控数据 | 识别合理数量的cluster | 12 clusters | ✓ |
| TC-08 | 细胞注释 | 聚类结果 | 自动注释主要类型 | 注释准确 | ✓ |
| TC-09 | 进度流式推送 | 执行分析任务 | 实时接收进度更新 | 流式正常 | ✓ |
| TC-10 | 长期记忆 | 恢复历史会话 | 项目状态正确恢复 | 恢复成功 | ✓ |

#### （2）数据管理模块功能测试

数据管理模块负责单细胞数据文件的上传、验证、存储和管理。用户可以通过点击"上传文件"按钮或直接拖拽文件至指定区域来上传单细胞数据文件。系统支持.h5ad（AnnData）格式的标准化单细胞数据文件，上传过程中会对文件格式与大小进行校验。上传成功后，数据条目将显示于页面对话框列表中，用户可点击"查看"按钮浏览数据详细信息，或点击"删除"按钮从当前会话中移除数据。

**表5-3 数据管理模块测试用例**

| 编号 | 测试内容 | 预期结果 | 测试结果 |
|------|----------|----------|----------|
| TC-DM-01 | 上传.h5ad格式文件 | 文件成功上传，显示文件信息 | 文件上传成功，显示文件名、大小等信息 |
| TC-DM-02 | 上传.csv/.txt等非h5ad格式文件 | 系统提示"仅支持.h5ad格式" | 系统弹出格式错误提示，拒绝上传 |
| TC-DM-03 | 上传500MB文件（小于2GB限制） | 文件正常上传 | 文件正常上传，上传进度条正常显示 |
| TC-DM-04 | 上传2.5GB文件（超过限制） | 系统提示"文件大小超过限制" | 系统检测文件超限，弹出大小限制提示 |
| TC-DM-05 | 拖拽文件至上传区域 | 拖拽区域高亮，文件上传成功 | 拖拽进入时区域边框高亮，松手后上传成功 |
| TC-DM-06 | 点击上传按钮选择文件 | 打开文件选择对话框，上传成功 | 打开系统文件选择窗口，选择后上传成功 |
| TC-DM-07 | 上传含2700细胞的文件 | 正确显示"细胞数: 2700" | 元数据提取正确，显示细胞数为2700 |
| TC-DM-08 | 上传含3000基因的文件 | 正确显示"基因数: 3000" | 元数据提取正确，显示基因数为3000 |
| TC-DM-09 | 上传已包含leiden聚类的文件 | 显示"已包含聚类结果" | 系统识别到leiden列，显示已聚类标签 |
| TC-DM-10 | 连续上传3个.h5ad文件 | 列表按时间倒序显示所有文件 | 3个文件均显示在列表中，按上传时间倒序排列 |
| TC-DM-11 | 点击"查看"按钮查看已上传文件 | 弹窗显示完整元数据信息 | 弹窗展示细胞数、基因数、是否有聚类等详细信息 |
| TC-DM-12 | 点击"删除"按钮并确认 | 文件从列表中移除 | 确认后文件条目从列表中消失 |
| TC-DM-13 | 点击"删除"后选择取消 | 文件保留在列表中 | 取消后文件仍保留在列表中，未删除 |
| TC-DM-14 | 上传与已有文件同名的文件 | 系统提示"文件已存在" | 系统检测到同名文件，提示覆盖或重命名 |
| TC-DM-15 | 上传文件名含特殊字符@#$%的文件 | 文件正常处理和存储 | 系统正常处理文件名，文件成功存储 |

数据管理模块测试覆盖了文件上传的完整流程，包括格式校验、大小限制、拖拽和按钮两种上传方式、元数据提取的准确性、文件列表管理以及删除操作的交互逻辑。测试结果表明，系统能够正确识别并拒绝不支持的文件格式和超大文件，拖拽上传功能响应流畅，元数据提取准确率达到100%，删除操作具备完善的确认机制。

#### （3）数据分析模块功能测试

数据分析模块是本系统的核心功能部分，包含数据智能分析、报告查看和自然语言交互三个子模块。数据智能分析子模块基于智能体工作流自动执行单细胞分析流程；报告查看子模块提供分析结果的可视化展示和导出功能；自然语言交互子模块支持用户通过对话方式描述分析目标并获得响应。

**表5-4 数据智能分析子模块测试用例**

| 编号 | 测试内容 | 预期结果 | 测试结果 |
|------|----------|----------|----------|
| TC-DA-01 | 调用load_h5ad_data工具加载.h5ad文件 | 返回细胞数、基因数等基础信息 | 成功加载2700个细胞、3000个基因，检测到包含聚类信息 |
| TC-DA-02 | 调用calculate_qc_metrics工具执行质控 | 过滤低质量细胞，生成QC统计 | 过滤后保留2580个细胞，生成包含质控指标的JSON报告 |
| TC-DA-03 | 调用normalize_and_hvg工具执行标准化 | 识别高变基因，对数转换数据 | 识别到2147个高变基因（n_top_genes=2000），数据完成log1p转换 |
| TC-DA-04 | 调用pca_reduction工具执行PCA降维 | 计算主成分，返回方差解释比例 | 前50个主成分累积解释度达85.3%，PC1解释度12.5% |
| TC-DA-05 | 调用cluster_and_umap工具执行聚类 | 使用Leiden算法聚类，计算UMAP | 识别出12个cluster，生成UMAP二维坐标文件 |
| TC-DA-06 | 调用find_marker_genes工具识别标记基因 | 使用Wilcoxon检验寻找每个cluster的标记基因 | 每个cluster识别出25个标记基因（n_genes=25） |
| TC-DA-07 | 调用annotate_cells工具执行细胞注释 | 基于标记基因自动注释细胞类型 | 识别出T细胞、B细胞、单核细胞、NK细胞等主要类型 |
| TC-DA-08 | 调用differential_expression工具进行差异分析 | 比较两组细胞的基因表达差异 | 生成包含上调/下调基因的CSV表格和统计数据 |
| TC-DA-09 | 调用generate_analysis_report工具生成报告 | 汇总所有分析结果生成Markdown报告 | 生成包含数据概览、分析状态、聚类信息的完整报告 |
| TC-DA-10 | 直接在未降维数据上执行聚类工具 | 系统检测到缺失PCA结果自动补全 | 系统自动执行PCA降维后再进行聚类 |
| TC-DA-11 | 执行分析任务时通过SSE查看进度 | 实时接收工具执行进度事件 | 前端通过EventSource接收到step_start、step_complete等事件 |
| TC-DA-12 | 检查工具执行产物的文件结构 | 按artifacts/目录分类存储 | 生成data/、tables/、plots/、reports/四个子目录的产物 |
| TC-DA-13 | 查看聚类工具生成的UMAP坐标文件 | 生成CSV格式的坐标数据 | umap_coords.csv包含UMAP_1、UMAP_2、cluster三列数据 |

**表5-5 报告查看子模块测试用例**

| 编号 | 测试内容 | 预期结果 | 测试结果 |
|------|----------|----------|----------|
| TC-RV-01 | UmapVisualization组件渲染UMAP散点图 | D3.js正确渲染数据点，按cluster着色 | 组件渲染2700个圆点，12个cluster用不同颜色区分 |
| TC-RV-02 | 点击UMAP图例隐藏特定cluster | 图例切换时更新显示状态 | 点击cluster 0图例后，该cluster数据点隐藏 |
| TC-RV-03 | 鼠标悬停UMAP数据点 | 显示细胞的详细信息 | Tooltip显示cluster ID和坐标信息 |
| TC-RV-04 | VolcanoPlot组件渲染差异表达火山图 | 显示logFC和-pvalue的散点分布 | D3.js正确渲染500个基因数据点，显著基因高亮显示 |
| TC-RV-05 | 鼠标悬停火山图上的基因点 | 显示基因名称和统计值 | Tooltip显示基因名、logFC值和p值 |
| TC-RV-06 | ClusterHeatmap组件渲染标记基因热图 | 显示各cluster的top标记基因表达模式 | 热图正确显示，颜色梯度从蓝（低表达）到红（高表达） |
| TC-RV-07 | PlotGallery组件展示所有图表 | 以网格形式展示所有分析图表 | 3×2网格布局显示QC图、UMAP图、热图等 |
| TC-RV-08 | 点击导出Markdown格式报告 | 下载.md格式的分析报告 | 成功下载包含完整分析内容的Markdown文件 |
| TC-RV-09 | 点击导出HTML格式报告 | 下载.html格式的分析报告 | 成功下载带样式的HTML文件，可在浏览器直接打开 |
| TC-RV-10 | 点击导出PDF格式报告 | 调用浏览器打印功能生成PDF | 打开新窗口预览HTML报告，触发打印对话框 |

**表5-6 自然语言交互子模块测试用例**

| 编号 | 测试内容 | 预期结果 | 测试结果 |
|------|----------|----------|----------|
| TC-NL-01 | 输入"对数据进行聚类分析" | 意图识别为clustering，生成执行计划 | 系统识别到clustering意图，返回QC→HVG→PCA→UMAP→Cluster计划 |
| TC-NL-02 | 输入"找出T细胞和B细胞的差异基因" | 意图识别为differential_expression | 系统识别到差异表达意图，设置group1="T cells"，group2="B cells" |
| TC-NL-03 | 输入"帮我分析数据"模糊指令 | 系统请求用户澄清具体需求 | 系统回复"请明确您想进行哪种类型的分析，如聚类、差异表达等" |
| TC-NL-04 | 第一轮执行聚类后输入"查看标记基因" | 正确引用上一次的聚类结果 | 系统使用聚类后的数据调用find_marker_genes工具 |
| TC-NL-05 | 输入分析目标后查看执行计划 | 显示结构化的步骤列表 | 界面展示包含工具名、参数说明的JSON格式计划 |
| TC-NL-06 | 修改计划中的resolution参数 | 计划实时更新参数值 | 输入resolution=1.0后，计划中的参数同步更新 |
| TC-NL-07 | 分析完成后查看结果摘要 | 生成自然语言形式的结果描述 | 系统回复"聚类分析完成，识别到12个cluster，主要细胞类型为..." |
| TC-NL-08 | 在未上传数据时执行分析请求 | 系统提示缺少必要数据 | 系统回复"请先上传单细胞数据文件" |
| TC-NL-09 | 切换到历史会话列表中的项目 | 恢复会话的对话历史和project_state | 界面显示历史消息和分析结果，project_state正确加载 |
| TC-NL-10 | 输入"继续上次的分析" | 系统检索project_state恢复上下文 | 系统加载上次的数据路径和已完成步骤，可继续分析 |

数据分析模块的测试全面覆盖了三个子模块的核心功能。数据智能分析子模块测试验证了load_h5ad_data、calculate_qc_metrics、normalize_and_hvg、pca_reduction、cluster_and_umap、find_marker_genes、annotate_cells、differential_expression和generate_analysis_report等九个核心分析工具的正确性，测试表明系统能够自动检测并补全缺失的前置步骤。报告查看子模块测试验证了UmapVisualization、VolcanoPlot、ClusterHeatmap等可视化组件的渲染质量，以及Markdown、HTML、PDF三种格式的报告导出功能。自然语言交互子模块测试验证了意图识别模块对简单指令和复合指令的处理能力，以及跨会话的project_state恢复机制。

#### （4）分析准确性验证

使用公开的PBMC 3K数据集（10x Genomics）进行验证，与官方分析结果对比：

| 分析项目 | 系统结果 | 参考结果 | 一致性 |
|----------|----------|----------|--------|
| 细胞数（过滤后） | 2,693 | 2,700 | 99.7% |
| 识别的cluster数 | 12 | 12-15 | 一致 |
| 主要细胞类型 | T细胞、B细胞、单核细胞等 | 相同 | 一致 |
| 标记基因（CD3D） | logFC=4.2, p=1e-80 | logFC≈4.0, p<1e-50 | 一致 |

### 5.5.3 性能测试

对PBMC 10K数据集进行性能测试：

| 测试项目 | 数据规模 | 耗时 | 内存占用 |
|----------|----------|------|----------|
| 数据加载 | 10,000细胞 × 30,000基因 | 3.2s | 1.2GB |
| 质量控制 | 同上 | 5.8s | 1.5GB |
| 标准化+HVG | 同上 | 8.1s | 1.8GB |
| PCA降维 | 同上 | 12.3s | 2.1GB |
| 聚类+UMAP | 同上 | 18.5s | 2.3GB |
| 标记基因 | 同上 | 25.7s | 2.5GB |
| **完整流程** | **10,000细胞** | **~60s** | **<3GB** |

### 5.5.4 系统界面测试

系统界面在不同浏览器和分辨率下的兼容性测试：

| 浏览器 | 版本 | 1920×1080 | 1366×768 | 移动端 |
|--------|------|-----------|-----------|--------|
| Chrome | 120+ | ✓ | ✓ | 部分支持 |
| Firefox | 120+ | ✓ | ✓ | 部分支持 |
| Safari | 17+ | ✓ | ✓ | 部分支持 |
| Edge | 120+ | ✓ | ✓ | 部分支持 |

## 5.6 本章小结

本章详细阐述了单细胞数据智能分析系统的需求分析、系统设计、开发实现与测试验证过程。

**主要工作总结**：

1. **需求分析**：通过业务流程分析和用例建模，明确了系统的功能需求和非功能需求，识别出7大功能模块和20个核心用例。

2. **系统设计**：采用微服务架构与分层架构相结合的设计模式，实现了表现层、API层、业务逻辑层、服务层、工具层和数据层的清晰分离。设计了基于LangGraph的智能体状态机，支持意图识别、自动规划、工具执行、智能重规划和结果响应的完整工作流。

3. **系统实现**：基于FastAPI、Streamlit、LangGraph和Scanpy等成熟技术栈，实现了完整的系统功能。通过Docker容器化部署，实现了scGPT微服务的隔离，解决了深度学习框架版本冲突问题。

4. **系统测试**：功能测试验证了系统核心功能的正确性；准确性验证使用公开数据集确认了分析结果的可靠性；性能测试表明系统可处理万级细胞规模的数据。

**系统创新点**：

1. **自然语言交互**：用户无需编写代码，通过自然语言描述即可完成复杂的单细胞分析流程。

2. **智能规划与执行**：系统能够根据用户意图自动生成分析计划，并支持失败后的智能重规划。

3. **长期记忆支持**：通过会话持久化和上下文检索，实现了跨会话的项目状态恢复。

4. **微服务架构**：scGPT等计算密集型服务独立部署，提高了系统的可扩展性和稳定性。

本系统的实现为验证本文提出的理论方法提供了有效平台，也为单细胞数据分析领域的智能化工具开发提供了参考。下一章将对全文工作进行总结，并探讨未来的改进方向。
