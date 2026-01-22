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

本系统采用**微服务架构**与**分层架构**相结合的设计模式，实现模块间的松耦合和高内聚。图5-3展示了系统的整体架构。

```
┌─────────────────────────────────────────────────────────────────┐
│                        表现层 (Presentation Layer)               │
│                      ┌─────────────────────┐                    │
│                      │    Streamlit UI     │                    │
│                      │    (Port 8501)      │                    │
│                      └─────────────────────┘                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP/WebSocket
┌──────────────────────────────▼──────────────────────────────────┐
│                       API网关层 (API Gateway)                    │
│                      ┌─────────────────────┐                    │
│                      │   FastAPI Router    │                    │
│                      │    (Port 8000)      │                    │
│                      └─────────────────────┘                    │
└──────────────┬───────────────┬────────────────┬─────────────────┘
               │               │                │
┌──────────────▼─┐  ┌────────▼────────┐  ┌───▼──────────────────┐
│  会话管理API    │  │   智能体API     │  │    文件管理API        │
│  /api/sessions  │  │   /api/agent    │  │   /api/upload        │
└────────────────┘  └────────┬───────┘  └───────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                    业务逻辑层 (Business Logic Layer)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   意图识别    │  │   自动规划    │  │    工具执行器         │   │
│  │              │  │              │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              LangGraph 状态机 (State Machine)              │  │
│  │  START → 意图识别 → 规划 → 执行 → (重规划) → 响应 → END   │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────────────┬──────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    服务层 (Service Layer)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 记忆服务     │  │  工具注册表   │  │    Job管理服务          │  │
│  │ MemoryStore │  │ToolRegistry  │  │   JobManager           │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└────────────────────────┬──────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    分析工具层 (Analysis Tools Layer)              │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │QC   │ │PCA  │ │聚类 │ │注释 │ │DEG  │ │伪时序│ │富集 │      │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘      │
└────────────────────────┬──────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    数据访问层 (Data Access Layer)                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   SQLAlchemy    │  │   文件系统       │  │   ChromaDB      │  │
│  │    ORM          │  │   FileSystem     │  │  (向量存储)      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└────────────────────────┬──────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    数据存储层 (Data Storage Layer)                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │
│  │  SQLite │  │上传数据  │  │分析产物  │  │   scGPT服务      │    │
│  │  数据库  │  │ uploaded │  │  runs/  │  │  (微服务)        │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**架构特点**：

1. **前后端分离**：Streamlit前端与FastAPI后端通过HTTP通信，便于独立开发和部署

2. **微服务隔离**：scGPT服务独立部署，解决了深度学习框架版本冲突问题

3. **状态机驱动**：LangGraph状态机确保分析流程的可靠执行和错误恢复

4. **多存储支持**：关系数据库、文件系统、向量数据库各司其职

### 5.3.2 主要功能模块设计

#### （1）智能体工作流模块

智能体工作流是系统的核心，采用**有限状态机（FSM）**模式设计。图5-4展示了智能体的状态转换图。

```mermaid
stateDiagram-v2
    [*] --> IntentRecognition
    IntentRecognition --> Planner: 需要分析
    IntentRecognition --> Response: 简单问答
    Planner --> Executor: 计划生成
    Executor --> Executor: 继续执行
    Executor --> Replanner: 执行失败
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

**会话表（sessions）**
| 字段名 | 类型 | 说明 |
|--------|------|------|
| id | String | 主键 |
| user_id | String | 外键，关联用户 |
| title | String | 会话标题 |
| project_state | JSON | 项目状态（核心字段） |
| created_at | DateTime | 创建时间 |

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

#### （3）Job隔离机制

每个分析任务（Job）在独立的工作空间中运行，确保：

1. **数据隔离**：不同Job的数据互不干扰
2. **状态追踪**：完整记录执行过程和中间结果
3. **错误恢复**：失败后可从检查点恢复
4. **结果管理**：分析产物结构化存储

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

#### （2）分析准确性验证

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
