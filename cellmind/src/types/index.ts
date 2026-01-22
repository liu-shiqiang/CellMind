// ============= 认证相关 =============
export type MessageRole = 'user' | 'assistant' | 'system';

export interface Message {
  id: string;
  sessionId: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  metadata?: {
    agentRunId?: string;
    stepId?: string;
    [key: string]: unknown;
  };
}

// ============= 用户认证相关 =============
export interface User {
  id: string;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login_at?: string;
  is_anonymous?: boolean;
}

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}

export interface AuthResponse extends AuthTokens {
  user: User;
}

// ============= Agent相关 =============
export type AgentRole =
  | 'planner' | 'executor' | 'replanner' | 'response'
  // LangGraph 节点名称
  | 'intent_recognition' | 'general_planner' | 'general_executor'
  | 'intelligent_replanner' | 'final_response' | 'START'
  | string;  // 允许任意字符串

export type StepStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface AgentStep {
  id: string;
  role: AgentRole;
  task: string;
  status: StepStatus;
  output?: string;
  toolUsed?: string;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
}

export interface AgentRun {
  id: string;
  sessionId: string;
  objective: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  steps: AgentStep[];
  createdAt: Date;
  completedAt?: Date;
}

// ============= 会话相关 =============
export interface Session {
  id: string;
  title: string;
  created_at: string;  // ISO format string from backend
  updated_at: string;  // ISO format string from backend
  message_count: number;
  agent_mode: boolean;

  // 兼容属性（从后端数据计算）
  createdAt?: Date;    // 已废弃，使用 created_at
  updatedAt?: Date;    // 已废弃，使用 updated_at
  messageCount?: number;  // 已废弃，使用 message_count
  agentMode?: boolean;    // 已废弃，使用 agent_mode

  uploadedFile?: {
    id: string;
    name: string;
    size: number;
  };
}

// ============= 文件相关 =============
export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  path: string;
  uploadedAt: Date;
}

// ============= 分析报告相关 =============
export interface AnalysisReport {
  id: string;
  runId: string;
  title: string;
  content: string;  // Markdown 格式的完整报告
  summary?: string;  // 报告摘要
  sections?: ReportSection[];
  createdAt: Date;
  metadata?: {
    dataFile?: string;
    nCells?: number;
    nGenes?: number;
    [key: string]: unknown;
  };
}

export interface ReportSection {
  id: string;
  title: string;
  content: string;
  status: 'success' | 'warning' | 'error';
}

// ============= UI状态相关 =============
export type ViewMode = 'chat' | 'split';

// ============= SSE事件 =============
export type SSEEventType =
  | 'plan_generated' | 'step_started' | 'step_completed' | 'step_failed'
  | 'analysis_complete' | 'error' | 'heartbeat' | 'start' | 'end'
  // 后端实际发送的事件类型
  | 'node_enter' | 'tool_call' | 'tool_result' | 'plan_update' | 'progress' | 'token'
  // 报告相关
  | 'report_generated';

export interface SSEEvent {
  type?: SSEEventType;
  event_type?: SSEEventType;  // 兼容字段
  run_id?: string;
  thread_id?: string;
  runId?: string;  // 兼容字段
  node?: string;
  ts?: string;
  timestamp?: string;
  payload?: {
    message?: string;
    progress?: number;
    plan?: string[];
    tool_count?: number;
    token?: string;
    is_complete?: boolean;
    response?: { content?: string };
    report?: AnalysisReport;  // 报告数据
    [key: string]: unknown;
  };
  data?: {  // 兼容字段
    sessionId?: string;
    message?: string;
    error?: string;
    status?: string;
    report?: AnalysisReport;  // 报告数据
    [key: string]: unknown;
  };
}
