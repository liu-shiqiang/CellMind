/**
 * useAgentStream - Agent专用Hook
 * 基于useSSE封装，提供Agent特定的流式处理功能
 */
import { useCallback, useEffect, useState } from 'react';
import { useSSE, SSEEvent, SSEStatus } from './useSSE';
import type { AgentRun, AgentStep, StepStatus, AnalysisReport } from '@/types';

export interface AgentStreamOptions {
  /** 运行ID（如果已有） */
  runId?: string;
  /** 会话ID */
  sessionId?: string;
  /** 目标描述 */
  objective?: string;
  /** 上传的文件ID列表 */
  files?: string[];
  /** SSE基础URL */
  baseUrl?: string;
  /** 使用Jobs API */
  useJobs?: boolean;
  /** 进入节点时的回调 */
  onNodeEnter?: (node: string, message?: string) => void;
  /** 调用工具时的回调 */
  onToolCall?: (tool: string, args: any) => void;
  /** 工具返回结果时的回调 */
  onToolResult?: (tool: string, result: any) => void;
  /** 进度更新回调 */
  onProgress?: (progress: number, message?: string) => void;
  /** 计划更新回调 */
  onPlanUpdate?: (plan: string[]) => void;
  /** 报告生成回调 */
  onReport?: (report: AnalysisReport) => void;
  /** 完成回调 */
  onComplete?: (message: string) => void;
  /** 错误回调 */
  onError?: (error: string) => void;
  /** 开始回调 */
  onStart?: (runId: string, sessionId: string) => void;
  /** 流式token回调 */
  onToken?: (token: string, isComplete?: boolean) => void;
  /** SSE流模式 */
  streamMode?: 'updates' | 'messages' | 'debug';
}

export interface AgentStreamReturn {
  /** 当前运行状态 */
  status: SSEStatus;
  /** 是否正在运行 */
  isRunning: boolean;
  /** 当前阶段/节点 */
  currentPhase: string;
  /** 运行ID */
  runId: string | null;
  /** 会话ID */
  sessionId: string | null;
  /** 错误信息 */
  error: string | null;
  /** 最后一个事件 */
  lastEvent: SSEEvent | null;
  /** 启动Agent运行 */
  runAgent: (objective: string, files?: string[]) => Promise<string>;
  /** 取消运行 */
  abort: () => void;
  /** 断开连接 */
  disconnect: () => void;
  /** 手动重连 */
  reconnect: () => void;
}

interface AgentRunResponse {
  run_id: string;
  session_id: string;
  status: string;
}

/**
 * Agent流式处理Hook
 *
 * @example
 * ```tsx
 * const { runAgent, isRunning, currentPhase } = useAgentStream({
 *   onNodeEnter: (node) => console.log('Entering:', node),
 *   onComplete: (msg) => console.log('Done:', msg),
 * });
 *
 * await runAgent('分析这个数据集', ['file_123']);
 * ```
 */
export function useAgentStream(options: AgentStreamOptions = {}): AgentStreamReturn {
  const {
    runId: initialRunId,
    sessionId: initialSessionId = 'new',
    objective: initialObjective,
    files: initialFiles = [],
    baseUrl = '/api',
    useJobs = true,
    onNodeEnter,
    onToolCall,
    onToolResult,
    onProgress,
    onPlanUpdate,
    onReport,
    onComplete,
    onError,
    onStart,
    onToken,
    streamMode = 'messages',
  } = options;

  const [runId, setRunId] = useState<string | null>(initialRunId || null);
  const [sessionId, setSessionId] = useState<string>(initialSessionId);
  const [currentPhase, setCurrentPhase] = useState<string>('idle');
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);

  // 构建SSE URL
  const sseUrl = runId
    ? useJobs
      ? `${baseUrl}/jobs/${runId}/events`
      : `${baseUrl}/agent/stream/${runId}`
    : '';

  // 使用useSSE处理SSE连接
  const { status, connected, lastEvent, disconnect: disconnectSSE, reconnect } = useSSE({
    url: sseUrl,
    eventTypes: ['start', 'node_enter', 'tool_call', 'tool_result', 'plan_update', 'progress', 'report_generated', 'end', 'error', 'heartbeat', 'token'],
    reconnect: false, // Agent运行不自动重连
    heartbeatInterval: 0,
    onMessage: (event: SSEEvent) => handleAgentEvent(event),
    onError: (err) => {
      const errorMsg = err.message || 'Connection error';
      setError(errorMsg);
      setIsRunning(false);
      onError?.(errorMsg);
    },
    onStatusChange: (newStatus) => {
      if (newStatus === 'disconnected' || newStatus === 'error') {
        setIsRunning(false);
      }
    },
  });

  // 处理Agent事件
  const handleAgentEvent = useCallback((event: SSEEvent) => {
    const eventType = event.type || event.event_type;
    const payload = event.payload || event.data || {};

    switch (eventType) {
      case 'start':
        setSessionId(payload.session_id || payload.thread_id || sessionId);
        setCurrentPhase('开始分析');
        break;

      case 'node_enter':
        const nodeName = event.node || payload.next_step || payload.node || 'unknown';
        setCurrentPhase(nodeName);
        onNodeEnter?.(nodeName, payload.message);
        break;

      case 'tool_call':
        setCurrentPhase('工具调用');
        const tools = payload.tools || payload.tool_calls || [payload.tool];
        if (Array.isArray(tools)) {
          tools.forEach((tool: any) => {
            onToolCall?.(tool.name || tool.tool, tool.args);
          });
        } else {
          onToolCall?.(tools, payload.args);
        }
        break;

      case 'tool_result':
        const resultTool = payload.tool || payload.tool_name;
        const messageContent = payload.message?.content || payload.content;

        // 解析工具结果，提取plot数据
        if (messageContent && typeof messageContent === 'string') {
          try {
            const resultData = JSON.parse(messageContent);
            // 检查各种可能的plot字段
            const plotData = resultData.data?.plots || resultData.data?.plot ||
                            resultData.data?.heatmap_plot || resultData.data?.annotated_umap_plot;
            if (plotData) {
              // 触发回调，传递完整的结果数据（包含plots）
              onToolResult?.(resultTool || 'unknown', resultData);
              return; // 已处理，不再调用默认回调
            }
          } catch (e) {
            // JSON解析失败，使用原始逻辑
          }
        }

        // 检查result中是否直接包含plots
        if (payload.result && typeof payload.result === 'object') {
          const plotData = payload.result.data?.plots || payload.result.data?.plot ||
                          payload.result.data?.heatmap_plot || payload.result.data?.annotated_umap_plot;
          if (plotData) {
            onToolResult?.(resultTool || 'unknown', payload.result);
            return;
          }
        }

        if (resultTool) {
          onToolResult?.(resultTool, payload.result);
        }
        break;

      case 'plan_update':
        setCurrentPhase('计划生成');
        const plan = payload.plan || [];
        if (Array.isArray(plan)) {
          onPlanUpdate?.(plan);
        }
        break;

      case 'progress':
        const progress = payload.progress !== undefined ? payload.progress : payload.value;
        if (typeof progress === 'number') {
          onProgress?.(Math.round(progress * 100), payload.message);
        }
        break;

      case 'report_generated':
        setCurrentPhase('报告生成');
        const reportData = payload.report || payload;
        if (reportData && onReport) {
          onReport({
            id: reportData.id || `report_${Date.now()}`,
            runId: runId || '',
            title: reportData.title || '分析报告',
            content: reportData.content || reportData.markdown || '',
            summary: reportData.summary,
            sections: reportData.sections,
            createdAt: reportData.createdAt ? new Date(reportData.createdAt) : new Date(),
            metadata: reportData.metadata,
          });
        }
        break;

      case 'end':
        setCurrentPhase('完成');
        setIsRunning(false);
        const response = payload.response || {};
        const responseContent = response.content || payload.message || '';
        onComplete?.(responseContent);
        disconnectSSE();
        break;

      case 'error':
        setCurrentPhase('错误');
        setIsRunning(false);
        const errorMsg = payload.error || payload.detail || payload.message || 'Unknown error';
        setError(errorMsg);
        onError?.(errorMsg);
        disconnectSSE();
        break;

      case 'token':
        if (payload.token) {
          onToken?.(String(payload.token), Boolean(payload.is_complete));
        }
        break;

      case 'heartbeat':
        // 心跳事件，不做处理
        break;

      default:
        console.log('[useAgentStream] Unhandled event type:', eventType, event);
    }
  }, [sessionId, runId, onNodeEnter, onToolCall, onToolResult, onPlanUpdate, onProgress, onReport, onComplete, onError, onToken, disconnectSSE]);

  // 启动Agent运行
  const runAgent = useCallback(async (
    objective: string,
    files: string[] = []
  ): Promise<string> => {
    // 重置状态
    setError(null);
    setIsRunning(true);
    setCurrentPhase('初始化');

    try {
      // 调用API启动Agent
      const response = useJobs
        ? await fetch(`${baseUrl}/jobs`, {
          method: 'POST',
          body: (() => {
            const formData = new FormData();
            formData.append('objective', objective);
            formData.append('thread_id', sessionId);
            formData.append('stream_mode', streamMode);
            if (files.length > 0) {
              formData.append('file_id', files[0]);
            }
            return formData;
          })(),
        })
        : await fetch(`${baseUrl}/agent/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            objective,
            files: files.length > 0 ? files : undefined,
            session_id: sessionId,
            thread_id: sessionId,
            stream_mode: streamMode,
          }),
        });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || 'Failed to start agent');
      }

      const data = await response.json();
      const newRunId = useJobs ? data.job_id : data.run_id;
      const newSessionId = useJobs ? data.thread_id : data.session_id;

      setRunId(newRunId);
      setSessionId(newSessionId);

      // 通知开始
      onStart?.(newRunId, newSessionId);

      return newRunId;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to start agent';
      setError(errorMsg);
      setIsRunning(false);
      setCurrentPhase('错误');
      onError?.(errorMsg);
      throw err;
    }
  }, [baseUrl, sessionId, onStart, onError, streamMode, useJobs]);

  // 断开连接
  const disconnect = useCallback(() => {
    disconnectSSE();
    setIsRunning(false);
    setCurrentPhase('idle');
  }, [disconnectSSE]);

  // 取消运行
  const abort = useCallback(async () => {
    if (runId) {
      try {
        await fetch(`${baseUrl}/agent/${runId}/abort`, {
          method: 'POST',
        });
      } catch (err) {
        console.error('[useAgentStream] Abort failed:', err);
      }
    }
    disconnect();
  }, [runId, baseUrl, disconnect]);

  return {
    status,
    isRunning: isRunning || connected,
    currentPhase,
    runId,
    sessionId,
    error,
    lastEvent,
    runAgent,
    abort,
    disconnect,
    reconnect,
  };
}

/**
 * 简化版Agent Hook - 自动管理状态
 */
export function useAgent(options: Omit<AgentStreamOptions, 'onNodeEnter' | 'onProgress'> = {}) {
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [progress, setProgress] = useState(0);

  return {
    ...useAgentStream({
      ...options,
      onNodeEnter: (node) => {
        setSteps(prev => [...prev, {
          id: `step_${node}_${Date.now()}`,
          role: node,
          task: `执行: ${node}`,
          status: 'running' as StepStatus,
          startedAt: new Date(),
        }]);
        options.onNodeEnter?.(node);
      },
      onProgress: (value) => {
        setProgress(value);
        options.onProgress?.(value);
      },
    }),
    steps,
    progress,
  };
}
