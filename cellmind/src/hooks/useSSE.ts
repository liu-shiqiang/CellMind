/**
 * useSSE - Server-Sent Events通用Hook
 * 提供SSE连接管理、自动重连、事件过滤等功能
 */
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

export interface SSEEvent {
  type: string;
  event_type?: string;  // 兼容字段
  run_id?: string;
  thread_id?: string;
  session_id?: string;
  timestamp?: string;
  payload?: any;
  data?: any;          // 兼容字段
  message?: string;
  [key: string]: any;
}

export interface SSEOptions {
  /** SSE端点URL */
  url: string;
  /** 过滤的事件类型，不传则接收所有事件 */
  eventTypes?: string[];
  /** 是否自动重连 */
  reconnect?: boolean;
  /** 重连间隔(ms)，默认3000ms */
  reconnectInterval?: number;
  /** 最大重连次数，默认5次 */
  maxReconnectAttempts?: number;
  /** 心跳间隔(ms)，默认30000ms，传0或null禁用 */
  heartbeatInterval?: number | null;
  /** 收到消息时的回调 */
  onMessage?: (event: SSEEvent) => void;
  /** 发生错误时的回调 */
  onError?: (error: Error) => void;
  /** 连接建立时的回调 */
  onOpen?: () => void;
  /** 连接关闭时的回调 */
  onClose?: () => void;
  /** 连接状态变化回调 */
  onStatusChange?: (status: SSEStatus) => void;
}

export type SSEStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error';

export interface SSEReturn {
  /** 当前连接状态 */
  status: SSEStatus;
  /** 是否已连接 */
  connected: boolean;
  /** 最后一个错误 */
  error: Error | null;
  /** 最后接收到的事件 */
  lastEvent: SSEEvent | null;
  /** 事件历史记录 */
  eventHistory: SSEEvent[];
  /** 最大历史记录数量 */
  maxHistorySize: number;
  /** 建立连接 */
  connect: () => void;
  /** 断开连接 */
  disconnect: () => void;
  /** 手动重连 */
  reconnect: () => void;
  /** 清空历史记录 */
  clearHistory: () => void;
  /** 获取特定类型的事件历史 */
  getEventsByType: (type: string) => SSEEvent[];
}

const DEFAULT_RECONNECT_INTERVAL = 3000;
const DEFAULT_MAX_RECONNECT = 5;
const DEFAULT_HEARTBEAT_INTERVAL = 30000;
const DEFAULT_MAX_HISTORY = 100;

/**
 * SSE通用Hook
 */
export function useSSE(options: SSEOptions): SSEReturn {
  const {
    url,
    eventTypes,
    reconnect = true,
    reconnectInterval = DEFAULT_RECONNECT_INTERVAL,
    maxReconnectAttempts = DEFAULT_MAX_RECONNECT,
    heartbeatInterval = DEFAULT_HEARTBEAT_INTERVAL,
    onMessage,
    onError,
    onOpen,
    onClose,
    onStatusChange,
  } = options;

  const [status, setStatus] = useState<SSEStatus>('idle');
  const [error, setError] = useState<Error | null>(null);
  const [lastEvent, setLastEvent] = useState<SSEEvent | null>(null);
  const [eventHistory, setEventHistory] = useState<SSEEvent[]>([]);

  // 使用ref追踪连接状态
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const heartbeatTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const maxHistorySizeRef = useRef(DEFAULT_MAX_HISTORY);
  const isManualDisconnectRef = useRef(false);

  // 更新状态并通知外部
  const updateStatus = useCallback((newStatus: SSEStatus) => {
    setStatus(newStatus);
    onStatusChange?.(newStatus);
  }, [onStatusChange]);

  // 添加事件到历史记录
  const addEventToHistory = useCallback((event: SSEEvent) => {
    setEventHistory(prev => {
      const newHistory = [...prev, event];
      // 限制历史记录大小
      if (newHistory.length > maxHistorySizeRef.current) {
        return newHistory.slice(-maxHistorySizeRef.current);
      }
      return newHistory;
    });
  }, []);

  // 清理定时器
  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
      heartbeatTimeoutRef.current = null;
    }
  }, []);

  // 处理SSE消息
  const handleMessage = useCallback((event: MessageEvent) => {
    const data = event.data;

    // 跳过空消息
    if (!data || data === '') {
      return;
    }

    // 处理 [DONE] 标记
    if (data === '[DONE]') {
      updateStatus('disconnected');
      onClose?.();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      return;
    }

    try {
      const parsedEvent: SSEEvent = JSON.parse(data);

      // 防御性检查
      if (!parsedEvent) {
        console.warn('[useSSE] Received empty event');
        return;
      }

      // 事件类型过滤
      const eventType = parsedEvent.type || parsedEvent.event_type;
      if (eventTypes && eventType && !eventTypes.includes(eventType)) {
        return; // 跳过不在过滤器中的事件
      }

      // 更新最后事件
      setLastEvent(parsedEvent);

      // 添加到历史记录
      addEventToHistory(parsedEvent);

      // 重置心跳计时器
      resetHeartbeat();

      // 调用外部回调
      onMessage?.(parsedEvent);
    } catch (err) {
      console.error('[useSSE] Parse error:', err, 'Raw data:', data);
    }
  }, [eventTypes, onMessage, addEventToHistory, updateStatus, onClose]);

  // 尝试重连
  const attemptReconnect = useCallback(() => {
    if (isManualDisconnectRef.current) {
      return;
    }

    if (reconnect && reconnectAttemptsRef.current < maxReconnectAttempts!) {
      reconnectAttemptsRef.current++;
      console.log(`[useSSE] Reconnecting... Attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts}`);

      updateStatus('connecting');
      setError(null);

      reconnectTimeoutRef.current = setTimeout(() => {
        connect();
      }, reconnectInterval);
    } else {
      updateStatus('error');
      const err = new Error(`Max reconnect attempts (${maxReconnectAttempts}) reached`);
      setError(err);
      onError?.(err);
    }
  }, [reconnect, maxReconnectAttempts, reconnectInterval, updateStatus, onError]);

  // 重置心跳计时器
  const resetHeartbeat = useCallback(() => {
    if (!heartbeatInterval || heartbeatInterval <= 0) {
      if (heartbeatTimeoutRef.current) {
        clearTimeout(heartbeatTimeoutRef.current);
        heartbeatTimeoutRef.current = null;
      }
      return;
    }
    if (heartbeatTimeoutRef.current) {
      clearTimeout(heartbeatTimeoutRef.current);
    }
    heartbeatTimeoutRef.current = setTimeout(() => {
      // 心跳超时，触发重连
      if (eventSourceRef.current?.readyState === EventSource.OPEN) {
        console.warn('[useSSE] Heartbeat timeout, reconnecting...');
        eventSourceRef.current.close();
        attemptReconnect();
      }
    }, heartbeatInterval);
  }, [heartbeatInterval, attemptReconnect]);

  // 建立连接
  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      return; // 已存在连接
    }

    updateStatus('connecting');
    setError(null);
    isManualDisconnectRef.current = false;

    try {
      const eventSource = new EventSource(url);

      eventSource.onopen = () => {
        console.log('[useSSE] Connection opened');
        updateStatus('connected');
        reconnectAttemptsRef.current = 0; // 重置重连计数
        onOpen?.();
        resetHeartbeat();
      };

      eventSource.onmessage = handleMessage;

      eventSource.onerror = (err) => {
        console.log('[useSSE] Connection error or closed');
        clearTimers();

        if (eventSource.readyState === EventSource.CLOSED) {
          if (!isManualDisconnectRef.current) {
            if (reconnect) {
              attemptReconnect();
            } else {
              updateStatus('disconnected');
              onClose?.();
            }
          } else {
            updateStatus('disconnected');
            onClose?.();
          }
        } else {
          updateStatus('error');
          const error = new Error('EventSource error');
          setError(error);
          onError?.(error);
        }
      };

      eventSourceRef.current = eventSource;
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Failed to create EventSource');
      setError(error);
      updateStatus('error');
      onError?.(error);
    }
  }, [url, updateStatus, onOpen, handleMessage, attemptReconnect, clearTimers, resetHeartbeat, onError, onClose]);

  // 断开连接
  const disconnect = useCallback(() => {
    isManualDisconnectRef.current = true;
    clearTimers();

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    updateStatus('idle');
    reconnectAttemptsRef.current = 0;
  }, [clearTimers, updateStatus]);

  // 手动重连
  const manualReconnect = useCallback(() => {
    disconnect();
    isManualDisconnectRef.current = false;
    reconnectAttemptsRef.current = 0;
    connect();
  }, [disconnect, connect]);

  // 清空历史记录
  const clearHistory = useCallback(() => {
    setEventHistory([]);
  }, []);

  // 获取特定类型的事件
  const getEventsByType = useCallback((type: string): SSEEvent[] => {
    return eventHistory.filter(event => {
      const eventType = event.type || event.event_type;
      return eventType === type;
    });
  }, [eventHistory]);

  // 组件挂载时自动连接（如果url存在）
  useEffect(() => {
    if (url) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [url]); // 只在url变化时重新连接

  return {
    status,
    connected: status === 'connected',
    error,
    lastEvent,
    eventHistory,
    maxHistorySize: maxHistorySizeRef.current,
    connect,
    disconnect,
    reconnect: manualReconnect,
    clearHistory,
    getEventsByType,
  };
}
