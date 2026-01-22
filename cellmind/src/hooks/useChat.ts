/**
 * useChat - 聊天Hook
 * 管理对话模式的交互
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Message } from '@/types';
import { useSSE, type SSEEvent } from './useSSE';

export interface ChatOptions {
  /** API基础URL */
  baseUrl?: string;
  /** 会话ID */
  sessionId?: string;
  /** 发送消息前的回调 */
  onBeforeSend?: (content: string) => void;
  /** 收到回复后的回调 */
  onMessageReceived?: (message: Message) => void;
  /** 流式token回调 */
  onToken?: (token: string) => void;
  /** 流式完成回调 */
  onComplete?: (message: string) => void;
  /** 发生错误的回调 */
  onError?: (error: string) => void;
  /** 启用流式回复 */
  stream?: boolean;
}

export interface ChatReturn {
  /** 是否正在发送 */
  isSending: boolean;
  /** 发送消息 */
  sendMessage: (content: string) => Promise<void>;
  /** 重试最后一条消息 */
  retryLastMessage: () => Promise<void>;
  /** 清空消息 */
  clearMessages: () => void;
}

/**
 * 聊天Hook - 简化版
 *
 * @example
 * ```tsx
 * const { sendMessage, isSending } = useChat({
 *   sessionId: 'session_123',
 *   onMessageReceived: (msg) => addMessage(msg),
 * });
 *
 * await sendMessage('什么是scRNA-seq?');
 * ```
 */
export function useChat(options: ChatOptions = {}): ChatReturn {
  const {
    baseUrl = '/api',
    sessionId = 'default',
    onBeforeSend,
    onMessageReceived,
    onToken,
    onComplete,
    onError,
    stream = false,
  } = options;

  const [isSending, setIsSending] = useState(false);
  const [streamUrl, setStreamUrl] = useState('');
  const accumulatedRef = useRef('');
  const pendingPromiseRef = useRef<{ resolve: () => void } | null>(null);

  const { disconnect } = useSSE({
    url: stream ? streamUrl : '',
    eventTypes: ['token', 'end', 'error'],
    reconnect: false,
    heartbeatInterval: 0,
    onMessage: (event: SSEEvent) => {
      const eventType = event.type || event.event_type;
      const payload = event.payload || event.data || {};
      if (eventType === 'token') {
        const token = String(payload.token || payload.content || '');
        if (!token) {
          return;
        }
        accumulatedRef.current += token;
        onToken?.(token);
      } else if (eventType === 'end') {
        const finalMessage = String(payload.message || accumulatedRef.current || '');
        onComplete?.(finalMessage);
        setIsSending(false);
        setStreamUrl('');
        disconnect();
        accumulatedRef.current = '';
        pendingPromiseRef.current?.resolve();
        pendingPromiseRef.current = null;
      } else if (eventType === 'error') {
        const errorMsg = String(payload.message || payload.detail || 'Stream error');
        onError?.(errorMsg);
        setIsSending(false);
        setStreamUrl('');
        disconnect();
        accumulatedRef.current = '';
        pendingPromiseRef.current?.resolve();
        pendingPromiseRef.current = null;
      }
    },
    onError: (err) => {
      const errorMsg = err.message || 'Stream error';
      onError?.(errorMsg);
      setIsSending(false);
      setStreamUrl('');
      accumulatedRef.current = '';
      pendingPromiseRef.current?.resolve();
      pendingPromiseRef.current = null;
    },
  });

  useEffect(() => {
    if (!stream) {
      accumulatedRef.current = '';
      setStreamUrl('');
    }
  }, [stream]);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isSending) {
      return;
    }

    setIsSending(true);
    onBeforeSend?.(content);

    try {
      if (stream) {
        accumulatedRef.current = '';
        const encodedMessage = encodeURIComponent(content);
        const encodedThread = encodeURIComponent(sessionId);
        setStreamUrl(`${baseUrl}/chat/stream?message=${encodedMessage}&thread_id=${encodedThread}`);
        return await new Promise<void>((resolve) => {
          pendingPromiseRef.current = { resolve };
        });
      }

      const response = await fetch(`${baseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          thread_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // 构造助手消息
      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant`,
        sessionId,
        role: 'assistant',
        content: data.message || '抱歉，我无法回答这个问题。',
        timestamp: new Date(),
      };

      onMessageReceived?.(assistantMessage);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to send message';
      onError?.(errorMsg);
    } finally {
      if (!stream) {
        setIsSending(false);
      }
    }
  }, [baseUrl, sessionId, isSending, onBeforeSend, onMessageReceived, onError, stream]);

  const retryLastMessage = useCallback(async () => {
    // TODO: 实现重试逻辑 - 需要记录最后一条消息
    console.warn('retryLastMessage not yet implemented');
  }, []);

  const clearMessages = useCallback(() => {
    // TODO: 清空消息 - 需要从上层传入
    console.warn('clearMessages not yet implemented');
  }, []);

  return {
    isSending,
    sendMessage,
    retryLastMessage,
    clearMessages,
  };
}

/**
 * 增强版聊天Hook - 带消息管理
 */
export function useChatWithMessages(options: ChatOptions = {}) {
  const {
    baseUrl = '/api',
    sessionId = 'default',
    onBeforeSend,
    onMessageReceived,
    onError,
  } = options;

  const [messages, setMessages] = useState<Message[]>([]);
  const [isSending, setIsSending] = useState(false);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isSending) {
      return;
    }

    setIsSending(true);

    // 添加用户消息
    const userMessage: Message = {
      id: `msg_${Date.now()}_user`,
      sessionId,
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);

    onBeforeSend?.(content);

    try {
      const response = await fetch(`${baseUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: content,
          thread_id: sessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // 添加助手消息
      const assistantMessage: Message = {
        id: `msg_${Date.now()}_assistant`,
        sessionId,
        role: 'assistant',
        content: data.message || '抱歉，我无法回答这个问题。',
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
      onMessageReceived?.(assistantMessage);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to send message';

      // 添加错误消息
      const errorMessage: Message = {
        id: `msg_${Date.now()}_error`,
        sessionId,
        role: 'assistant',
        content: `错误: ${errorMsg}`,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);

      onError?.(errorMsg);
    } finally {
      setIsSending(false);
    }
  }, [baseUrl, sessionId, isSending, onBeforeSend, onMessageReceived, onError]);

  const clearMessages = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    isSending,
    sendMessage,
    clearMessages,
  };
}
