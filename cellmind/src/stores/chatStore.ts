import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Message, UploadedFile } from '@/types';
import api from '@/services/api';

interface ChatState {
  // 状态
  currentSessionId: string;
  messages: Message[];
  isAgentMode: boolean;
  uploadedFile: UploadedFile | null;
  isProcessing: boolean;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  setCurrentSession: (sessionId: string) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  updateMessage: (messageId: string, updater: (message: Message) => Message) => void;
  upsertMessage: (message: Message) => void;
  toggleAgentMode: () => void;
  setAgentMode: (mode: boolean) => void;
  setUploadedFile: (file: UploadedFile | null) => void;
  removeFile: () => void;
  setProcessing: (processing: boolean) => void;
  clearMessages: () => void;

  // 新增：消息持久化
  saveMessage: (message: Omit<Message, 'id' | 'timestamp'>) => Promise<string>;
  persistMessage: (message: Omit<Message, 'id' | 'timestamp'>) => Promise<void>;
  loadSessionMessages: (sessionId: string) => Promise<void>;
}

// 辅助函数：生成消息ID
const generateMessageId = () => `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // 初始状态
      currentSessionId: 'new',
      messages: [],
      isAgentMode: false,
      uploadedFile: null,
      isProcessing: false,

      // Actions
      sendMessage: async (content: string) => {
        const state = get();
        const sessionId = state.currentSessionId;

        // 1. 添加并保存用户消息
        const userMessageId = await get().saveMessage({
          sessionId,
          role: 'user',
          content,
        });

        try {
          // 调用聊天 API
          const response = await fetch('http://localhost:8000/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: content, thread_id: sessionId }),
          });

          if (!response.ok) throw new Error('Chat request failed');

          const data = await response.json();

          // 2. 添加并保存助手回复
          await get().saveMessage({
            sessionId,
            role: 'assistant',
            content: data.message || '抱歉，我无法回答这个问题。',
          });
        } catch (error) {
          console.error('Chat error:', error);
          // 保存错误消息
          await get().saveMessage({
            sessionId,
            role: 'assistant',
            content: '抱歉，发生了错误，请稍后重试。',
          });
        }
      },

      setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),

      setMessages: (messages) => set({ messages }),

      addMessage: (message) =>
        set((state) => ({ messages: [...state.messages, message] })),

      updateMessage: (messageId, updater) =>
        set((state) => ({
          messages: state.messages.map((message) =>
            message.id === messageId ? updater(message) : message
          ),
        })),

      upsertMessage: (message) =>
        set((state) => {
          const exists = state.messages.some((item) => item.id === message.id);
          if (exists) {
            return {
              messages: state.messages.map((item) =>
                item.id === message.id ? message : item
              ),
            };
          }
          return { messages: [...state.messages, message] };
        }),

      toggleAgentMode: () =>
        set((state) => ({ isAgentMode: !state.isAgentMode })),

      setAgentMode: (mode) => set({ isAgentMode: mode }),

      setUploadedFile: (file) => set({ uploadedFile: file }),

      removeFile: () => set({ uploadedFile: null }),

      setProcessing: (processing) => set({ isProcessing: processing }),

      clearMessages: () => set({ messages: [] }),

      // ========== 新增：消息持久化 ==========

      /**
       * 保存消息到后端数据库
       * @returns 消息ID
       */
      saveMessage: async (message) => {
        const messageId = generateMessageId();
        const timestamp = new Date();

        const newMessage: Message = {
          ...message,
          id: messageId,
          timestamp,
        };

        // 1. 先添加到本地状态（乐观更新）
        set((state) => ({ messages: [...state.messages, newMessage] }));

        // 2. 异步保存到后端
        try {
          await api.post(`/sessions/${message.sessionId}/messages`, {
            role: message.role,
            content: message.content,
            metadata: message.metadata || {},
          });
          console.log('[ChatStore] Message saved to backend:', messageId);
        } catch (error) {
          console.error('[ChatStore] Failed to save message to backend:', error);
          // 可以在这里标记消息为"未保存"状态，供UI显示
        }

        return messageId;
      },

      /**
       * 仅持久化消息到后端（不改变本地状态）
       */
      persistMessage: async (message) => {
        try {
          await api.post(`/sessions/${message.sessionId}/messages`, {
            role: message.role,
            content: message.content,
            metadata: message.metadata || {},
          });
        } catch (error) {
          console.error('[ChatStore] Failed to persist message to backend:', error);
        }
      },

      /**
       * 从后端加载会话消息
       */
      loadSessionMessages: async (sessionId: string) => {
        try {
          const response: { messages: any[] } = await api.get(`/sessions/${sessionId}/messages`, {
            params: { limit: 100 },
          });

          const messages: Message[] = response.messages.map(msg => ({
            id: msg.id,
            sessionId: sessionId,
            role: msg.role,
            content: msg.content,
            timestamp: new Date(msg.timestamp),
            metadata: msg.metadata,
          }));

          set({ messages });
          console.log(`[ChatStore] Loaded ${messages.length} messages for session:`, sessionId);
        } catch (error) {
          console.error('[ChatStore] Failed to load messages:', error);
          throw error;
        }
      },
    }),
    {
      name: 'cellmind-chat',
      partialize: (state) => ({
        // 只持久化 sessionId，不持久化 isAgentMode
        // Agent 模式应该在每次页面加载时重置为关闭状态
        currentSessionId: state.currentSessionId,
      }),
    }
  )
);
