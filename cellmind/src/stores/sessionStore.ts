import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Session } from '@/types';
import { sessionService } from '@/services';
import { useAuthStore } from './authStore';

interface SessionState {
  sessions: Session[];
  currentSession: Session | null;
  isInitialized: boolean;

  // Actions
  initSessions: () => Promise<void>;
  setSessions: (sessions: Session[]) => void;
  setCurrentSession: (session: Session | null) => void;
  addSession: (session: Session) => void;
  updateSession: (sessionId: string, updates: Partial<Session>) => void;
  removeSession: (sessionId: string) => void;
  createLocalSession: (title?: string) => Session;
  syncSessionsFromBackend: () => Promise<void>;
}

/**
 * 生成临时会话ID（本地未同步到后端的会话）
 */
const generateLocalSessionId = () => `local_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

/**
 * 判断会话是否为本地会话（未同步到后端）
 */
export const isLocalSession = (sessionId: string): boolean => {
  return sessionId.startsWith('local_');
};

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSession: null,
      isInitialized: false,

      /**
       * 初始化会话列表
       * - 未登录：不加载任何会话
       * - 已登录：从后端加载该用户的会话
       */
      initSessions: async () => {
        try {
          const authState = useAuthStore.getState();

          // 未登录状态下不加载会话
          if (!authState.isAuthenticated) {
            set({ sessions: [], currentSession: null, isInitialized: true });
            return;
          }

          // 已登录：从后端同步会话
          await get().syncSessionsFromBackend();
          set({ isInitialized: true });

          const state = get();
          // 如果没有会话，不自动创建（用户需要手动点击"新建分析"）
          if (state.sessions.length > 0 && !state.currentSession) {
            // 设置第一个会话为当前会话
            set({ currentSession: state.sessions[0] });
          }
        } catch (error) {
          console.error('[SessionStore] Failed to initialize sessions:', error);
          set({ isInitialized: true });
        }
      },

      /**
       * 从后端同步会话列表
       * 只在已登录状态下调用
       */
      syncSessionsFromBackend: async () => {
        try {
          const authState = useAuthStore.getState();

          // 未登录状态下清空会话列表
          if (!authState.isAuthenticated) {
            set({ sessions: [], currentSession: null });
            return;
          }

          const backendSessions = await sessionService.getSessions(50);
          set({ sessions: backendSessions });
          console.log(`[SessionStore] Synced ${backendSessions.length} sessions from backend`);
        } catch (error) {
          console.error('[SessionStore] Failed to sync sessions from backend:', error);
          // 未认证时清空会话列表
          if (error.response?.status === 401 || error.response?.status === 403) {
            set({ sessions: [], currentSession: null });
          }
          throw error;
        }
      },

      setSessions: (sessions) => set({ sessions }),

      setCurrentSession: (session) => set({ currentSession: session }),

      addSession: (session) =>
        set((state) => ({
          sessions: [session, ...state.sessions],
          currentSession: session,
        })),

      updateSession: (sessionId: string, updates: Partial<Session>) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, ...updates } : s
          ),
          currentSession:
            state.currentSession?.id === sessionId
              ? { ...state.currentSession, ...updates }
              : state.currentSession,
        })),

      removeSession: (sessionId) =>
        set((state) => ({
          sessions: state.sessions.filter((s) => s.id !== sessionId),
          currentSession:
            state.currentSession?.id === sessionId ? null : state.currentSession,
        })),

      /**
       * 创建本地临时会话（仅用于降级场景）
       * 注意：此会话不会自动同步到后端，调用者需要处理同步逻辑
       */
      createLocalSession: (title) => {
        const now = new Date().toISOString();
        const newSession: Session = {
          id: generateLocalSessionId(),
          title: title || `New Analysis ${new Date().toLocaleString()}`,
          created_at: now,
          updated_at: now,
          message_count: 0,
          agent_mode: false,
        };

        get().addSession(newSession);
        console.warn('[SessionStore] Created local-only session:', newSession.id);
        return newSession;
      },
    }),
    {
      name: 'cellmind-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSession: state.currentSession?.id,
      }),
    }
  )
);
