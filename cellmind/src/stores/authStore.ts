import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { User, AuthTokens, LoginRequest, RegisterRequest } from '@/types';

interface AuthState {
  // 状态
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isAnonymous: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  setAuth: (tokens: AuthTokens, user: User) => void;
  setUser: (user: User) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearError: () => void;
  logout: () => void;

  // 便捷方法
  getAuthHeader: () => { Authorization: string } | null;
}

const TOKEN_REFRESH_THRESHOLD = 5 * 60 * 1000; // 5分钟（毫秒）
const ANON_EMAIL_DOMAIN = '@anonymous.local';
const ANON_USERNAME_PREFIX = 'anon_';

export function isAnonymousUser(user: User | null): boolean {
  if (!user) return false;
  if (user.is_anonymous) return true;
  return user.username.startsWith(ANON_USERNAME_PREFIX) || user.email.endsWith(ANON_EMAIL_DOMAIN);
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // 初始状态
      user: null,
      token: null,
      isAuthenticated: false,
      isAnonymous: false,
      isLoading: false,
      error: null,

      // 设置认证信息
      setAuth: (tokens: AuthTokens, user: User) => {
        set({
          token: tokens.access_token,
          user,
          isAuthenticated: true,
          isAnonymous: isAnonymousUser(user),
          error: null,
        });

        // 存储 refresh token 到 localStorage（单独处理）
        localStorage.setItem('refresh_token', tokens.refresh_token);

        // 设置 token 过期提醒
        const expiresAt = Date.now() + tokens.expires_in * 1000;
        localStorage.setItem('token_expires_at', expiresAt.toString());
      },

      // 更新用户信息
      setUser: (user: User) => set({ user, isAnonymous: isAnonymousUser(user) }),

      // 设置加载状态
      setLoading: (isLoading: boolean) => set({ isLoading }),

      // 设置错误信息
      setError: (error: string | null) => set({ error }),

      // 清除错误信息
      clearError: () => set({ error: null }),

      // 登出
      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          isAnonymous: false,
          error: null,
        });
        localStorage.removeItem('refresh_token');
        localStorage.removeItem('token_expires_at');
      },

      // 获取认证头
      getAuthHeader: () => {
        const { token } = get();
        if (!token) return null;
        return { Authorization: `Bearer ${token}` };
      },
    }),
    {
      name: 'cellmind-auth',
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
        isAnonymous: state.isAnonymous,
      }),
    }
  )
);

// 辅助函数：检查 token 是否即将过期
export function isTokenExpiringSoon(): boolean {
  const expiresAt = localStorage.getItem('token_expires_at');
  if (!expiresAt) return true;
  return Date.now() >= parseInt(expiresAt) - TOKEN_REFRESH_THRESHOLD;
}

// 辅助函数：获取 refresh token
export function getRefreshToken(): string | null {
  return localStorage.getItem('refresh_token');
}

// 辅助函数：获取用户显示名称
export function getUserDisplayName(): string {
  const { user } = useAuthStore.getState();
  if (!user) return '未登录';
  if (isAnonymousUser(user)) return '匿名用户';
  return user.full_name || user.username;
}
