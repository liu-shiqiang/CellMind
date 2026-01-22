/**
 * 认证服务 API
 * 处理用户登录、注册、登出等认证相关 API 调用
 */
import api from './api';
import type { LoginRequest, RegisterRequest, AuthResponse, User } from '@/types';

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: 'bearer';
  expires_in: number;
}

export const authService = {
  /**
   * 用户登录
   */
  async login(data: LoginRequest): Promise<AuthResponse> {
    const response: AuthResponse = await api.post('/auth/login', data);
    return response;
  },

  /**
   * 用户注册
   */
  async register(data: RegisterRequest): Promise<AuthResponse> {
    const response: AuthResponse = await api.post('/auth/register', data);
    return response;
  },

  /**
   * 匿名登录
   */
  async guestLogin(): Promise<AuthResponse> {
    const response: AuthResponse = await api.post('/auth/guest');
    return response;
  },

  /**
   * 用户登出
   */
  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },

  /**
   * 获取当前用户信息
   */
  async getCurrentUser(): Promise<User> {
    const response: User = await api.get('/auth/me');
    return response;
  },

  /**
   * 验证 Token 有效性
   */
  async verifyToken(): Promise<User> {
    const response: User = await api.post('/auth/verify');
    return response;
  },

  /**
   * 刷新 Token
   */
  async refreshToken(refreshToken: string): Promise<AuthTokens> {
    const response: AuthTokens = await api.post('/auth/refresh', { refresh_token: refreshToken });
    return response;
  },
};
