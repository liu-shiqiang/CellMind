import api from './api';
import type { Session } from '@/types';

export interface SessionCreateRequest {
  title?: string;
}

export interface SessionListResponse {
  sessions: Session[];
  total: number;
}

export const sessionService = {
  /**
   * 获取会话列表
   */
  async getSessions(limit = 20): Promise<Session[]> {
    const response: SessionListResponse = await api.get('/sessions', {
      params: { limit },
    });
    return response.sessions;
  },

  /**
   * 获取会话详情
   */
  async getSession(sessionId: string): Promise<Session> {
    return api.get(`/sessions/${sessionId}`);
  },

  /**
   * 创建会话
   */
  async createSession(title?: string): Promise<Session> {
    const request: SessionCreateRequest = title ? { title } : {};
    return api.post('/sessions', request);
  },

  /**
   * 更新会话
   */
  async updateSession(
    sessionId: string,
    updates: { title?: string }
  ): Promise<Session> {
    return api.put(`/sessions/${sessionId}`, updates);
  },

  /**
   * 删除会话
   */
  async deleteSession(sessionId: string): Promise<void> {
    return api.delete(`/sessions/${sessionId}`);
  },

  /**
   * 获取会话消息
   */
  async getSessionMessages(sessionId: string, limit = 100) {
    return api.get(`/sessions/${sessionId}/messages`, {
      params: { limit },
    });
  },
};
