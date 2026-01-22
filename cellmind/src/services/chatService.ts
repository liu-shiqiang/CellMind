import api from './api';
import type { Message } from '@/types';

export interface ChatRequest {
  message: string;
  session_id?: string;
}

export interface ChatResponse {
  message_id: string;
  session_id: string;
  role: string;
  content: string;
  timestamp: string;
}

export const chatService = {
  /**
   * 发送消息 (非流式)
   */
  async sendMessage(content: string, sessionId: string): Promise<Message> {
    const response: ChatResponse = await api.post('/chat/send', {
      message: content,
      session_id: sessionId || undefined,
    });

    return {
      id: response.message_id,
      sessionId: response.session_id,
      role: response.role as Message['role'],
      content: response.content,
      timestamp: new Date(response.timestamp),
    };
  },

  /**
   * 流式聊天
   * @param onChunk 接收到文本片段时的回调
   * @param onComplete 完成时的回调
   * @param onError 错误时的回调
   * @returns 清理函数
   */
  streamChat(
    content: string,
    sessionId: string,
    onChunk: (chunk: string) => void,
    onComplete: (fullMessage: string) => void,
    onError: (error: Error) => void
  ): () => void {
    const url = `${api.defaults.baseURL}/chat/stream`;
    const params = new URLSearchParams({
      message: content,
      session_id: sessionId || 'new',
    });

    const eventSource = new EventSource(`${url}?${params}`);

    let fullResponse = '';

    eventSource.onmessage = (e) => {
      if (e.data === '[DONE]') {
        eventSource.close();
        onComplete(fullResponse);
        return;
      }

      try {
        const data = JSON.parse(e.data);
        if (data.type === 'token' && data.content) {
          fullResponse += data.content;
          onChunk(data.content);
        }
      } catch (err) {
        console.error('Parse SSE error:', err);
      }
    };

    eventSource.onerror = (err) => {
      eventSource.close();
      onError(new Error('Stream connection error'));
    };

    // 返回清理函数
    return () => eventSource.close();
  },
};
