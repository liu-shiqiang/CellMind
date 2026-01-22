import api from './api';
import type { SSEEvent, Message, UploadedFile } from '@/types';

export interface AgentRunRequest {
  objective: string;
  files?: string[];
  session_id?: string;
  stream_mode?: 'updates' | 'messages' | 'debug';
}

export interface AgentRunResponse {
  run_id: string;
  session_id: string;
  status: string;
}

export interface AgentCallbacks {
  onEvent?: (event: SSEEvent) => void;
  onStart?: (runId: string) => void;
  onComplete?: (message: Message) => void;
  onReport?: (report: any) => void;
  onError?: (error: Error) => void;
}

export const agentService = {
  /**
   * 启动Agent分析
   */
  async runAgent(
    objective: string,
    file: UploadedFile | null,
    sessionId: string,
    callbacks: AgentCallbacks = {}
  ): Promise<void> {
    try {
      // 1. 创建运行
      const request: AgentRunRequest = {
        objective,
        session_id: sessionId || undefined,
        files: file ? [file.id] : [],
      };

      const response: AgentRunResponse = await api.post('/agent/run', request);
      const runId = response.run_id;

      callbacks.onStart?.(runId);

      // 2. 建立SSE连接
      this.streamAgent(runId, callbacks);

    } catch (error) {
      callbacks.onError?.(error as Error);
    }
  },

  /**
   * 流式监听Agent执行
   */
  streamAgent(runId: string, callbacks: AgentCallbacks = {}): () => void {
    const url = `${api.defaults.baseURL}/agent/stream/${runId}`;
    const eventSource = new EventSource(url);

    let finalMessage = '';
    let sessionId = '';
    let hasReceivedData = false;
    let streamClosed = false;

    // 安全的关闭函数
    const closeStream = () => {
      if (!streamClosed) {
        streamClosed = true;
        eventSource.close();
      }
    };

    // 发送完成消息的辅助函数
    const sendComplete = () => {
      if (finalMessage) {
        callbacks.onComplete?.({
          id: `msg_${Date.now()}`,
          sessionId,
          role: 'assistant',
          content: finalMessage,
          timestamp: new Date(),
        });
      }
    };

    eventSource.onmessage = (e) => {
      // 跳过空消息
      if (!e.data || e.data === '') {
        return;
      }

      // 处理 [DONE] 标记
      if (e.data === '[DONE]') {
        closeStream();
        sendComplete();
        return;
      }

      hasReceivedData = true;

      try {
        const event: SSEEvent = JSON.parse(e.data);

        // 防御性检查：确保解析后的 event 对象有效
        if (!event) {
          console.warn('Received empty SSE event');
          return;
        }

        callbacks.onEvent?.(event);

        // 获取数据源：优先使用 payload，兼容 data 字段
        const eventData = event.payload || event.data || {};
        const eventType = event.type || event.event_type;

        // 处理不同事件类型
        switch (eventType) {
          case 'start':
            // 兼容多种字段名：sessionId / thread_id / session_id
            sessionId = (eventData as any).sessionId ||
                        (eventData as any).thread_id ||
                        (eventData as any).session_id ||
                        event.thread_id ||
                        event.sessionId ||
                        '';
            break;

          case 'report_generated':
            // 处理报告生成事件
            const reportData = (eventData as any).report || eventData;
            if (reportData && callbacks.onReport) {
              callbacks.onReport(reportData);
            }
            break;

          case 'analysis_complete':
          case 'end':
            // 从 payload.response.content 或 data.message 获取最终消息
            const responseContent = (eventData as any).response?.content ||
                                    (eventData as any).message ||
                                    '';
            if (responseContent) {
              finalMessage = responseContent;
            }
            // 正常结束时也发送完成
            closeStream();
            sendComplete();
            break;

          case 'error':
            const errorMsg = (eventData as any).error ||
                            (eventData as any).message ||
                            (eventData as any).detail ||
                            'Unknown error';
            callbacks.onError?.(new Error(errorMsg));
            break;

          case 'token':
            // 处理流式 token
            if ((eventData as any).response?.content) {
              finalMessage += (eventData as any).response.content;
            }
            break;
        }
      } catch (err) {
        console.error('Parse SSE error:', err, 'Raw data:', e.data);
      }
    };

    eventSource.onerror = (err) => {
      console.log('SSE connection closed (this is normal after stream ends)');
      closeStream();

      // 如果已经收到数据，则认为流正常结束，发送完成消息
      if (hasReceivedData) {
        sendComplete();
      } else {
        // 如果没有收到任何数据就出错了，才报告错误
        callbacks.onError?.(new Error('Agent stream connection error'));
      }
    };

    // 返回清理函数
    return () => eventSource.close();
  },

  /**
   * 获取运行状态
   */
  async getRunStatus(runId: string) {
    return api.get(`/agent/runs/${runId}`);
  },
};
