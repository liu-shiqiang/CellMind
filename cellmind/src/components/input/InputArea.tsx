/**
 * InputArea - 输入区域组件
 * 支持文件上传和Agent模式切换
 * 使用useAgentStream Hook重构，简化代码
 */
import React, { useState, useRef, useEffect } from 'react';
import { Send, Plus, FileUp, Zap, AlertCircle } from 'lucide-react';
import { useChatStore, useAgentStore, useAuthStore } from '@/stores';
import { uploadService } from '@/services';
import { FilePill } from './FilePill';
import { useAgentStream, useChat } from '@/hooks';
import type { UploadedFile, AnalysisReport, AgentStep } from '@/types';

interface InputAreaProps {
  onSendMessage?: (message: string) => void;
}

export const InputArea: React.FC<InputAreaProps> = ({ onSendMessage }) => {
  const [input, setInput] = useState('');
  const [showMenu, setShowMenu] = useState(false);
  const [showLoginPrompt, setShowLoginPrompt] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const { isAuthenticated, isAnonymous } = useAuthStore();

  const {
    uploadedFile,
    isAgentMode,
    isProcessing,
    currentSessionId,
    setUploadedFile,
    removeFile,
    toggleAgentMode,
    setProcessing,
    updateMessage,
    upsertMessage,
    saveMessage,
    persistMessage,
    setCurrentSession,
  } = useChatStore();

  const streamingMessageIdRef = useRef<string | null>(null);
  const streamingContentRef = useRef<string>('');
  const chatStreamingMessageIdRef = useRef<string | null>(null);
  const chatStreamingContentRef = useRef<string>('');

  const {
    runId,
    sessionId,
    currentPhase,
    isRunning,
    error: agentError,
    runAgent,
    abort,
    disconnect,
  } = useAgentStream({
    baseUrl: '/api',
    useJobs: true,
    sessionId: currentSessionId,
    onNodeEnter: (node, message) => {
      setCurrentPhase(node);
      addStep({
        id: `step_${node}_${Date.now()}`,
        role: node,
        task: message || `执行: ${node}`,
        status: 'running',
        startedAt: new Date(),
      });
    },
    onToolCall: (tool, args) => {
      addStep({
        id: `tool_${Date.now()}`,
        role: 'tool',
        task: `调用 ${tool}`,
        status: 'running',
        startedAt: new Date(),
      });
    },
    onToolResult: (tool, result) => {
      // 更新工具步骤为完成
      // TODO: 需要获取对应的step_id
    },
    onPlanUpdate: (plan) => {
      addStep({
        id: `plan_${Date.now()}`,
        role: 'planner',
        task: `计划: ${plan.length} 步`,
        status: 'completed',
        output: plan.join('\n'),
        startedAt: new Date(),
        completedAt: new Date(),
      });
    },
    onProgress: (progress, message) => {
      setCurrentPhase(message || `处理中 ${Math.round(progress * 100)}%`);
    },
    onReport: (report) => {
      setReport(report);
    },
    onComplete: (message) => {
      completeRun(message);
      if (streamingMessageIdRef.current) {
        const messageId = streamingMessageIdRef.current;
        streamingContentRef.current = message || streamingContentRef.current;
        updateMessage(messageId, (current) => ({
          ...current,
          content: streamingContentRef.current || current.content,
          timestamp: new Date(),
        }));
        if (streamingContentRef.current) {
          persistMessage({
            sessionId: currentSessionId,
            role: 'assistant',
            content: streamingContentRef.current,
          });
        }
        streamingMessageIdRef.current = null;
        streamingContentRef.current = '';
      } else {
        saveMessage({
          sessionId: currentSessionId,
          role: 'assistant',
          content: message,
        });
      }
    },
    onError: (errorMsg) => {
      failRun(errorMsg);
      streamingMessageIdRef.current = null;
      streamingContentRef.current = '';
      saveMessage({
        sessionId: currentSessionId,
        role: 'assistant',
        content: `错误: ${errorMsg}`,
      });
    },
    onStart: (runId, sessionId) => {
      setCurrentSession(sessionId);
      streamingMessageIdRef.current = null;
      streamingContentRef.current = '';
      startRun({
        id: runId,
        sessionId,
        objective: '', // 会在handleSend中设置
        status: 'running',
        steps: [],
        createdAt: new Date(),
      });
      setCurrentPhase('开始分析');
    },
    onToken: (token) => {
      if (!token) {
        return;
      }
      if (!streamingMessageIdRef.current) {
        const messageId = `msg_${Date.now()}_assistant_stream`;
        streamingMessageIdRef.current = messageId;
        streamingContentRef.current = '';
        upsertMessage({
          id: messageId,
          sessionId: sessionId || 'current',
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          metadata: { agentRunId: runId || undefined },
        });
      }
      streamingContentRef.current += token;
      const messageId = streamingMessageIdRef.current;
      if (messageId) {
        updateMessage(messageId, (current) => ({
          ...current,
          content: streamingContentRef.current,
          timestamp: new Date(),
        }));
      }
    },
  });

  const {
    sendMessage: chatSendMessage,
    isSending: chatIsSending,
  } = useChat({
    baseUrl: '/api',
    sessionId: currentSessionId,
    stream: true,
    onToken: (token) => {
      if (!token) {
        return;
      }
      if (!chatStreamingMessageIdRef.current) {
        const messageId = `msg_${Date.now()}_assistant_stream`;
        chatStreamingMessageIdRef.current = messageId;
        chatStreamingContentRef.current = '';
        upsertMessage({
          id: messageId,
          sessionId: currentSessionId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
        });
      }
      chatStreamingContentRef.current += token;
      const messageId = chatStreamingMessageIdRef.current;
      if (messageId) {
        updateMessage(messageId, (current) => ({
          ...current,
          content: chatStreamingContentRef.current,
          timestamp: new Date(),
        }));
      }
    },
    onComplete: (message) => {
      if (chatStreamingMessageIdRef.current) {
        const messageId = chatStreamingMessageIdRef.current;
        const finalContent = message || chatStreamingContentRef.current;
        updateMessage(messageId, (current) => ({
          ...current,
          content: finalContent,
          timestamp: new Date(),
        }));
        if (finalContent) {
          persistMessage({
            sessionId: currentSessionId,
            role: 'assistant',
            content: finalContent,
          });
        }
        chatStreamingMessageIdRef.current = null;
        chatStreamingContentRef.current = '';
      } else if (message) {
        saveMessage({
          sessionId: currentSessionId,
          role: 'assistant',
          content: message,
        });
      }
    },
    onError: (errorMsg) => {
      chatStreamingMessageIdRef.current = null;
      chatStreamingContentRef.current = '';
      saveMessage({
        sessionId: currentSessionId,
        role: 'assistant',
        content: `错误: ${errorMsg}`,
      });
    },
  });

  const { startRun, addStep, completeRun, failRun, setReport, setCurrentPhase } = useAgentStore();

  // 关闭菜单
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isProcessing) return;

    // Agent模式需要上传文件
    if (isAgentMode && !uploadedFile) {
      alert('请先上传.h5ad文件');
      return;
    }

    const messageContent = input;
    setInput('');
    setProcessing(true);

    // 添加用户消息到聊天
    await saveMessage({
      sessionId: currentSessionId,
      role: 'user',
      content: messageContent,
    });

    try {
      if (isAgentMode && uploadedFile) {
        // Agent模式 - 使用新的Hook
        await runAgent(messageContent, [uploadedFile.id]);

        // 更新session
    if (sessionId) {
      setCurrentSession(sessionId);
    }
      } else {
        // 对话模式 - 使用Chat Hook
        await chatSendMessage(messageContent);
      }
    } catch (err) {
      console.error('Send error:', err);
      const errorMsg = err instanceof Error ? err.message : '发送失败';
      failRun(errorMsg);
      saveMessage({
        sessionId: currentSessionId,
        role: 'assistant',
        content: `错误: ${errorMsg}`,
      });
    } finally {
      if (!isRunning) {
        setProcessing(false);
      }
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // 验证文件
    const validation = uploadService.validateFile(file);
    if (!validation.valid) {
      alert(validation.error);
      return;
    }

    try {
      setProcessing(true);
      const uploaded = await uploadService.uploadFile(file);
      setUploadedFile(uploaded);
      setShowMenu(false);
    } catch (error) {
      console.error('Upload error:', error);
      alert('文件上传失败');
    } finally {
      setProcessing(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="p-8 bg-white border-t border-slate-100">
      <div className="max-w-3xl mx-auto flex flex-col gap-3">
        {/* 错误提示 */}
        {agentError && (
          <div className="flex items-center gap-2 px-4 py-2 bg-red-50 border border-red-200 rounded-lg">
            <AlertCircle size={16} className="text-red-600 flex-shrink-0" />
            <span className="text-sm text-red-700">{agentError}</span>
          </div>
        )}

        {/* 已上传文件标签 */}
        {uploadedFile && <FilePill file={uploadedFile} onRemove={removeFile} />}

        {/* 输入框 */}
        <form onSubmit={handleSend} className="relative">
          <div className="bg-slate-50 rounded-[2.5rem] border border-slate-200 p-2.5 flex items-center gap-2 shadow-xl">
            {/* 菜单按钮 */}
            <div className="relative" ref={menuRef}>
              <button
                type="button"
                onClick={() => setShowMenu(!showMenu)}
                className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${
                  showMenu ? 'bg-slate-200 rotate-45' : 'bg-white hover:bg-slate-100 shadow-sm text-slate-600'
                }`}
              >
                <Plus size={24} />
              </button>

              {showMenu && (
                <div className="absolute bottom-full left-0 mb-4 w-56 bg-white border border-slate-200 rounded-2xl shadow-2xl p-2 z-50">
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full flex items-center gap-3 px-3 py-3 hover:bg-slate-50 rounded-xl text-slate-700 transition-colors"
                  >
                    <FileUp size={18} className="text-blue-500" />
                    <span className="text-sm font-semibold">上传H5AD文件</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (!isAuthenticated || isAnonymous) {
                        setShowLoginPrompt(true);
                        setShowMenu(false);
                        return;
                      }
                      toggleAgentMode();
                      setShowMenu(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-3 hover:bg-slate-50 rounded-xl text-slate-700 transition-colors"
                  >
                    <Zap size={18} className={isAgentMode ? 'text-amber-500' : 'text-slate-400'} />
                    <span className="text-sm font-semibold">
                      {isAgentMode ? '关闭Agent模式' : 'Agent模式'}
                    </span>
                  </button>
                </div>
              )}
            </div>

            {/* 输入框 */}
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isAgentMode ? '描述分析目标...' : '询问CellMind...'}
              className="flex-1 bg-transparent py-4 px-3 outline-none text-base font-medium text-slate-800 placeholder:text-slate-400"
              disabled={isProcessing}
            />

            {/* 发送按钮 */}
            <button
              type="submit"
              disabled={!input.trim() || isProcessing}
              className={`w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md ${
                input.trim()
                  ? 'bg-blue-600 text-white hover:scale-105 active:scale-95'
                  : 'bg-slate-200 text-slate-400 cursor-not-allowed'
              }`}
            >
              {isProcessing && isRunning ? (
                <div className="w-4 h-4 border-2 border-white/300 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
        </form>

        {/* Agent模式提示 */}
        {isAgentMode && (
          <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-full w-fit mx-auto shadow-sm">
            <div className="flex items-center gap-1.5 text-blue-700">
              <Zap size={14} className={isRunning ? 'animate-pulse' : ''} />
              <span className="text-[10px] font-black uppercase tracking-widest">
                Agent模式已启用
              </span>
            </div>
            <div className="h-3 w-[1px] bg-blue-200 mx-1" />
            <span className="text-[10px] font-bold text-slate-500">
              {uploadedFile ? '已上传文件' : '请上传数据文件'}
            </span>
          </div>
        )}

        {/* 运行状态 */}
        {isRunning && (
          <div className="flex items-center gap-2 px-4 py-2 bg-slate-50 border border-slate-200 rounded-full w-fit mx-auto">
            <span className="text-xs text-slate-600">
              {currentPhase || '处理中...'}
            </span>
            {runId && (
              <span className="text-xs text-slate-400">
                (ID: {runId.slice(0, 8)}...)
              </span>
            )}
          </div>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        className="hidden"
        accept=".h5ad"
        onChange={handleFileChange}
      />

      {/* 登录提示 */}
      {showLoginPrompt && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-sm mx-4 text-center">
            <Zap size={32} className="text-amber-500 mx-auto mb-4" />
            <h3 className="text-lg font-bold text-slate-800 mb-2">需要登录</h3>
            <p className="text-sm text-slate-600 mb-6">
              Agent 模式需要登录后才能使用，请先登录后再试。
            </p>
            <button
              onClick={() => {
                setShowLoginPrompt(false);
                // 触发全局登录对话框
                window.dispatchEvent(new CustomEvent('open-login-dialog'));
              }}
              className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
            >
              去登录
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
