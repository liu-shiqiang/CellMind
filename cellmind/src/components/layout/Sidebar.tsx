import React, { useState, useEffect } from 'react';
import { Brain, Plus, Sparkles, History, LayoutDashboard, Trash2, LogIn, User as UserIcon } from 'lucide-react';
import { useSessionStore, isLocalSession, useAuthStore } from '@/stores';
import { useChatStore } from '@/stores';
import { sessionService } from '@/services';
import { LoginDialog, UserMenu } from '@/components/auth';
import type { Session } from '@/types';

export const Sidebar: React.FC = () => {
  const { sessions, currentSession, initSessions, setCurrentSession, removeSession, syncSessionsFromBackend } = useSessionStore();
  const { clearMessages, loadSessionMessages, setCurrentSession: setChatSessionId, setUploadedFile, currentSessionId } = useChatStore();
  const { isAuthenticated } = useAuthStore();
  const [loadingSession, setLoadingSession] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [showLoginDialog, setShowLoginDialog] = useState(false);

  // 初始化时从后端加载会话列表
  useEffect(() => {
    initSessions();
  }, []);

  // 监听全局登录对话框事件
  useEffect(() => {
    const handleOpenLogin = () => setShowLoginDialog(true);
    window.addEventListener('open-login-dialog', handleOpenLogin);
    return () => {
      window.removeEventListener('open-login-dialog', handleOpenLogin);
    };
  }, []);

  // 初始化完成后同步当前会话与消息
  useEffect(() => {
    if (!currentSession?.id || loadingSession) {
      return;
    }
    if (currentSessionId === currentSession.id) {
      return;
    }
    setChatSessionId(currentSession.id);
    loadSessionMessages(currentSession.id).catch((error) => {
      console.error('Failed to load session messages:', error);
    });
  }, [currentSession?.id, currentSessionId, loadingSession, setChatSessionId, loadSessionMessages]);

  // 新建会话 - 同步后端
  const handleNewChat = async () => {
    if (isCreating) return;
    setIsCreating(true);

    try {
      // 调用后端API创建会话
      const newSession = await sessionService.createSession('New Analysis');

      // 更新本地状态
      setCurrentSession(newSession);
      setChatSessionId(newSession.id);
      clearMessages();
      setUploadedFile(null);

      // 刷新会话列表
      await syncSessionsFromBackend();
    } catch (error) {
      console.error('Failed to create session:', error);
      alert('创建会话失败，请检查网络连接');
    } finally {
      setIsCreating(false);
    }
  };

  // 加载会话 - 从后端拉取数据
  const handleLoadSession = async (sessionId: string) => {
    // 如果已经在加载该会话，跳过
    if (loadingSession === sessionId) return;

    try {
      setLoadingSession(sessionId);

      // 检查是否为本地会话（未同步到后端）
      if (isLocalSession(sessionId)) {
        console.warn('[Sidebar] Loading local-only session, messages may not be available');
        // 对于本地会话，只设置当前会话，不清空消息（用户可能正在编辑）
        const localSession = sessions.find((s) => s.id === sessionId);
        if (localSession) {
          setCurrentSession(localSession);
          setChatSessionId(localSession.id);
        }
        return;
      }

      // 1. 从后端加载会话详情
      const session = await sessionService.getSession(sessionId);

      // 2. 使用 chatStore 的 loadSessionMessages 加载消息
      await loadSessionMessages(sessionId);

      // 3. 更新本地会话状态
      setCurrentSession(session);
      setChatSessionId(session.id);

      // 4. 清理临时状态（切换会话时清除上传文件）
      setUploadedFile(null);
    } catch (error) {
      console.error('Failed to load session:', error);
      // 降级：使用本地会话数据
      const localSession = sessions.find((s) => s.id === sessionId);
      if (localSession) {
        setCurrentSession(localSession);
        setChatSessionId(localSession.id);
        clearMessages();
      } else {
        alert('加载会话失败');
      }
    } finally {
      setLoadingSession(null);
    }
  };

  // 删除会话
  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation(); // 防止触发加载会话

    if (!confirm('确定要删除这个会话吗？')) return;

    try {
      // 检查是否为本地会话
      if (isLocalSession(sessionId)) {
        // 本地会话直接从列表中移除
        console.warn('[Sidebar] Deleting local-only session from frontend only');
        removeSession(sessionId);
      } else {
        // 后端会话，调用API删除
        await sessionService.deleteSession(sessionId);
        removeSession(sessionId);
      }

      // 如果删除的是当前会话，清空消息并选择第一个可用会话
      if (currentSession?.id === sessionId) {
        clearMessages();
        setUploadedFile(null);

        const remainingSessions = sessions.filter((s) => s.id !== sessionId);
        if (remainingSessions.length > 0) {
          setCurrentSession(remainingSessions[0]);
          setChatSessionId(remainingSessions[0].id);
        }
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      alert('删除会话失败，请稍后重试');
    }
  };

  return (
    <aside className="w-[280px] bg-slate-50 border-r border-slate-200 flex flex-col h-full z-30">
      {/* Header */}
      <div className="p-4">
        <div className="flex items-center gap-3 mb-6 px-2">
          <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
            <Brain size={18} className="text-white" />
          </div>
          <h1 className="font-bold text-xl tracking-tight text-slate-800">
            CellMind
          </h1>
        </div>

        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-between px-3 py-2.5 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all shadow-sm group"
        >
          <div className="flex items-center gap-2">
            <Plus size={18} className="text-blue-600" />
            <span className="text-sm font-semibold text-slate-700">新建分析</span>
          </div>
          <Sparkles size={14} className="text-slate-300 group-hover:text-blue-400" />
        </button>
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-y-auto px-3">
        <h3 className="px-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
          <History size={12} /> 历史记录
        </h3>
        <div className="space-y-1">
          {sessions.map((session) => (
            <div
              key={session.id}
              className="relative group"
            >
              <button
                onClick={() => handleLoadSession(session.id)}
                disabled={loadingSession === session.id}
                className={`w-full text-left px-3 py-3 rounded-xl transition-all ${
                  currentSession?.id === session.id
                    ? 'bg-white border border-slate-200 shadow-sm'
                    : 'hover:bg-slate-100'
                } ${loadingSession === session.id ? 'opacity-60' : ''}`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <span
                      className={`text-sm font-semibold block truncate ${
                        currentSession?.id === session.id ? 'text-blue-600' : 'text-slate-700'
                      }`}
                    >
                      {session.title}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      {session.message_count || 0} 条消息
                    </span>
                  </div>
                  {loadingSession === session.id && (
                    <span className="text-[10px] text-blue-500">加载中...</span>
                  )}
                </div>
              </button>

              {/* 删除按钮 - 悬停时显示 */}
              <button
                onClick={(e) => handleDeleteSession(session.id, e)}
                className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-100 rounded-lg transition-all"
                title="删除会话"
              >
                <Trash2 size={12} className="text-slate-400 hover:text-red-500" />
              </button>
            </div>
          ))}

          {sessions.length === 0 && (
            <div className="text-center py-8 text-slate-400 text-sm">
              暂无历史记录
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-200">
        {isAuthenticated ? (
          <UserMenu />
        ) : (
          <button
            onClick={() => setShowLoginDialog(true)}
            className="flex items-center gap-3 px-2 w-full hover:bg-slate-100 rounded-lg transition-colors py-1.5"
          >
            <div className="w-9 h-9 rounded-full bg-slate-300 flex items-center justify-center">
              <UserIcon size={18} className="text-slate-500" />
            </div>
            <div className="text-left">
              <span className="text-sm font-medium text-slate-700">点击登录</span>
              <p className="text-[10px] text-slate-400">登录后使用 Agent 模式</p>
            </div>
            <LogIn size={16} className="text-slate-400 ml-auto" />
          </button>
        )}
      </div>

      {/* 登录对话框 */}
      <LoginDialog isOpen={showLoginDialog} onClose={() => setShowLoginDialog(false)} />
    </aside>
  );
};
