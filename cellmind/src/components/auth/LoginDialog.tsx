/**
 * LoginDialog - 登录对话框组件
 * 支持登录和注册切换
 */
import React, { useState } from 'react';
import { X, User, Mail, Lock, AlertCircle } from 'lucide-react';
import { useAuthStore, useSessionStore } from '@/stores';
import { authService } from '@/services';
import type { LoginRequest, RegisterRequest } from '@/types';
import { PasswordResetDialog } from './PasswordResetDialog';

interface LoginDialogProps {
  isOpen: boolean;
  onClose: () => void;
}

type TabType = 'login' | 'register';

export const LoginDialog: React.FC<LoginDialogProps> = ({ isOpen, onClose }) => {
  const [tab, setTab] = useState<TabType>('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);
  const [showPasswordReset, setShowPasswordReset] = useState(false);

  // 登录表单
  const [loginData, setLoginData] = useState<LoginRequest>({
    username: '',
    password: '',
  });

  // 注册表单
  const [registerData, setRegisterData] = useState<RegisterRequest>({
    username: '',
    email: '',
    password: '',
    full_name: '',
  });
  const [confirmPassword, setConfirmPassword] = useState('');

  const { setAuth } = useAuthStore();
  const { syncSessionsFromBackend } = useSessionStore();

  // 重置表单
  const resetForms = () => {
    setLoginData({ username: '', password: '' });
    setRegisterData({ username: '', email: '', password: '', full_name: '' });
    setConfirmPassword('');
    setError('');
    setShowSuccess(false);
  };

  // 切换 Tab
  const handleTabChange = (newTab: TabType) => {
    setTab(newTab);
    resetForms();
  };

  // 处理登录
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await authService.login(loginData);
      setAuth(response, response.user);
      // 登录成功后刷新会话列表，以获取该用户的会话
      await syncSessionsFromBackend();
      onClose();
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || '登录失败，请稍后重试';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  // 处理注册
  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    // 验证密码
    if (registerData.password.length < 8) {
      setError('密码长度至少为8位');
      return;
    }
    if (registerData.password !== confirmPassword) {
      setError('两次输入的密码不一致');
      return;
    }

    setLoading(true);

    try {
      const response = await authService.register(registerData);
      setAuth(response, response.user);
      // 注册成功后刷新会话列表，以获取该用户的会话
      await syncSessionsFromBackend();
      setShowSuccess(true);
      setTimeout(() => onClose(), 1500);
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || '注册失败，请稍后重试';
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* 背景遮罩 */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* 对话框 */}
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* 关闭按钮 */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-slate-400 hover:text-slate-600 transition-colors"
        >
          <X size={20} />
        </button>

        {/* 头部 */}
        <div className="px-8 pt-8 pb-4">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
              <User size={20} className="text-white" />
            </div>
            <h2 className="text-2xl font-bold text-slate-800">
              {tab === 'login' ? '登录 CellMind' : '注册 CellMind'}
            </h2>
          </div>
          <p className="text-slate-500 text-sm">
            {tab === 'login' ? '欢迎回来！请登录以继续使用' : '创建账号以开始您的分析之旅'}
          </p>
        </div>

        {/* Tab 切换 */}
        <div className="px-8 flex gap-2 border-b border-slate-200">
          <button
            onClick={() => handleTabChange('login')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              tab === 'login'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            登录
          </button>
          <button
            onClick={() => handleTabChange('register')}
            className={`flex-1 py-3 text-sm font-medium transition-colors ${
              tab === 'register'
                ? 'text-blue-600 border-b-2 border-blue-600'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            注册
          </button>
        </div>

        {/* 表单内容 */}
        <div className="px-8 py-6">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-start gap-2">
              <AlertCircle size={16} className="text-red-500 mt-0.5 flex-shrink-0" />
              <span className="text-sm text-red-600">{error}</span>
            </div>
          )}

          {showSuccess && (
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg text-sm text-green-600">
              注册成功！正在自动登录...
            </div>
          )}

          {tab === 'login' ? (
            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  用户名或邮箱
                </label>
                <div className="relative">
                  <User size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    value={loginData.username}
                    onChange={(e) => setLoginData({ ...loginData, username: e.target.value })}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="输入用户名或邮箱"
                    required
                  />
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="block text-sm font-medium text-slate-700">
                    密码
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowPasswordReset(true)}
                    className="text-xs text-blue-600 hover:text-blue-700 transition-colors"
                  >
                    忘记密码？
                  </button>
                </div>
                <div className="relative">
                  <Lock size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={loginData.password}
                    onChange={(e) => setLoginData({ ...loginData, password: e.target.value })}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="输入密码"
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-lg transition-colors"
              >
                {loading ? '登录中...' : '登录'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleRegister} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  用户名
                </label>
                <div className="relative">
                  <User size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    value={registerData.username}
                    onChange={(e) => setRegisterData({ ...registerData, username: e.target.value })}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="输入用户名"
                    pattern="[a-zA-Z0-9_]+"
                    title="用户名只能包含字母、数字和下划线"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  邮箱
                </label>
                <div className="relative">
                  <Mail size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="email"
                    value={registerData.email}
                    onChange={(e) => setRegisterData({ ...registerData, email: e.target.value })}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="输入邮箱"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  全名（可选）
                </label>
                <input
                  type="text"
                  value={registerData.full_name}
                  onChange={(e) => setRegisterData({ ...registerData, full_name: e.target.value })}
                  className="w-full px-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="输入您的全名"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  密码
                </label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={registerData.password}
                    onChange={(e) => setRegisterData({ ...registerData, password: e.target.value })}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="输入密码（至少8位，包含大小写字母和数字）"
                    minLength={8}
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1.5">
                  确认密码
                </label>
                <div className="relative">
                  <Lock size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full pl-10 pr-4 py-2.5 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    placeholder="再次输入密码"
                    required
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-lg transition-colors"
              >
                {loading ? '注册中...' : '注册'}
              </button>
            </form>
          )}
        </div>

        {/* 底部提示 */}
        <div className="px-8 pb-6 text-center text-sm text-slate-500">
          {tab === 'login' ? (
            <span>
              还没有账号？{' '}
              <button
                onClick={() => handleTabChange('register')}
                className="text-blue-600 hover:underline"
              >
                立即注册
              </button>
            </span>
          ) : (
            <span>
              已有账号？{' '}
              <button
                onClick={() => handleTabChange('login')}
                className="text-blue-600 hover:underline"
              >
                立即登录
              </button>
            </span>
          )}
        </div>
      </div>

      {/* 密码重置对话框 */}
      <PasswordResetDialog
        isOpen={showPasswordReset}
        onClose={() => setShowPasswordReset(false)}
      />
    </div>
  );
};
