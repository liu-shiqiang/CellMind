/**
 * UserMenu - 用户菜单组件
 * 显示用户头像、用户名和登出按钮
 */
import React, { useState, useRef, useEffect } from 'react';
import { User, LogOut, ChevronDown, Settings } from 'lucide-react';
import { useAuthStore, useSessionStore, getUserDisplayName } from '@/stores';

export const UserMenu: React.FC = () => {
  const { user, logout } = useAuthStore();
  const { syncSessionsFromBackend } = useSessionStore();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // 点击外部关闭菜单
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleLogout = async () => {
    logout();
    // 登出后刷新会话列表，以获取匿名用户的会话
    await syncSessionsFromBackend();
    setIsOpen(false);
  };

  const displayName = getUserDisplayName();
  const avatarLetter = displayName.charAt(0).toUpperCase();

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-2 py-1.5 hover:bg-slate-100 rounded-lg transition-colors"
      >
        {/* 头像 */}
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center text-white text-sm font-bold">
          {avatarLetter}
        </div>

        {/* 用户名 */}
        <span className="text-sm font-medium text-slate-700 max-w-[100px] truncate">
          {displayName}
        </span>

        {/* 下拉箭头 */}
        <ChevronDown
          size={14}
          className={`text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* 下拉菜单 */}
      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-48 bg-white rounded-xl shadow-lg border border-slate-200 py-1">
          {/* 用户信息 */}
          <div className="px-4 py-2 border-b border-slate-100">
            <p className="text-xs text-slate-500">当前用户</p>
            <p className="text-sm font-medium text-slate-800 truncate">{displayName}</p>
            {user?.email && (
              <p className="text-xs text-slate-500 truncate">{user.email}</p>
            )}
          </div>

          {/* 菜单项 */}
          <button className="w-full px-4 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2 transition-colors">
            <Settings size={14} />
            账号设置
          </button>

          <button
            onClick={handleLogout}
            className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2 transition-colors"
          >
            <LogOut size={14} />
            登出
          </button>
        </div>
      )}
    </div>
  );
};
