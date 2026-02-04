import React, { useState, useRef, useEffect } from 'react';
import { Bot, User, Download, ChevronDown } from 'lucide-react';
import { MessageContent } from './MessageContent';
import { Avatar } from '../ui/Avatar';
import { formatTime } from '@/utils/helpers';
import { exportReport, ExportFormat } from '@/utils/reportExporter';
import type { Message } from '@/types';

interface MessageItemProps {
  message: Message;
  compact?: boolean;  // 紧凑模式，用于分屏视图
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, compact = false }) => {
  const isUser = message.role === 'user';
  const [showExportMenu, setShowExportMenu] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  // 关闭导出菜单当点击外部
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleExport = (format: ExportFormat) => {
    // 使用会话的第一个消息作为标题，或者使用默认标题
    const title = 'cellmind_analysis_report';
    exportReport(message.content, title, format);
    setShowExportMenu(false);
  };

  if (compact) {
    // 紧凑模式 - 用于分屏视图左侧
    return (
      <div className={`flex gap-2 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <Avatar
          className={`!w-7 !h-7 ${
            isUser
              ? 'bg-slate-100 text-slate-600'
              : 'bg-blue-600 text-white'
          }`}
        >
          {isUser ? <User size={14} /> : <Bot size={14} />}
        </Avatar>

        <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
          <div
            className={`text-sm leading-relaxed ${
              isUser
                ? 'bg-slate-50 px-3 py-2 rounded-2xl text-slate-800'
                : 'text-slate-700'
            }`}
          >
            <MessageContent
              content={message.content}
              toolData={message.metadata as any}
            />
          </div>
        </div>
      </div>
    );
  }

  // 正常模式
  return (
    <div className={`flex gap-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      <Avatar
        className={
          isUser
            ? 'bg-slate-100 text-slate-600'
            : 'bg-blue-600 text-white shadow-xl shadow-blue-100'
        }
      >
        {isUser ? <User size={20} /> : <Bot size={20} />}
      </Avatar>

      <div
        className={`flex flex-col max-w-[85%] ${isUser ? 'items-end' : 'items-start'}`}
      >
        <div className="flex items-start gap-2 group">
          <div
            className={`text-base leading-relaxed ${
              isUser
                ? 'bg-slate-50 px-6 py-4 rounded-3xl border border-slate-100 text-slate-800'
                : 'text-slate-800 font-medium'
            }`}
          >
            <MessageContent
              content={message.content}
              toolData={message.metadata as any}
            />
          </div>
          {/* 导出按钮 - 仅显示在助手消息上，悬停时显示 */}
          {!isUser && (
            <div className="relative" ref={exportMenuRef}>
              <button
                onClick={() => setShowExportMenu(!showExportMenu)}
                className="opacity-0 group-hover:opacity-100 p-2 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-all"
                title="导出报告"
              >
                <Download size={16} />
              </button>
              {showExportMenu && (
                <div className="absolute top-8 left-0 w-40 bg-white border border-slate-200 rounded-xl shadow-xl z-50 animate-in fade-in slide-in-from-top-2">
                  <button
                    onClick={() => handleExport('markdown')}
                    className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 rounded-t-xl text-slate-700 transition-colors text-sm"
                  >
                    Markdown (.md)
                  </button>
                  <button
                    onClick={() => handleExport('html')}
                    className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 text-slate-700 transition-colors text-sm"
                  >
                    HTML (.html)
                  </button>
                  <button
                    onClick={() => handleExport('pdf')}
                    className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 rounded-b-xl text-slate-700 transition-colors text-sm"
                  >
                    PDF (打印)
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
        <span className="text-xs text-slate-400 mt-1">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
};
