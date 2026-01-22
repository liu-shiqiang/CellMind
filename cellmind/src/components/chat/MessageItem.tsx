import React from 'react';
import { Bot, User } from 'lucide-react';
import { MessageContent } from './MessageContent';
import { Avatar } from '../ui/Avatar';
import { formatTime } from '@/utils/helpers';
import type { Message } from '@/types';

interface MessageItemProps {
  message: Message;
  compact?: boolean;  // 紧凑模式，用于分屏视图
}

export const MessageItem: React.FC<MessageItemProps> = ({ message, compact = false }) => {
  const isUser = message.role === 'user';

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
            <MessageContent content={message.content} />
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
        <div
          className={`text-base leading-relaxed ${
            isUser
              ? 'bg-slate-50 px-6 py-4 rounded-3xl border border-slate-100 text-slate-800'
              : 'text-slate-800 font-medium'
          }`}
        >
          <MessageContent content={message.content} />
        </div>
        <span className="text-xs text-slate-400 mt-1">
          {formatTime(message.timestamp)}
        </span>
      </div>
    </div>
  );
};
