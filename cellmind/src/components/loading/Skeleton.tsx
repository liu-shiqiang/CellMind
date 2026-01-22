/**
 * Skeleton - 骨架屏加载组件
 * 提供内容加载时的占位符
 */
import React from 'react';
import { cn } from '@/utils/helpers';

export interface SkeletonProps {
  /** 骨架类型 */
  variant?: 'text' | 'circle' | 'rect' | 'rounded';
  /** 宽度 */
  width?: string | number;
  /** 高度 */
  height?: string | number;
  /** 数量 */
  count?: number;
  /** 自定义类名 */
  className?: string;
  /** 动画类型 */
  animation?: 'pulse' | 'wave' | 'none';
}

/**
 * 基础Skeleton组件
 */
export const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'text',
  width,
  height,
  count = 1,
  className = '',
  animation = 'pulse',
}) => {
  const variants = {
    text: 'rounded h-4',
    circle: 'rounded-full',
    rect: 'rounded-sm',
    rounded: 'rounded-lg',
  };

  const animations = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: '',
  };

  const items = Array.from({ length: count });

  return (
    <>
      {items.map((_, i) => (
        <div
          key={i}
          className={cn(
            'bg-slate-200',
            variants[variant],
            animations[animation],
            className
          )}
          style={{ width: typeof width === 'number' ? `${width}px` : width, height: typeof height === 'number' ? `${height}px` : height }}
        />
      ))}
    </>
  );
};

/**
 * 骨架屏配置
 */
interface SkeletonConfig {
  lines?: number;
  avatar?: boolean;
  title?: boolean;
  subtitle?: boolean;
}

/**
 * 消息骨架屏
 */
export const MessageSkeleton: React.FC<SkeletonConfig> = ({
  lines = 2,
  avatar = true,
  title = false,
  subtitle = false,
}) => {
  return (
    <div className="flex gap-3 p-4">
      {avatar && (
        <Skeleton variant="circle" width={32} height={32} />
      )}
      <div className="flex-1 space-y-2">
        {title && (
          <Skeleton width="40%" height={16} />
        )}
        {subtitle && (
          <Skeleton width="30%" height={14} />
        )}
        {Array.from({ length: lines }).map((_, i) => (
          <Skeleton key={i} width={i === lines - 1 ? '80%' : '100%'} />
        ))}
      </div>
    </div>
  );
};

/**
 * 聊天列表骨架屏
 */
export const ChatSkeleton: React.FC = () => {
  return (
    <div className="space-y-4 p-6">
      <MessageSkeleton lines={2} avatar />
      <MessageSkeleton lines={3} avatar={false} />
      <MessageSkeleton lines={2} avatar />
    </div>
  );
};

/**
 * Agent状态骨架屏
 */
export const AgentStatusSkeleton: React.FC = () => {
  return (
    <div className="border border-slate-200 rounded-2xl p-4 space-y-3">
      <div className="flex items-center gap-2">
        <Skeleton variant="circle" width={16} height={16} />
        <Skeleton width={120} height={16} />
      </div>
      <div className="space-y-2 pl-6">
        <Skeleton width="60%" height={12} />
        <Skeleton width="80%" height={12} />
        <Skeleton width="40%" height={12} />
      </div>
    </div>
  );
};

/**
 * 输入区域骨架屏
 */
export const InputAreaSkeleton: React.FC = () => {
  return (
    <div className="p-8 bg-white border-t border-slate-100">
      <div className="max-w-3xl mx-auto">
        <div className="bg-slate-50 rounded-[2.5rem] border border-slate-200 p-4 flex items-center gap-3">
          <Skeleton variant="circle" width={48} height={48} />
          <Skeleton width="100%" height={20} />
          <Skeleton variant="circle" width={48} height={48} />
        </div>
      </div>
    </div>
  );
};

/**
 * 侧边栏骨架屏
 */
export const SidebarSkeleton: React.FC = () => {
  return (
    <div className="w-64 bg-slate-50 h-full p-4 space-y-3">
      <Skeleton width="60%" height={24} className="mb-6" />
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="space-y-2">
          <Skeleton width="80%" height={16} />
          <Skeleton width="40%" height={12} />
        </div>
      ))}
    </div>
  );
};

/**
 * 报告卡片骨架屏
 */
export const ReportCardSkeleton: React.FC = () => {
  return (
    <div className="border border-slate-200 rounded-xl p-6 space-y-4">
      <div className="flex items-start gap-4">
        <Skeleton variant="circle" width={48} height={48} />
        <div className="flex-1 space-y-2">
          <Skeleton width="40%" height={20} />
          <Skeleton width="70%" height={14} />
        </div>
      </div>
      <div className="space-y-2">
        <Skeleton width="100%" height={14} />
        <Skeleton width="100%" height={14} />
        <Skeleton width="60%" height={14} />
      </div>
    </div>
  );
};

/**
 * 表格骨架屏
 */
export const TableSkeleton: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 5,
  cols = 4,
}) => {
  return (
    <div className="w-full space-y-3">
      {/* 表头 */}
      <div className="flex gap-4">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} width={`${100 / cols}%`} height={16} />
        ))}
      </div>
      {/* 表格行 */}
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="flex gap-4">
          {Array.from({ length: cols }).map((_, j) => (
            <Skeleton key={j} width={`${100 / cols}%`} height={14} />
          ))}
        </div>
      ))}
    </div>
  );
};

/**
 * 卡片骨架屏
 */
export const CardSkeleton: React.FC<{ withImage?: boolean }> = ({
  withImage = true,
}) => {
  return (
    <div className="border border-slate-200 rounded-xl p-4 space-y-4">
      {withImage && (
        <Skeleton width="100%" height={160} />
      )}
      <Skeleton width="60%" height={20} />
      <div className="space-y-2">
        <Skeleton width="100%" height={14} />
        <Skeleton width="90%" height={14} />
        <Skeleton width="70%" height={14} />
      </div>
    </div>
  );
};

/**
 * 加载中组件
 */
export const LoadingSpinner: React.FC<{ size?: number; className?: string }> = ({
  size = 24,
  className = '',
}) => {
  return (
    <div
      className={cn('animate-spin rounded-full border-2 border-slate-300 border-t-blue-600', className)}
      style={{ width: size, height: size }}
    />
  );
};

/**
 * 全屏加载
 */
export const FullPageLoading: React.FC<{ message?: string }> = ({
  message = '加载中...',
}) => {
  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-white/80 backdrop-blur-sm z-50">
      <LoadingSpinner size={40} />
      <p className="mt-4 text-slate-600">{message}</p>
    </div>
  );
};

/**
 * 点状加载指示器
 */
export const DotLoading: React.FC<{ count?: number }> = ({ count = 3 }) => {
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
};

/**
 * 进度条骨架屏
 */
export const ProgressSkeleton: React.FC<{ showLabel?: boolean }> = ({
  showLabel = true,
}) => {
  return (
    <div className="space-y-2">
      {showLabel && <Skeleton width={60} height={14} />}
      <Skeleton width="100%" height={8} variant="rounded" />
    </div>
  );
};

/**
 * 脉搏加载动画
 */
export const PulseLoader: React.FC<{ size?: number }> = ({ size = 40 }) => (
  <div
    className="animate-pulse bg-blue-600 rounded-full"
    style={{ width: size, height: size }}
  />
);

/**
 * 组合加载组件 - 带背景
 */
export const LoadingOverlay: React.FC<{
  loading: boolean;
  message?: string;
  children: React.ReactNode;
}> = ({ loading, message = '加载中...', children }) => {
  if (!loading) {
    return <>{children}</>;
  }

  return (
    <div className="relative">
      <div className="opacity-30 pointer-events-none">{children}</div>
      <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/60 backdrop-blur-sm">
        <LoadingSpinner size={32} />
        <p className="mt-3 text-slate-600">{message}</p>
      </div>
    </div>
  );
};
