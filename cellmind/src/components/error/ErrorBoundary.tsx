/**
 * ErrorBoundary - 错误边界组件
 * 捕获子组件树中的JavaScript错误，显示备用UI
 */
import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export interface ErrorBoundaryProps {
  children: ReactNode;
  /** 自定义错误UI组件 */
  fallback?: React.ComponentType<ErrorFallbackProps>;
  /** 错误回调 */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** 是否在开发环境显示完整错误栈 */
  showStackTrace?: boolean;
}

export interface ErrorFallbackProps {
  error: Error;
  errorInfo?: ErrorInfo | null;
  retry: () => void;
  reset: () => void;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * 默认错误回退UI
 */
export const DefaultErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  errorInfo,
  retry,
  reset,
}) => {
  const isDev = process.env.NODE_ENV === 'development';

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] p-8">
      <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mb-4">
        <AlertTriangle className="w-8 h-8 text-red-600" />
      </div>

      <h2 className="text-xl font-bold text-slate-900 mb-2">
        出错了
      </h2>

      <p className="text-slate-500 mb-6 text-center max-w-md">
        {error.message || '遇到意外错误，请重试'}
      </p>

      {isDev && (
        <details className="mb-6 text-left w-full max-w-lg">
          <summary className="cursor-pointer text-sm text-slate-400 hover:text-slate-600 mb-2">
            错误详情
          </summary>
          <pre className="text-xs bg-slate-100 p-4 rounded-lg overflow-auto max-h-48">
            {error.toString()}
            {errorInfo?.componentStack}
          </pre>
        </details>
      )}

      <div className="flex gap-3">
        <button
          onClick={retry}
          className="px-6 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors flex items-center gap-2"
        >
          <RefreshCw size={16} />
          重试
        </button>
        <button
          onClick={reset}
          className="px-6 py-2 bg-slate-200 text-slate-700 rounded-full hover:bg-slate-300 transition-colors"
        >
          重置
        </button>
      </div>
    </div>
  );
};

/**
 * 简化版错误回退UI - 用于小空间
 */
export const CompactErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  retry,
}) => (
  <div className="flex items-center gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
    <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0" />
    <div className="flex-1 min-w-0">
      <p className="text-sm text-red-800 font-medium truncate">
        {error.message || '发生错误'}
      </p>
    </div>
    <button
      onClick={retry}
      className="px-3 py-1 bg-red-600 text-white text-sm rounded hover:bg-red-700"
    >
      重试
    </button>
  </div>
);

/**
 * ErrorBoundary组件
 *
 * @example
 * ```tsx
 * <ErrorBoundary onError={(error) => logError(error)}>
 *   <MyComponent />
 * </ErrorBoundary>
 *
 * // 使用自定义回退UI
 * <ErrorBoundary fallback={MyCustomFallback}>
 *   <MyComponent />
 * </ErrorBoundary>
 * ```
 */
export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // 更新state以包含errorInfo
    this.setState({ errorInfo });

    // 调用错误回调
    this.props.onError?.(error, errorInfo);

    // 上报错误到日志服务
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);

    // 可以在这里添加错误上报服务
    // reportErrorToService(error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    // 可以在这里重置整个应用状态
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      const { fallback: FallbackComponent = DefaultErrorFallback } = this.props;

      return (
        <FallbackComponent
          error={this.state.error!}
          errorInfo={this.state.errorInfo}
          retry={this.handleRetry}
          reset={this.handleReset}
        />
      );
    }

    return this.props.children;
  }
}

/**
 * Hook版本的错误边界（使用useError）
 * 注意: useError仍在实验阶段，建议使用类组件版本
 */
// export function useErrorBoundary() {
//   return React.useError();
// }

/**
 * HOC模式的错误边界
 */
export function withErrorBoundary<P extends object>(
  Component: React.ComponentType<P>,
  errorBoundaryProps?: Omit<ErrorBoundaryProps, 'children'>
) {
  const WrappedComponent: React.FC<P> = (props) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;

  return WrappedComponent;
}

/**
 * 错误类型
 */
export class AgentError extends Error {
  constructor(
    message: string,
    public code?: string,
    public runId?: string,
    public details?: any
  ) {
    super(message);
    this.name = 'AgentError';
  }
}

export class NetworkError extends Error {
  constructor(message: string, public statusCode?: number) {
    super(message);
    this.name = 'NetworkError';
  }
}

export class FileValidationError extends Error {
  constructor(
    message: string,
    public fileName?: string,
    public validationErrors?: string[]
  ) {
    super(message);
    this.name = 'FileValidationError';
  }
}

/**
 * 错误日志工具
 */
export function logError(error: Error, context?: Record<string, any>) {
  const errorLog = {
    name: error.name,
    message: error.message,
    stack: error.stack,
    context,
    timestamp: new Date().toISOString(),
    userAgent: navigator.userAgent,
    url: window.location.href,
  };

  // 开发环境输出到控制台
  if (process.env.NODE_ENV === 'development') {
    console.error('[Error Log]', errorLog);
  }

  // 生产环境上报到服务
  if (process.env.NODE_ENV === 'production') {
    // TODO: 实现错误上报服务
    // reportToService(errorLog);
  }
}

/**
 * 创建错误处理装饰器
 */
export function handleError<T extends (...args: any[]) => any>(
  fn: T,
  errorHandler?: (error: Error, ...args: any[]) => void
): T {
  return ((...args: any[]) => {
    try {
      return fn(...args);
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      errorHandler?.(err, ...args);
      logError(err, { function: fn.name, args });
      throw err;
    }
  }) as T;
}
