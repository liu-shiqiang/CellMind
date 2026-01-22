/**
 * Error组件导出
 */

export {
  ErrorBoundary,
  DefaultErrorFallback,
  CompactErrorFallback,
  withErrorBoundary,
} from './ErrorBoundary';

export {
  AgentError,
  NetworkError,
  FileValidationError,
  logError,
  handleError,
} from './ErrorBoundary';

export type {
  ErrorBoundaryProps,
  ErrorFallbackProps,
} from './ErrorBoundary';
