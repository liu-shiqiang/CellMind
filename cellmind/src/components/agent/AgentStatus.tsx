import React from 'react';
import { CheckCircle, Circle, AlertCircle, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import type { AgentStep } from '@/types';
import { useAgentStore } from '@/stores/agentStore';

interface AgentStatusProps {
  steps: AgentStep[];
  currentPhase?: string;
}

export const AgentStatus: React.FC<AgentStatusProps> = ({ steps, currentPhase }) => {
  const { isAgentStatusExpanded, toggleAgentStatus, report, setViewMode } = useAgentStore();

  if (steps.length === 0) return null;

  const completedCount = steps.filter(s => s.status === 'completed').length;
  const runningCount = steps.filter(s => s.status === 'running').length;
  const failedCount = steps.filter(s => s.status === 'failed').length;
  const isComplete = completedCount === steps.length && steps.length > 0;

  const getStatusIcon = (status: AgentStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'running':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Circle className="w-4 h-4 text-slate-300" />;
    }
  };

  // 处理查看报告
  const handleViewReport = () => {
    if (report) {
      setViewMode('split');
    }
  };

  return (
    <div className="my-4 bg-slate-50 rounded-xl border border-slate-200 overflow-hidden">
      {/* 可折叠的头部 */}
      <div
        className="p-4 cursor-pointer hover:bg-slate-100/50 transition-colors"
        onClick={toggleAgentStatus}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-700 flex items-center gap-2">
            <Loader2 className={`w-4 h-4 ${!isComplete ? 'animate-spin text-blue-600' : 'text-green-600'}`} />
            分析进度
            {currentPhase && !isComplete && (
              <span className="text-xs font-normal text-slate-500">
                - {currentPhase}
              </span>
            )}
            <span className="text-xs font-normal text-slate-500 ml-2">
              ({completedCount}/{steps.length} 完成)
            </span>
          </h3>
          <button className="text-slate-400 hover:text-slate-600 transition-colors">
            {isAgentStatusExpanded ? (
              <ChevronUp size={18} />
            ) : (
              <ChevronDown size={18} />
            )}
          </button>
        </div>

        {/* 进度条 */}
        <div className="mt-3 h-1.5 bg-slate-200 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              failedCount > 0
                ? 'bg-red-500'
                : isComplete
                ? 'bg-green-500'
                : 'bg-blue-500'
            }`}
            style={{ width: `${(completedCount / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* 展开内容 */}
      {isAgentStatusExpanded && (
        <div className="px-4 pb-4 border-t border-slate-200/50">
          <div className="pt-3 space-y-2">
            {steps.map((step) => (
              <div key={step.id} className="flex items-start gap-3 text-sm">
                <div className="mt-0.5">{getStatusIcon(step.status)}</div>
                <div className="flex-1">
                  <p className="text-slate-700">{step.task}</p>
                  {step.output && step.status === 'completed' && (
                    <p className="text-slate-500 text-xs mt-1">{step.output}</p>
                  )}
                  {step.error && (
                    <p className="text-red-500 text-xs mt-1">{step.error}</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* 分析完成后的操作按钮 */}
          {isComplete && report && (
            <div className="mt-4 pt-4 border-t border-slate-200">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleViewReport();
                }}
                className="w-full py-2.5 px-4 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                查看完整分析报告
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
