import React from 'react';
import {
  FileText,
  ChevronDown,
  ChevronUp,
  Download,
  ExternalLink
} from 'lucide-react';
import type { AnalysisReport } from '@/types';

interface ReportCardProps {
  report: AnalysisReport;
  onViewReport: () => void;
  onDownload?: () => void;
}

export const ReportCard: React.FC<ReportCardProps> = ({
  report,
  onViewReport,
  onDownload
}) => {
  const [isExpanded, setIsExpanded] = React.useState(false);

  // 统计完成的步骤数
  const completedSteps = report.sections?.filter(s => s.status === 'success').length || 0;
  const totalSteps = report.sections?.length || 0;

  return (
    <div className="my-4 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200 overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* 报告卡片头部 */}
      <div
        className="p-4 cursor-pointer flex items-center justify-between"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-600 flex items-center justify-center shadow-sm">
            <FileText className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">{report.title}</h3>
            <p className="text-xs text-slate-500">
              {completedSteps}/{totalSteps} 分析完成 · {new Date(report.createdAt).toLocaleString('zh-CN')}
            </p>
          </div>
        </div>
        <button className="text-slate-400 hover:text-slate-600 transition-colors">
          {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {/* 展开的内容 - 报告摘要 */}
      {isExpanded && report.summary && (
        <div className="px-4 pb-4">
          <div className="bg-white/70 rounded-lg p-3 border border-blue-100">
            <p className="text-sm text-slate-600 leading-relaxed whitespace-pre-wrap">
              {report.summary}
            </p>
          </div>

          {/* 操作按钮 */}
          <div className="flex gap-2 mt-3">
            <button
              onClick={(e) => {
                e.stopPropagation();
                onViewReport();
              }}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <ExternalLink size={16} />
              查看完整报告
            </button>
            {onDownload && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDownload();
                }}
                className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 text-slate-700 text-sm font-medium rounded-lg border border-slate-200 transition-colors"
              >
                <Download size={16} />
                下载报告
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
