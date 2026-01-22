import React, { useRef, useEffect } from 'react';
import { Bot } from 'lucide-react';
import { MessageItem } from './MessageItem';
import { ReportCard } from '@/components/report';
import { useAgentStore } from '@/stores/agentStore';
import api from '@/services/api';
import type { Message } from '@/types';

interface ChatBoxProps {
  messages: Message[];
  isProcessing?: boolean;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ messages, isProcessing = false }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const { report, setViewMode, run } = useAgentStore();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing, report]);

  // 处理查看报告
  const handleViewReport = () => {
    if (report) {
      setViewMode('split');
    }
  };

  // 处理下载报告
  const handleDownloadReport = async () => {
    const runId = run?.id;
    if (runId) {
      try {
        const response: { artifacts: Array<{ path: string }> } = await api.get(
          `/jobs/${runId}/artifacts`,
          { params: { type: 'reports' } }
        );
        const reportFile = response.artifacts.find((item) => item.path.endsWith('.md'))
          || response.artifacts.find((item) => item.path.endsWith('.json'));
        if (reportFile) {
          const baseUrl = api.defaults.baseURL || '';
          const url = `${baseUrl}/jobs/${runId}/artifacts/download?path=${encodeURIComponent(reportFile.path)}`;
          const a = document.createElement('a');
          a.href = url;
          a.download = reportFile.path.split('/').pop() || `analysis_report_${runId}.md`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          return;
        }
      } catch (error) {
        console.error('Failed to download report:', error);
      }
    }

    if (report) {
      const blob = new Blob([report.content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${report.title}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  if (messages.length === 0 && !isProcessing && !report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-8">
        <div className="w-20 h-20 bg-blue-600 rounded-[2.5rem] flex items-center justify-center shadow-2xl shadow-blue-100">
          <Bot className="text-white w-10 h-10" />
        </div>
        <div className="max-w-xl">
          <h2 className="text-3xl font-black text-slate-900 mb-3 tracking-tight">
            今天CellMind可以帮你做什么？
          </h2>
          <p className="text-slate-500 text-base leading-relaxed font-medium">
            分析、聚类和解读单细胞数据集，使用我们的多Agent RAG系统。
            上传你的数据或从研究假设开始。
          </p>
        </div>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">
        {messages.map((message) => (
          <MessageItem key={message.id} message={message} />
        ))}

        {/* 报告卡片 - 在消息后显示 */}
        {report && (
          <ReportCard
            report={report}
            onViewReport={handleViewReport}
            onDownload={handleDownloadReport}
          />
        )}

        {isProcessing && (
          <div className="flex gap-4">
            <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" />
            </div>
            <div className="text-slate-400 text-sm">AI正在思考...</div>
          </div>
        )}
      </div>
    </div>
  );
};
