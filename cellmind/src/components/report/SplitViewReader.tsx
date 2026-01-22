import React, { useRef, useEffect } from 'react';
import { X, Maximize2, Minimize2, Download } from 'lucide-react';
import type { AnalysisReport } from '@/types';
import { useAgentStore } from '@/stores/agentStore';

// 简单的 Markdown 渲染组件
const MarkdownRenderer: React.FC<{ content: string }> = ({ content }) => {
  const html = React.useMemo(() => {
    let html = content;

    // 转义 HTML
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    // 标题
    html = html.replace(/^### (.*$)/gim, '<h3 class="text-lg font-semibold text-slate-800 mt-6 mb-3">$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2 class="text-xl font-bold text-slate-900 mt-8 mb-4 pb-2 border-b border-slate-200">$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1 class="text-2xl font-bold text-slate-900 mt-6 mb-4">$1</h1>');

    // 粗体
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-slate-800">$1</strong>');

    // 斜体
    html = html.replace(/\*(.*?)\*/g, '<em class="text-slate-600">$1</em>');

    // 代码块
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre class="bg-slate-800 text-slate-100 p-4 rounded-lg overflow-x-auto my-4 text-sm"><code>$2</code></pre>');

    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code class="bg-slate-100 text-slate-800 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>');

    // 列表
    html = html.replace(/^\- (.*$)/gim, '<li class="ml-4 text-slate-700 my-1">$1</li>');
    html = html.replace(/(<li.*<\/li>)/s, '<ul class="list-disc">$1</ul>');

    // 链接
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" class="text-blue-600 hover:text-blue-800 underline" target="_blank" rel="noopener">$1</a>');

    // 换行
    html = html.replace(/\n\n/g, '</p><p class="my-3 text-slate-700 leading-relaxed">');
    html = '<p class="my-3 text-slate-700 leading-relaxed">' + html + '</p>';

    // 清理空段落
    html = html.replace(/<p class="[^"]*"><\/p>/g, '');

    return html;
  }, [content]);

  return (
    <div
      className="prose prose-slate max-w-none"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};

interface SplitViewReaderProps {
  report: AnalysisReport;
  messages: React.ReactNode;
  inputArea: React.ReactNode;
}

export const SplitViewReader: React.FC<SplitViewReaderProps> = ({
  report,
  messages,
  inputArea
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = React.useState(false);
  const { setViewMode } = useAgentStore();

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
  }, [report]);

  const handleClose = () => {
    setViewMode('chat');
  };

  const handleDownload = () => {
    // 创建 Blob 并下载
    const blob = new Blob([report.content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${report.title}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div
      className={`
        fixed inset-0 z-50 bg-white flex flex-col
        ${isFullscreen ? 'p-0' : 'p-4'}
        transition-all duration-300
      `}
    >
      {/* 顶部工具栏 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 bg-white">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-slate-800">{report.title}</h2>
          {report.metadata?.nCells && (
            <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
              {report.metadata.nCells.toLocaleString()} 细胞
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleDownload}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="下载报告"
          >
            <Download size={18} className="text-slate-600" />
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title={isFullscreen ? '退出全屏' : '全屏'}
          >
            {isFullscreen ? (
              <Minimize2 size={18} className="text-slate-600" />
            ) : (
              <Maximize2 size={18} className="text-slate-600" />
            )}
          </button>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-slate-100 rounded-lg transition-colors"
            title="关闭"
          >
            <X size={18} className="text-slate-600" />
          </button>
        </div>
      </div>

      {/* 分屏内容区域 */}
      <div className="flex-1 flex overflow-hidden">
        {/* 左侧对话区域 (1/3) */}
        <div className="w-1/3 min-w-[300px] border-r border-slate-200 flex flex-col bg-slate-50">
          <div className="flex-1 overflow-y-auto p-4">
            {messages}
          </div>
          <div className="border-t border-slate-200 bg-white">
            {inputArea}
          </div>
        </div>

        {/* 右侧报告区域 (2/3) */}
        <div className="flex-1 overflow-hidden bg-white">
          <div
            ref={scrollRef}
            className="h-full overflow-y-auto p-8"
          >
            <div className="max-w-4xl mx-auto">
              {/* 报告头部 */}
              <div className="mb-8 pb-6 border-b border-slate-200">
                <h1 className="text-3xl font-bold text-slate-900 mb-4">
                  {report.title}
                </h1>
                <div className="flex flex-wrap gap-4 text-sm text-slate-500">
                  <span>生成时间: {new Date(report.createdAt).toLocaleString('zh-CN')}</span>
                  {report.metadata?.dataFile && (
                    <span>数据文件: {report.metadata.dataFile.split('/').pop()}</span>
                  )}
                </div>
              </div>

              {/* 报告内容 */}
              <MarkdownRenderer content={report.content} />

              {/* 报告底部 */}
              <div className="mt-12 pt-6 border-t border-slate-200 text-center text-sm text-slate-400">
                <p>本报告由 CellMind 自动生成</p>
                <button
                  onClick={handleDownload}
                  className="mt-3 text-blue-600 hover:text-blue-800"
                >
                  下载 Markdown 版本
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
