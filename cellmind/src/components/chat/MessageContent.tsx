import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { InlinePlot, PlotItem } from '../analysis';

interface MessageContentProps {
  content: string;
  toolResults?: ToolResult[];
  toolData?: Record<string, any>;
}

interface ToolResult {
  tool: string;
  data?: {
    plots?: PlotItem[];
    plot?: PlotItem;
    heatmap_plot?: PlotItem;
    annotated_umap_plot?: PlotItem;
    [key: string]: any;
  };
  [key: string]: any;
}

/**
 * 解析消息内容中的图表数据
 * 从工具结果或直接嵌入在消息中的JSON数据提取图表
 */
const extractPlotsFromContent = (content: string): PlotItem[] => {
  const plots: PlotItem[] = [];

  try {
    // 尝试匹配 markdown 代码块中的 JSON 数据
    const jsonBlockRegex = /```json\s*([\s\S]*?)\s*```/g;
    let match;

    while ((match = jsonBlockRegex.exec(content)) !== null) {
      try {
        const data = JSON.parse(match[1]);
        // 检查是否是包含plots的数据
        if (data.plots && Array.isArray(data.plots)) {
          plots.push(...data.plots);
        }
        if (data.plot) {
          plots.push(data.plot);
        }
      } catch {
        // 忽略解析错误
      }
    }
  } catch {
    // 忽略错误
  }

  return plots;
};

export const MessageContent: React.FC<MessageContentProps> = ({ content, toolResults = [], toolData = {} }) => {
  // 从工具结果中提取图表
  const plotsFromTools: PlotItem[] = [];

  for (const result of toolResults) {
    // 检查plots数组
    if (result.data?.plots && Array.isArray(result.data.plots)) {
      plotsFromTools.push(...result.data.plots);
    }
    // 检查各种可能的单个plot字段
    const singlePlot = result.data?.plot || result.data?.heatmap_plot || result.data?.annotated_umap_plot;
    if (singlePlot) {
      plotsFromTools.push(singlePlot);
    }
  }

  // 从内容中提取图表
  const plotsFromContent = extractPlotsFromContent(content);

  // 从toolData中提取图表
  const plotsFromToolData: PlotItem[] = [];
  if (toolData.plots && Array.isArray(toolData.plots)) {
    plotsFromToolData.push(...toolData.plots);
  }

  // 合并所有图表，去重
  const allPlots = [...plotsFromTools, ...plotsFromContent, ...plotsFromToolData];
  const uniquePlots = allPlots.filter((plot, index, self) =>
    index === self.findIndex((p) => p.name === plot.name)
  );

  // 将内容中的图表数据移除（避免重复显示）
  let cleanContent = content;
  if (uniquePlots.length > 0) {
    // 移除 ```json 包含的图表数据
    cleanContent = cleanContent.replace(/```json\s*[\s\S]*?\s*```/g, (match) => {
      try {
        const data = JSON.parse(match.replace(/```json\s*/, '').replace(/\s*```/, ''));
        if (data.plots || data.plot) {
          return ''; // 移除包含图表的JSON块
        }
        return match;
      } catch {
        return match;
      }
    });
  }

  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node, ...props }) => (
            <h1 className="text-xl font-bold mt-4 mb-2" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-lg font-bold mt-3 mb-2" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-base font-bold mt-2 mb-1" {...props} />
          ),
          p: ({ node, ...props }) => (
            <p className="my-2 leading-relaxed" {...props} />
          ),
          ul: ({ node, ...props }) => (
            <ul className="list-disc list-inside my-2 space-y-1" {...props} />
          ),
          ol: ({ node, ...props }) => (
            <ol className="list-decimal list-inside my-2 space-y-1" {...props} />
          ),
          li: ({ node, ...props }) => (
            <li className="ml-4" {...props} />
          ),
          code: ({ node, inline, ...props }) =>
            inline ? (
              <code className="bg-slate-100 px-1.5 py-0.5 rounded text-sm text-slate-700 font-mono" {...props} />
            ) : (
              <code className="block bg-slate-100 p-3 rounded-lg my-2 overflow-x-auto text-sm text-slate-700 font-mono" {...props} />
            ),
          pre: ({ node, ...props }) => (
            <pre className="bg-slate-100 p-3 rounded-lg my-2 overflow-x-auto" {...props} />
          ),
          blockquote: ({ node, ...props }) => (
            <blockquote className="border-l-4 border-slate-300 pl-4 italic my-2 text-slate-600" {...props} />
          ),
          a: ({ node, ...props }) => (
            <a className="text-blue-600 hover:text-blue-700 underline" {...props} />
          ),
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border border-slate-200" {...props} />
            </div>
          ),
          th: ({ node, ...props }) => (
            <th className="border border-slate-200 px-4 py-2 bg-slate-50 font-semibold text-left" {...props} />
          ),
          td: ({ node, ...props }) => (
            <td className="border border-slate-200 px-4 py-2" {...props} />
          ),
        }}
      >
        {cleanContent}
      </ReactMarkdown>

      {/* 显示图表 */}
      {uniquePlots.length > 0 && (
        <div className="mt-4 space-y-4">
          {uniquePlots.map((plot) => (
            <InlinePlot key={plot.name} plot={plot} />
          ))}
        </div>
      )}
    </div>
  );
};
