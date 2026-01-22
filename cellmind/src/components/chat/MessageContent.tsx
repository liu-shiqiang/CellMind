import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MessageContentProps {
  content: string;
}

export const MessageContent: React.FC<MessageContentProps> = ({ content }) => {
  return (
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
      {content}
    </ReactMarkdown>
  );
};
