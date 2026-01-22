import React from 'react';
import { Sidebar } from './components/layout/Sidebar';
import { ChatBox } from './components/chat/ChatBox';
import { AgentStatus } from './components/agent/AgentStatus';
import { InputArea } from './components/input/InputArea';
import { SplitViewReader } from './components/report';
import { useChatStore, useAgentStore } from './stores';
import { MessageItem } from './components/chat/MessageItem';
import { Bot } from 'lucide-react';
import type { Message } from '@/types';

// 独立的聊天消息渲染组件，用于分屏模式
const ChatMessages: React.FC<{ messages: Message[]; isProcessing: boolean }> = ({
  messages,
  isProcessing
}) => {
  if (messages.length === 0 && !isProcessing) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-8">
        <div className="w-16 h-16 bg-blue-600 rounded-[2rem] flex items-center justify-center shadow-xl">
          <Bot className="text-white w-8 h-8" />
        </div>
        <div className="max-w-xl px-4">
          <h2 className="text-2xl font-bold text-slate-900 mb-2">
            今天CellMind可以帮你做什么？
          </h2>
          <p className="text-slate-500 text-sm">
            上传数据或开始分析
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 py-4">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} compact />
      ))}
      {isProcessing && (
        <div className="flex gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" />
          </div>
          <div className="text-slate-400 text-sm py-2">AI正在思考...</div>
        </div>
      )}
    </div>
  );
};

function App() {
  const { messages, sendMessage } = useChatStore();
  const { isProcessing, run, currentPhase, report, viewMode } = useAgentStore();

  const steps = run?.steps ?? [];

  React.useEffect(() => {
    // 初始化会话
  }, []);

  const handleSendMessage = async (content: string) => {
    await sendMessage(content);
  };

  // 分屏模式渲染
  if (viewMode === 'split' && report) {
    return (
      <div className="flex h-screen bg-white text-slate-900 font-sans">
        <Sidebar />
        <SplitViewReader
          report={report}
          messages={<ChatMessages messages={messages} isProcessing={isProcessing} />}
          inputArea={<InputArea onSendMessage={handleSendMessage} />}
        />
      </div>
    );
  }

  // 普通聊天模式渲染
  return (
    <div className="flex h-screen bg-white text-slate-900 font-sans">
      <Sidebar />

      <main className="flex-1 flex flex-col relative bg-white overflow-hidden">
        <ChatBox messages={messages} isProcessing={isProcessing} />

        {/* Agent状态展示 - 在最后一条AI消息后显示 */}
        {messages.length > 0 && steps.length > 0 && (
          <div className="max-w-4xl mx-auto w-full px-6">
            <AgentStatus steps={steps} currentPhase={currentPhase} />
          </div>
        )}

        <InputArea onSendMessage={handleSendMessage} />
      </main>
    </div>
  );
}

export default App;
