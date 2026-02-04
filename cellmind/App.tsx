
import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Send, Plus, Brain, Sparkles, FileUp, Zap, X, Activity,
  Database, Binary, Beaker, History, LayoutDashboard, Settings,
  MoreVertical, ChevronRight, MousePointer2, FileText, Edit, Check, Trash2, Download, ChevronDown
} from 'lucide-react';
import { AgentRole, AnalysisStep, Message, AnalysisState, CellCluster } from './types';
import { generateAnalysisPlan, interpretResults } from './services/geminiService';
import { AgentStatus } from './components/AgentStatus';
import { UmapVisualization } from './components/UmapVisualization';
import { apiService } from './services/apiService';
import { exportReport, ExportFormat } from './utils/reportExporter';

interface ChatSession {
  id: string;
  title: string;
  date: string;
  preview: string;
  messages: Message[];
  clusters: CellCluster[];
}

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [isAgentMode, setIsAgentMode] = useState(false);
  const [showPlusMenu, setShowPlusMenu] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string>('new');

  // Session editing states
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState('');

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);

  // Export menu state
  const [showExportMenu, setShowExportMenu] = useState(false);
  const exportMenuRef = useRef<HTMLDivElement>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  const [sessions, setSessions] = useState<ChatSession[]>([
    { id: '1', title: 'PBMC Immune Response', date: '2 hours ago', preview: 'Analyzing T-cell heterogeneity...', messages: [], clusters: [] }
  ]);
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [state, setState] = useState<AnalysisState>({
    isProcessing: false,
    steps: [],
    clusters: [],
    currentPhase: 'Initialization'
  });
  
  const [selectedCluster, setSelectedCluster] = useState<CellCluster | null>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, state.steps]);

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setShowPlusMenu(false);
      }
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const startNewChat = () => {
    setCurrentSessionId('new');
    setMessages([]);
    setIsAgentMode(false);
    setUploadedFile(null);
    setState({ isProcessing: false, steps: [], clusters: [], currentPhase: 'Initialization' });
    setSelectedCluster(null);
  };

  const loadSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setMessages(session.messages);
    setState(prev => ({ ...prev, clusters: session.clusters, steps: [] }));
  };

  // Session editing functions
  const handleStartEdit = (sessionId: string, currentTitle: string) => {
    setEditingSessionId(sessionId);
    setEditingTitle(currentTitle);
  };

  const handleSaveEdit = async (sessionId: string) => {
    if (!editingTitle.trim()) {
      handleCancelEdit();
      return;
    }

    try {
      await apiService.updateSession(sessionId, editingTitle.trim());
      setSessions(prev =>
        prev.map(s =>
          s.id === sessionId
            ? { ...s, title: editingTitle.trim() }
            : s
        )
      );
      handleCancelEdit();
    } catch (error) {
      console.error('Failed to update session:', error);
      alert('Failed to update session title');
    }
  };

  const handleCancelEdit = () => {
    setEditingSessionId(null);
    setEditingTitle('');
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();

    if (!confirm('Are you sure you want to delete this session?')) {
      return;
    }

    try {
      await apiService.deleteSession(sessionId);
      setSessions(prev => prev.filter(s => s.id !== sessionId));

      // If deleted session was current, start new chat
      if (currentSessionId === sessionId) {
        startNewChat();
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      alert('Failed to delete session');
    }
  };

  // Auto-focus edit input
  useEffect(() => {
    if (editingSessionId && editInputRef.current) {
      editInputRef.current.focus();
      editInputRef.current.select();
    }
  }, [editingSessionId]);

  const mockExecution = useCallback(async (prompt: string) => {
    setState(s => ({ ...s, currentPhase: 'Embedding', steps: s.steps.map(st => st.id === '1' ? { ...st, status: 'running' } : st) }));
    await new Promise(r => setTimeout(r, 1500));
    
    const clusters: CellCluster[] = [
      { id: '1', size: 1200, markers: ['CD3D', 'CD3E', 'CD4'], suggestedType: 'T Cells', embedding: Array.from({length: 50}, () => [Math.random()*4-2, Math.random()*4-2]) },
      { id: '2', size: 800, markers: ['CD19', 'MS4A1'], suggestedType: 'B Cells', embedding: Array.from({length: 40}, () => [Math.random()*4+2, Math.random()*4+2]) },
    ];

    setState(s => ({ 
      ...s, 
      clusters, 
      steps: s.steps.map(st => st.id === '1' ? { ...st, status: 'completed', output: 'CellMind: scGPT processing complete.' } : st) 
    }));

    setState(s => ({ ...s, currentPhase: 'Annotation', steps: s.steps.map(st => st.id === '2' ? { ...st, status: 'running' } : st) }));
    await new Promise(r => setTimeout(r, 1200));
    
    const interpretation = await interpretResults(prompt, clusters[0].markers);
    const finalAssistantMsg: Message = { id: Date.now().toString(), role: 'assistant', content: interpretation, timestamp: new Date() };

    setMessages(prev => [...prev, finalAssistantMsg]);
    setState(s => ({ ...s, isProcessing: false }));
  }, [currentSessionId, messages]);

  const handleStartAnalysis = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!input.trim() || state.isProcessing) return;

    // Logic Implementation: Check requirements for Agent Mode
    if (isAgentMode) {
      if (!uploadedFile) {
        alert("Please upload an .h5ad file to proceed with Agent Analysis.");
        return;
      }
    }

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    const userPrompt = input;
    setInput('');
    setState(s => ({ ...s, isProcessing: true, steps: [], clusters: [] }));

    try {
      if (isAgentMode && uploadedFile) {
        const plan = await generateAnalysisPlan(userPrompt);
        setState(s => ({ ...s, steps: plan }));
        mockExecution(userPrompt);
      } else {
        // Condition 1: Normal Conversation
        const response = await interpretResults(userPrompt, []);
        const finalAssistantMsg: Message = { id: Date.now().toString(), role: 'assistant', content: response, timestamp: new Date() };
        setMessages(prev => [...prev, finalAssistantMsg]);
        setState(s => ({ ...s, isProcessing: false }));
      }
    } catch (error) {
      console.error(error);
      setState(s => ({ ...s, isProcessing: false }));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.name.endsWith('.h5ad')) {
      setUploadedFile(file);
      setShowPlusMenu(false);
    } else if (file) {
      alert("Only .h5ad files are supported.");
    }
  };

  // Drag and drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only clear if leaving the main area, not entering a child element
    if (e.currentTarget === e.target) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith('.h5ad')) {
      setUploadedFile(file);
    } else if (file) {
      alert("Only .h5ad files are supported.");
    }
  };

  // Export handler
  const handleExport = (format: ExportFormat) => {
    // Get the last assistant message as report content
    const lastAssistantMsg = messages.filter(m => m.role === 'assistant').pop();
    if (!lastAssistantMsg) {
      alert('No analysis report to export');
      return;
    }

    const currentSession = sessions.find(s => s.id === currentSessionId);
    const title = currentSession?.title || 'analysis_report';

    exportReport(lastAssistantMsg.content, title, format);
    setShowExportMenu(false);
  };

  return (
    <div className="flex h-screen bg-white text-slate-900 font-sans">
      {/* Sidebar */}
      <aside className="w-[280px] bg-slate-50 border-r border-slate-200 flex flex-col h-full z-30">
        <div className="p-4">
          <div className="flex items-center gap-3 mb-6 px-2">
            <div className="w-8 h-8 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg">
               <Brain size={18} className="text-white" />
            </div>
            <h1 className="font-bold text-xl tracking-tight text-slate-800">CellMind</h1>
          </div>
          
          <button onClick={startNewChat} className="w-full flex items-center justify-between px-3 py-2.5 bg-white border border-slate-200 rounded-xl hover:bg-slate-50 transition-all shadow-sm group">
            <div className="flex items-center gap-2">
              <Plus size={18} className="text-blue-600" />
              <span className="text-sm font-semibold text-slate-700">New Analysis</span>
            </div>
            <Sparkles size={14} className="text-slate-300 group-hover:text-blue-400" />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 space-y-6">
          <div>
            <h3 className="px-3 text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
              <History size={12} /> History
            </h3>
            <div className="space-y-1">
              {sessions.map(session => {
                const isEditing = editingSessionId === session.id;
                const isActive = currentSessionId === session.id;

                return (
                  <div
                    key={session.id}
                    className={`relative rounded-xl transition-all group overflow-hidden ${isActive ? 'bg-white border border-slate-200 shadow-sm' : 'hover:bg-slate-100'}`}
                  >
                    {isEditing ? (
                      <div className="px-3 py-2 flex items-center gap-2">
                        <input
                          ref={editInputRef}
                          type="text"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              handleSaveEdit(session.id);
                            } else if (e.key === 'Escape') {
                              handleCancelEdit();
                            }
                          }}
                          className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-2 py-1.5 text-sm font-medium text-slate-800 outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                        />
                        <button
                          onClick={() => handleSaveEdit(session.id)}
                          className="p-1.5 rounded-lg bg-green-50 text-green-600 hover:bg-green-100 transition-colors"
                          title="Save"
                        >
                          <Check size={14} />
                        </button>
                        <button
                          onClick={handleCancelEdit}
                          className="p-1.5 rounded-lg bg-slate-100 text-slate-500 hover:bg-slate-200 transition-colors"
                          title="Cancel"
                        >
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => loadSession(session)}
                        className="w-full text-left px-3 py-3"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex-1 min-w-0">
                            <span className={`text-sm font-semibold block truncate ${isActive ? 'text-blue-600' : 'text-slate-700'}`}>
                              {session.title}
                            </span>
                            <span className="text-[10px] text-slate-400">{session.date}</span>
                          </div>
                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleStartEdit(session.id, session.title);
                              }}
                              className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-600 transition-colors"
                              title="Edit title"
                            >
                              <Edit size={12} />
                            </button>
                            <button
                              onClick={(e) => handleDeleteSession(session.id, e)}
                              className="p-1.5 rounded-lg hover:bg-red-50 text-slate-400 hover:text-red-500 transition-colors"
                              title="Delete session"
                            >
                              <Trash2 size={12} />
                            </button>
                          </div>
                        </div>
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-200">
           <div className="flex items-center gap-3 px-2">
            <div className="w-9 h-9 rounded-full bg-slate-800 flex items-center justify-center text-white text-xs font-black">BM</div>
            <span className="text-xs font-bold text-slate-800">Biotech Master</span>
          </div>
        </div>
      </aside>

      {/* Main Analysis Thread */}
      <main
        className="flex-1 flex flex-col relative bg-white overflow-hidden"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Drag and Drop Overlay */}
        {isDragging && (
          <div className="absolute inset-0 bg-blue-500/10 backdrop-blur-sm z-50 flex items-center justify-center border-4 border-dashed border-blue-500 m-4 rounded-3xl animate-in fade-in duration-200">
            <div className="text-center">
              <div className="w-20 h-20 bg-blue-500 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-xl shadow-blue-200">
                <FileUp size={40} className="text-white" />
              </div>
              <h3 className="text-xl font-bold text-slate-800 mb-2">Drop to Upload H5AD File</h3>
              <p className="text-slate-500">Release to upload your single-cell data</p>
            </div>
          </div>
        )}

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">
            
            {messages.length === 0 && !state.isProcessing && (
              <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-8 animate-in fade-in duration-700">
                <div className="w-20 h-20 bg-blue-600 rounded-[2.5rem] flex items-center justify-center shadow-2xl shadow-blue-100">
                  <Brain className="text-white w-10 h-10" />
                </div>
                <div className="max-w-xl">
                  <h2 className="text-3xl font-black text-slate-900 mb-3 tracking-tight">How can CellMind assist your research today?</h2>
                  <p className="text-slate-500 text-base leading-relaxed font-medium">
                    Analyze, cluster, and interpret single-cell datasets with our Multi-Agent RAG system. 
                    Upload your matrix or start with a research hypothesis.
                  </p>
                </div>
              </div>
            )}

            {messages.map((m, idx) => (
              <div key={m.id} className={`flex gap-6 ${m.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-10 h-10 rounded-xl flex-shrink-0 flex items-center justify-center font-bold ${m.role === 'user' ? 'bg-slate-100 text-slate-600' : 'bg-blue-600 text-white shadow-xl shadow-blue-100'}`}>
                  {m.role === 'user' ? 'BM' : <Brain size={22} />}
                </div>
                <div className={`flex flex-col max-w-[85%] space-y-6 ${m.role === 'user' ? 'items-end' : 'items-start'}`}>
                  <div className="flex items-start gap-3 group">
                    <div className={`text-base leading-relaxed ${m.role === 'user' ? 'bg-slate-50 px-6 py-4 rounded-3xl border border-slate-100 text-slate-800' : 'text-slate-800 font-medium'}`}>
                      {m.content}
                    </div>
                    {m.role === 'assistant' && (
                      <div className="relative" ref={exportMenuRef}>
                        <button
                          onClick={() => setShowExportMenu(showExportMenu === m.id ? null : m.id)}
                          className="opacity-0 group-hover:opacity-100 p-2 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-all"
                          title="Export report"
                        >
                          <Download size={16} />
                        </button>
                        {showExportMenu === m.id && (
                          <div className="absolute top-8 left-0 w-40 bg-white border border-slate-200 rounded-xl shadow-xl z-50 animate-in fade-in slide-in-from-top-2">
                            <button
                              onClick={() => handleExport('markdown')}
                              className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 rounded-t-xl text-slate-700 transition-colors text-sm"
                            >
                              <FileText size={14} />
                              Markdown (.md)
                            </button>
                            <button
                              onClick={() => handleExport('html')}
                              className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 text-slate-700 transition-colors text-sm"
                            >
                              <FileText size={14} />
                              HTML (.html)
                            </button>
                            <button
                              onClick={() => handleExport('pdf')}
                              className="w-full flex items-center gap-2 px-3 py-2.5 hover:bg-slate-50 rounded-b-xl text-slate-700 transition-colors text-sm"
                            >
                              <FileText size={14} />
                              PDF (Print)
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  {m.role === 'assistant' && state.steps.length > 0 && idx === messages.length - 1 && <AgentStatus steps={state.steps} />}
                  {m.role === 'assistant' && state.clusters.length > 0 && idx === messages.length - 1 && <UmapVisualization clusters={state.clusters} onSelectCluster={setSelectedCluster} />}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* New Input UI Design */}
        <div className="p-8 bg-white border-t border-slate-100 relative z-20">
          <div className="max-w-3xl mx-auto flex flex-col gap-3">
            
            {/* Input Form Body */}
            <div className="relative">
              {/* File Attachment Pill */}
              {uploadedFile && (
                <div className="absolute -top-12 left-0 flex items-center gap-2 px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg border border-blue-100 shadow-sm animate-in slide-in-from-bottom-2 duration-300">
                  <FileText size={14} />
                  <span className="text-xs font-bold truncate max-w-[150px]">{uploadedFile.name}</span>
                  <button onClick={() => setUploadedFile(null)} className="hover:text-red-500">
                    <X size={14} />
                  </button>
                </div>
              )}

              <form onSubmit={handleStartAnalysis} className="bg-slate-50 rounded-[2.5rem] border border-slate-200 p-2.5 flex items-center gap-2 shadow-xl shadow-slate-200/40">
                {/* Plus Button with Dropdown */}
                <div className="relative" ref={menuRef}>
                  <button 
                    type="button" 
                    onClick={() => setShowPlusMenu(!showPlusMenu)}
                    className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${showPlusMenu ? 'bg-slate-200 rotate-45' : 'bg-white hover:bg-slate-100 shadow-sm text-slate-600'}`}
                  >
                    <Plus size={24} />
                  </button>
                  
                  {showPlusMenu && (
                    <div className="absolute bottom-full left-0 mb-4 w-56 bg-white border border-slate-200 rounded-2xl shadow-2xl p-2 z-50 animate-in fade-in slide-in-from-bottom-2">
                      <button 
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full flex items-center gap-3 px-3 py-3 hover:bg-slate-50 rounded-xl text-slate-700 transition-colors"
                      >
                        <FileUp size={18} className="text-blue-500" />
                        <span className="text-sm font-semibold">Add H5AD File</span>
                      </button>
                      <button 
                        type="button"
                        onClick={() => { setIsAgentMode(!isAgentMode); setShowPlusMenu(false); }}
                        className="w-full flex items-center gap-3 px-3 py-3 hover:bg-slate-50 rounded-xl text-slate-700 transition-colors"
                      >
                        <Zap size={18} className={isAgentMode ? "text-amber-500" : "text-slate-400"} />
                        <span className="text-sm font-semibold">{isAgentMode ? "Disable Agent Mode" : "Agent Mode"}</span>
                      </button>
                    </div>
                  )}
                </div>

                <input 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={isAgentMode ? "Describe analysis intent..." : "Ask CellMind anything..."}
                  className="flex-1 bg-transparent py-4 px-3 outline-none text-base font-medium text-slate-800 placeholder:text-slate-400"
                />

                <button 
                  type="submit"
                  disabled={!input.trim() || state.isProcessing}
                  className={`w-12 h-12 rounded-full flex items-center justify-center transition-all shadow-md ${input.trim() ? 'bg-blue-600 text-white hover:scale-105' : 'bg-slate-200 text-slate-400 cursor-not-allowed'}`}
                >
                  <Send size={20} />
                </button>
              </form>
            </div>

            {/* Agent Mode Active Status Label */}
            {isAgentMode && (
              <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100 rounded-full w-fit mx-auto shadow-sm animate-in fade-in duration-500">
                <div className="flex items-center gap-1.5 text-blue-700">
                  <Sparkles size={14} className="animate-pulse" />
                  <span className="text-[10px] font-black uppercase tracking-widest">Agent Mode Active</span>
                </div>
                <div className="h-3 w-[1px] bg-blue-200 mx-1" />
                <span className="text-[10px] font-bold text-slate-500">Upload data to analyze</span>
                <button onClick={() => setIsAgentMode(false)} className="ml-2 hover:text-blue-700">
                  <X size={12} />
                </button>
              </div>
            )}

            <input type="file" ref={fileInputRef} className="hidden" accept=".h5ad" onChange={handleFileChange} />
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
