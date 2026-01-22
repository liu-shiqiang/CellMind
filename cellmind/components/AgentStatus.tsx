
import React from 'react';
import { AnalysisStep, AgentRole } from '../types';
import { CheckCircle, Clock, Loader2, AlertCircle, Bot, Activity, Brain, FileText } from 'lucide-react';

interface Props {
  steps: AnalysisStep[];
}

const RoleIcon = ({ role }: { role: AgentRole }) => {
  switch (role) {
    case AgentRole.PLANNER: return <Activity className="w-4 h-4 text-blue-500" />;
    case AgentRole.EXECUTOR: return <Brain className="w-4 h-4 text-purple-500" />;
    case AgentRole.REFLECTION: return <Loader2 className="w-4 h-4 text-orange-500" />;
    case AgentRole.INTERPRETER: return <FileText className="w-4 h-4 text-emerald-500" />;
    default: return <Bot className="w-4 h-4" />;
  }
};

export const AgentStatus: React.FC<Props> = ({ steps }) => {
  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-bold flex items-center gap-2">
          <Bot className="text-blue-600" />
          Multi-Agent Workflow
        </h2>
        <span className="text-xs font-medium text-slate-400 bg-slate-100 px-2 py-1 rounded-full">RAG-Sync Active</span>
      </div>
      
      <div className="space-y-4">
        {steps.map((step, idx) => (
          <div key={step.id} className="relative pl-8">
            {idx !== steps.length - 1 && (
              <div className="absolute left-3.5 top-7 w-[2px] h-full bg-slate-100" />
            )}
            <div className="absolute left-0 top-1 p-1 bg-white border border-slate-200 rounded-full z-10 shadow-sm">
              {step.status === 'completed' ? <CheckCircle className="w-5 h-5 text-green-500" /> :
               step.status === 'running' ? <Loader2 className="w-5 h-5 text-blue-500 animate-spin" /> :
               step.status === 'failed' ? <AlertCircle className="w-5 h-5 text-red-500" /> :
               <Clock className="w-5 h-5 text-slate-300" />}
            </div>
            
            <div className={`p-4 rounded-xl border transition-all ${step.status === 'running' ? 'bg-blue-50 border-blue-200 scale-[1.02]' : 'bg-slate-50 border-slate-100'}`}>
              <div className="flex items-center gap-2 mb-1">
                <RoleIcon role={step.role} />
                <span className="text-xs font-bold uppercase tracking-tighter text-slate-500">{step.role}</span>
              </div>
              <p className="text-sm font-medium text-slate-700">{step.task}</p>
              {step.output && (
                <div className="mt-2 text-xs text-slate-500 bg-white/50 p-2 rounded border border-slate-200 italic">
                  {step.output}
                </div>
              )}
            </div>
          </div>
        ))}

        {steps.length === 0 && (
          <div className="text-center py-12 text-slate-400">
            <Activity className="w-12 h-12 mx-auto mb-3 opacity-20" />
            <p className="text-sm">Initiate an analysis goal to start the multi-agent system.</p>
          </div>
        )}
      </div>
    </div>
  );
};
