import { create } from 'zustand';
import type { AgentRun, AgentStep, StepStatus, AnalysisReport, ViewMode } from '@/types';

interface AgentState {
  run: AgentRun | null;
  isProcessing: boolean;
  currentPhase: string;
  // 报告相关
  report: AnalysisReport | null;
  viewMode: ViewMode;
  isAgentStatusExpanded: boolean;

  // Actions
  startRun: (run: AgentRun) => void;
  updateStep: (stepId: string, updates: Partial<AgentStep>) => void;
  addStep: (step: AgentStep) => void;
  updateStepStatus: (stepId: string, status: StepStatus, output?: string) => void;
  completeRun: (result?: string) => void;
  failRun: (error: string) => void;
  reset: () => void;
  setProcessing: (processing: boolean) => void;
  setCurrentPhase: (phase: string) => void;

  // 报告相关 Actions
  setReport: (report: AnalysisReport | null) => void;
  setViewMode: (mode: ViewMode) => void;
  toggleAgentStatus: () => void;
  setAgentStatusExpanded: (expanded: boolean) => void;
}

export const useAgentStore = create<AgentState>((set) => ({
  run: null,
  isProcessing: false,
  currentPhase: 'Initialization',
  report: null,
  viewMode: 'chat',
  isAgentStatusExpanded: true,

  startRun: (run) =>
    set({
      run,
      isProcessing: true,
      viewMode: 'chat',
      report: null,
      isAgentStatusExpanded: true,
    }),

  updateStep: (stepId, updates) =>
    set((state) => ({
      run: state.run
        ? {
            ...state.run,
            steps: state.run.steps.map((step) =>
              step.id === stepId ? { ...step, ...updates } : step
            ),
          }
        : null,
    })),

  addStep: (step) =>
    set((state) => ({
      run: state.run
        ? { ...state.run, steps: [...state.run.steps, step] }
        : null,
    })),

  updateStepStatus: (stepId, status, output) =>
    set((state) => ({
      run: state.run
        ? {
            ...state.run,
            steps: state.run.steps.map((step) =>
              step.id === stepId
                ? {
                    ...step,
                    status,
                    ...(output && { output }),
                    ...(status === 'running' && { startedAt: new Date() }),
                    ...(status === 'completed' && { completedAt: new Date() }),
                  }
                : step
            ),
          }
        : null,
    })),

  completeRun: (result) =>
    set((state) => {
      // 将所有正在运行的步骤标记为完成
      const updatedSteps = state.run?.steps.map((step) =>
        step.status === 'running'
          ? { ...step, status: 'completed' as const, completedAt: new Date() }
          : step
      ) || [];

      return {
        run: state.run
          ? {
              ...state.run,
              status: 'completed',
              completedAt: new Date(),
              steps: updatedSteps,
            }
          : null,
        isProcessing: false,
        // 分析完成后自动折叠状态栏
        isAgentStatusExpanded: false,
      };
    }),

  failRun: (error) =>
    set((state) => ({
      run: state.run ? { ...state.run, status: 'failed' } : null,
      isProcessing: false,
    })),

  reset: () => set({
    run: null,
    isProcessing: false,
    currentPhase: 'Initialization',
    report: null,
    viewMode: 'chat',
    isAgentStatusExpanded: true,
  }),

  setProcessing: (processing) => set({ isProcessing: processing }),

  setCurrentPhase: (phase) => set({ currentPhase: phase }),

  // 报告相关 Actions
  setReport: (report) => set({ report }),

  setViewMode: (mode) => set({ viewMode: mode }),

  toggleAgentStatus: () =>
    set((state) => ({ isAgentStatusExpanded: !state.isAgentStatusExpanded })),

  setAgentStatusExpanded: (expanded) => set({ isAgentStatusExpanded: expanded }),
}));
