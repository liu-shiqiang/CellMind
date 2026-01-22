
export enum AgentRole {
  PLANNER = 'Planner',
  EXECUTOR = 'Executor',
  REFLECTION = 'Reflection',
  INTERPRETER = 'Interpreter'
}

export interface AnalysisStep {
  id: string;
  role: AgentRole;
  task: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  output?: string;
  toolUsed?: string;
}

export interface CellCluster {
  id: string;
  size: number;
  markers: string[];
  suggestedType?: string;
  embedding: [number, number][];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agentRole?: AgentRole;
  timestamp: Date;
}

export interface AnalysisState {
  isProcessing: boolean;
  steps: AnalysisStep[];
  clusters: CellCluster[];
  currentPhase: 'Initialization' | 'Embedding' | 'Clustering' | 'Annotation' | 'Interpretation';
}
