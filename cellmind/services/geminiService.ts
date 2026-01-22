
import { GoogleGenAI, Type } from "@google/genai";
import { AgentRole, AnalysisStep } from "../types";

const ai = new GoogleGenAI({ apiKey: (process.env.API_KEY as string) });

const SYSTEM_PROMPT = `You are CellMind, a world-class AI orchestration system for Single-Cell RAG and Multi-Agent analysis. 
Your identity is professional, precise, and scientifically authoritative.
You coordinate specialized biological agents:
- Planner: Strategizes the analysis workflow.
- Executor: Handles high-dimensional data tools (scGPT, UMAP).
- Reflection: Ensures analytical consistency and error checking.
- Interpreter: Bridges the gap between data and biological meaning using RAG.
Always refer to yourself as CellMind when appropriate.`;

export async function generateAnalysisPlan(userInput: string): Promise<AnalysisStep[]> {
  const response = await ai.models.generateContent({
    model: "gemini-3-pro-preview",
    contents: `Based on this request: "${userInput}", as CellMind, generate a structured multi-agent plan for single-cell analysis.`,
    config: {
      systemInstruction: SYSTEM_PROMPT,
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            id: { type: Type.STRING },
            role: { type: Type.STRING, description: "One of: Planner, Executor, Reflection, Interpreter" },
            task: { type: Type.STRING },
            status: { type: Type.STRING }
          },
          required: ["id", "role", "task", "status"]
        }
      }
    }
  });

  return JSON.parse(response.text || "[]");
}

export async function interpretResults(context: string, markers: string[]): Promise<string> {
  const response = await ai.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: `Researcher Query: ${context}. Identified Markers: ${markers.join(', ')}. As CellMind, provide a detailed biological interpretation using the integrated Cell Ontology RAG. Use professional biomedical language.`,
    config: {
      systemInstruction: SYSTEM_PROMPT,
      temperature: 0.2
    }
  });
  return response.text || "No interpretation generated.";
}
