import axios from "axios";

export interface QueryRequest {
  question: string;
  lang?: string;
}

export interface QueryResponse {
  answer: string;
  confidence: number;
  forms: string[];
  sources: string[];
  meta?: Record<string, unknown>;
}

const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000";

export async function queryBackend(payload: QueryRequest): Promise<QueryResponse> {
  const response = await axios.post(`${API_BASE_URL}/query`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  return response.data as QueryResponse;
}
