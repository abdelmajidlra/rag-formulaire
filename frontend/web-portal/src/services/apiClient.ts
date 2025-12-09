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

export async function queryIrccApi(payload: QueryRequest): Promise<QueryResponse> {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
  const res = await fetch(`${baseUrl}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  return res.json();
}
