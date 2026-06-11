/**
 * API client for the Osler·Rx demo backend (demo/demo_app.py, Flask, port 5000).
 *
 * iOS simulator shares the Mac's localhost, so the default base works as-is.
 * On a physical device set EXPO_PUBLIC_API_BASE to the Mac's LAN IP
 * (ATS exceptions for local networking are pre-configured in app.json).
 */

// 127.0.0.1 (not "localhost"): macOS AirPlay Receiver squats on *:5000 and iOS
// resolves localhost to ::1 first, which would hit AirPlay instead of Flask.
export const API_BASE =
  process.env.EXPO_PUBLIC_API_BASE ?? 'http://127.0.0.1:5000';

// ── Types mirroring demo/agent.py bundle ─────────────────────────────────

export interface SampleCase {
  id: string;
  title: string;
  indication: string;
  age?: number;
  sex?: string;
  weight_kg?: number;
  egfr?: number;
  hepatic_status?: string;
  allergies?: string[];
  current_medications?: string[];
  conditions?: string[];
  symptoms?: string[];
  vitals?: Record<string, number>;
  labs?: Record<string, number>;
}

export interface Indication {
  value: string;
  label: string;
}

export interface CasesResponse {
  cases: SampleCase[];
  indications: Indication[];
  env_llm: boolean;
}

export interface MatchedTarget {
  target: string;
  effect_type: string;
}

export interface DrugCandidate {
  drug: string;
  clinical_role: { label?: string; role?: string };
  mechanism_score?: number;
  mechanism_chain?: string;
  matched_targets?: MatchedTarget[];
  safety: { decision: string; reasons?: { message?: string }[] };
  dose?: { verbatim?: string; patient_specific_allowed?: boolean };
  final_answer?: string;
  rationale?: string;
}

export interface AnalyzeResult {
  patient: {
    age?: number;
    sex?: string;
    renal_label?: string;
    allergies?: string[];
    meds?: string[];
    flags?: string[];
  };
  indication?: string;
  indication_label?: string;
  target_states?: string[];
  candidates: DrugCandidate[];
  mechanism_only?: boolean;
}

export interface Perturbation {
  variable: string;
  direction: 'high' | 'low';
  cause?: string;
}

export interface DiseaseModel {
  source?: string;
  source_file?: string;
  disease?: string;
  description?: string;
  perturbations: Perturbation[];
  symptoms: string[];
}

export interface GraphNode {
  id: string;
  label: string;
  group: string;
  title?: string;
}

export interface GraphEdge {
  from: string;
  to: string;
  label?: string;
  dashes?: boolean;
  color?: string;
}

export interface TraceStep {
  icon: string;
  title: string;
  detail?: string;
}

export interface Bundle {
  indication: string;
  parser: string;
  fields: Record<string, unknown>;
  result: AnalyzeResult;
  disease_model: DiseaseModel;
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
  openfda_loaded?: string[] | null;
  trace: TraceStep[];
  patient_id: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface LlmSettings {
  apiKey?: string;
  provider?: 'openai' | 'gemini';
}

// ── Endpoints ────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) {
    throw new Error((j as { error?: string }).error ?? `HTTP ${r.status}`);
  }
  return j as T;
}

export async function getCases(): Promise<CasesResponse> {
  const r = await fetch(`${API_BASE}/api/cases`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return (await r.json()) as CasesResponse;
}

export interface AnalyzeRequest {
  fields?: Record<string, unknown>;
  text?: string;
  patient_id?: string;
  use_openfda?: boolean;
  llm?: LlmSettings;
}

export async function analyze(req: AnalyzeRequest): Promise<Bundle> {
  return post<Bundle>('/api/analyze', {
    fields: req.fields ?? {},
    text: req.text ?? '',
    patient_id: req.patient_id,
    use_openfda: !!req.use_openfda,
    api_key: req.llm?.apiKey ?? '',
    provider: req.llm?.provider,
  });
}

export async function chat(
  patientId: string,
  messages: ChatMessage[],
  bundle: Bundle | null,
  llm?: LlmSettings,
): Promise<{ reply: string; ok: boolean }> {
  return post('/api/chat', {
    patient_id: patientId,
    messages,
    // Send grounding with the request so chat survives backend restarts
    result: bundle?.result,
    disease_model: bundle?.disease_model,
    api_key: llm?.apiKey ?? '',
    provider: llm?.provider,
  });
}

/**
 * v2 (not used yet): live streaming of /api/analyze_stream (NDJSON, one JSON
 * object per line: {type:"step"|"error"|"final"}). Implement with Expo SDK 52+
 * `expo/fetch`, whose Response.body is a ReadableStream:
 *
 *   import { fetch as expoFetch } from 'expo/fetch';
 *   const res = await expoFetch(`${API_BASE}/api/analyze_stream`, {...});
 *   const reader = res.body.getReader();
 *   // accumulate chunks, split on '\n', JSON.parse each complete line.
 *
 * Do NOT add react-native-sse — expo/fetch covers this natively.
 */
