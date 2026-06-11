/**
 * App-wide state: analyzed patients (bundles), current selection, LLM settings.
 *
 * API keys are persisted with expo-secure-store; this context only holds a
 * runtime copy. Patients live in memory only — a demo session starts fresh,
 * matching the web demo's behavior.
 */
import * as SecureStore from 'expo-secure-store';
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';

import type { Bundle, ChatMessage, LlmSettings } from '@/api/osler';

const KEY_API_KEY = 'oslian.llm.apiKey';
const KEY_PROVIDER = 'oslian.llm.provider';

export interface Patient {
  id: string;
  title: string;
  bundle: Bundle;
  chat: ChatMessage[];
}

interface AppState {
  patients: Patient[];
  currentId: string | null;
  current: Patient | null;
  addPatient: (bundle: Bundle, title: string) => void;
  selectPatient: (id: string) => void;
  appendChat: (id: string, msg: ChatMessage) => void;
  llm: LlmSettings;
  setLlm: (next: LlmSettings) => Promise<void>;
  /** highlight target for Reasoning → Drugs (set by graph node taps) */
  highlightDrug: string | null;
  setHighlightDrug: (drug: string | null) => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [llm, setLlmState] = useState<LlmSettings>({});
  const [highlightDrug, setHighlightDrug] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [apiKey, provider] = await Promise.all([
          SecureStore.getItemAsync(KEY_API_KEY),
          SecureStore.getItemAsync(KEY_PROVIDER),
        ]);
        setLlmState({
          apiKey: apiKey ?? undefined,
          provider: (provider as LlmSettings['provider']) ?? 'openai',
        });
      } catch {
        setLlmState({ provider: 'openai' });
      }
    })();
  }, []);

  const setLlm = useCallback(async (next: LlmSettings) => {
    setLlmState(next);
    try {
      if (next.apiKey) await SecureStore.setItemAsync(KEY_API_KEY, next.apiKey);
      else await SecureStore.deleteItemAsync(KEY_API_KEY);
      if (next.provider)
        await SecureStore.setItemAsync(KEY_PROVIDER, next.provider);
    } catch {
      // demo-grade: persistence failure is non-fatal, runtime copy still works
    }
  }, []);

  const addPatient = useCallback((bundle: Bundle, title: string) => {
    setPatients((prev) => {
      const id = bundle.patient_id || `p${prev.length + 1}`;
      const existing = prev.findIndex((p) => p.id === id);
      const patient: Patient = {
        id,
        title,
        bundle,
        chat: existing >= 0 ? prev[existing].chat : [],
      };
      const next =
        existing >= 0
          ? prev.map((p, i) => (i === existing ? patient : p))
          : [...prev, patient];
      return next;
    });
    setCurrentId(bundle.patient_id);
  }, []);

  const selectPatient = useCallback((id: string) => setCurrentId(id), []);

  const appendChat = useCallback((id: string, msg: ChatMessage) => {
    setPatients((prev) =>
      prev.map((p) => (p.id === id ? { ...p, chat: [...p.chat, msg] } : p)),
    );
  }, []);

  const current = useMemo(
    () => patients.find((p) => p.id === currentId) ?? null,
    [patients, currentId],
  );

  const value = useMemo(
    () => ({
      patients,
      currentId,
      current,
      addPatient,
      selectPatient,
      appendChat,
      llm,
      setLlm,
      highlightDrug,
      setHighlightDrug,
    }),
    [patients, currentId, current, addPatient, selectPatient, appendChat, llm, setLlm, highlightDrug],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp must be used inside <AppProvider>');
  return v;
}
