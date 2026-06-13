/**
 * Analyze screen — mirrors the web demo's import-case box: preset chips,
 * free-text case notes, openFDA toggle, and a navy agent-trace overlay that
 * plays placeholder steps while /api/analyze runs, then the real trace.
 */
import { useRouter } from 'expo-router';
import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import {
  analyze,
  API_BASE,
  getCases,
  type SampleCase,
  type TraceStep,
} from '@/api/osler';
import CasePresetChips from '@/components/CasePresetChips';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

const PLACEHOLDER_STEPS: TraceStep[] = [
  { icon: '🧩', title: 'Parsing case…' },
  { icon: '🎯', title: 'Mapping indication → treatment targets…' },
  { icon: '⚙️', title: 'Symbolic engine ranking drugs…' },
  { icon: '📖', title: 'Loading disease world-model…' },
];

const PLACEHOLDER_MS = 900;
const REAL_STEP_MS = 250;

function summarizeCase(c: SampleCase): string {
  const parts: string[] = [];
  if (c.age != null || c.sex) parts.push([c.age, c.sex].filter((x) => x != null).join(''));
  parts.push(`Indication: ${c.indication}`);
  if (c.weight_kg != null) parts.push(`${c.weight_kg} kg`);
  if (c.egfr != null) parts.push(`eGFR ${c.egfr}`);
  if (c.hepatic_status) parts.push(`Hepatic: ${c.hepatic_status}`);
  if (c.allergies?.length) parts.push(`Allergies: ${c.allergies.join(', ')}`);
  if (c.current_medications?.length) parts.push(`Meds: ${c.current_medications.join(', ')}`);
  if (c.conditions?.length) parts.push(`Conditions: ${c.conditions.join(', ')}`);
  if (c.symptoms?.length) parts.push(`Symptoms: ${c.symptoms.join(', ')}`);
  if (c.vitals && Object.keys(c.vitals).length)
    parts.push(
      `Vitals: ${Object.entries(c.vitals).map(([k, v]) => `${k} ${v}`).join(', ')}`,
    );
  if (c.labs && Object.keys(c.labs).length)
    parts.push(
      `Labs: ${Object.entries(c.labs).map(([k, v]) => `${k} ${v}`).join(', ')}`,
    );
  return parts.join(' · ');
}

export default function AnalyzeScreen() {
  const router = useRouter();
  const { patients, addPatient, llm } = useApp();

  const [cases, setCases] = useState<SampleCase[]>([]);
  const [bannerVisible, setBannerVisible] = useState(false);
  const [selected, setSelected] = useState<SampleCase | null>(null);
  const [text, setText] = useState('');
  const [useOpenFda, setUseOpenFda] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // overlay state
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [busy, setBusy] = useState(false);

  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([]);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    (async () => {
      try {
        const res = await getCases();
        if (!mountedRef.current) return;
        setCases(res.cases);
        setBannerVisible(false);
      } catch {
        if (mountedRef.current) setBannerVisible(true);
      }
    })();
    return () => {
      mountedRef.current = false;
      timersRef.current.forEach(clearTimeout);
      timersRef.current = [];
    };
  }, []);

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
  }, []);

  const onSelectPreset = useCallback((c: SampleCase) => {
    setSelected((prev) => (prev?.id === c.id ? null : c));
  }, []);

  const onAnalyze = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSteps([]);
    setOverlayVisible(true);

    // Play placeholder steps one-by-one while the request runs.
    PLACEHOLDER_STEPS.forEach((step, i) => {
      const t = setTimeout(() => {
        if (mountedRef.current) setSteps((prev) => [...prev, step]);
      }, i * PLACEHOLDER_MS);
      timersRef.current.push(t);
    });

    const { id, ...presetFields } = selected ?? ({} as SampleCase);
    void id;

    try {
      const bundle = await analyze({
        fields: selected ? (presetFields as unknown as Record<string, unknown>) : {},
        text,
        patient_id: selected?.id ?? `case-${patients.length + 1}`,
        use_openfda: useOpenFda,
        llm,
      });
      if (!mountedRef.current) return;

      // Replace placeholders with the real trace, revealed rapidly.
      clearTimers();
      setSteps([]);
      const trace = bundle.trace ?? [];
      trace.forEach((step, i) => {
        const t = setTimeout(() => {
          if (mountedRef.current) setSteps((prev) => [...prev, step]);
        }, i * REAL_STEP_MS);
        timersRef.current.push(t);
      });

      const title =
        selected?.title ??
        (text.trim() ? text.trim().slice(0, 40) : null) ??
        bundle.result.indication_label ??
        'Case';

      const done = setTimeout(() => {
        if (!mountedRef.current) return;
        addPatient(bundle, title);
        setOverlayVisible(false);
        setBusy(false);
        router.push('/reasoning');
      }, trace.length * REAL_STEP_MS + 400);
      timersRef.current.push(done);
    } catch (e) {
      clearTimers();
      if (!mountedRef.current) return;
      setOverlayVisible(false);
      setBusy(false);
      setError(e instanceof Error ? e.message : 'Analyze failed');
    }
  }, [busy, selected, text, useOpenFda, llm, patients.length, addPatient, router, clearTimers]);

  return (
    <KeyboardAvoidingView
      style={styles.screen}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled">
        {bannerVisible && (
          <View style={styles.banner}>
            <Text style={styles.bannerText}>
              Backend not reachable at {API_BASE} — start demo/demo_app.py
            </Text>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel="Dismiss warning"
              onPress={() => setBannerVisible(false)}
              hitSlop={12}
              style={styles.bannerClose}>
              <Text style={styles.bannerCloseText}>✕</Text>
            </Pressable>
          </View>
        )}

        <Text style={styles.sectionLabel}>SAMPLE CASES</Text>
        <CasePresetChips
          cases={cases}
          selectedId={selected?.id ?? null}
          onSelect={onSelectPreset}
        />
        {selected && (
          <View style={styles.summaryCard}>
            <Text style={styles.summaryTitle}>{selected.title}</Text>
            <Text style={styles.summaryText}>{summarizeCase(selected)}</Text>
          </View>
        )}

        <Text style={styles.sectionLabel}>CASE NOTES</Text>
        <TextInput
          style={styles.textarea}
          multiline
          textAlignVertical="top"
          value={text}
          onChangeText={setText}
          placeholder="64M crushing chest pain, acute coronary syndrome. BP 88/54, HR 112, eGFR 72."
          placeholderTextColor={colors.textMuted}
        />

        <View style={styles.switchRow}>
          <Text style={styles.switchLabel}>Fetch live openFDA labels</Text>
          <Switch
            value={useOpenFda}
            onValueChange={setUseOpenFda}
            trackColor={{ false: colors.silverLight, true: colors.accent }}
            thumbColor="#FFFFFF"
          />
        </View>

        {error && <Text style={styles.errorText}>{error}</Text>}

        <Pressable
          accessibilityRole="button"
          onPress={onAnalyze}
          disabled={busy}
          style={({ pressed }) => [
            styles.primaryBtn,
            (pressed || busy) && { opacity: 0.7 },
          ]}>
          <Text style={styles.primaryBtnText}>Analyze case →</Text>
        </Pressable>
      </ScrollView>

      <Modal visible={overlayVisible} transparent animationType="fade">
        <View style={styles.overlayBackdrop}>
          <View style={styles.traceCard}>
            <View style={styles.traceHeader}>
              <Text style={styles.traceHeaderText}>⚡ AGENT WORKFLOW</Text>
              <ActivityIndicator size="small" color={colors.traceText} />
            </View>
            <ScrollView style={styles.traceBody}>
              {steps.map((s, i) => (
                <View key={`${i}-${s.title}`} style={styles.traceStep}>
                  <Text style={styles.traceIcon}>{s.icon || '•'}</Text>
                  <View style={styles.traceStepBody}>
                    <Text style={styles.traceTitle}>{s.title}</Text>
                    {!!s.detail && <Text style={styles.traceDetail}>{s.detail}</Text>}
                  </View>
                </View>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  content: {
    padding: spacing.lg,
    paddingBottom: 40,
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FEF3C7',
    borderWidth: 1,
    borderColor: colors.amber,
    borderRadius: radius.sm,
    padding: spacing.md,
    marginBottom: spacing.lg,
    gap: spacing.sm,
  },
  bannerText: {
    flex: 1,
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.amber,
  },
  bannerClose: {
    minWidth: 28,
    minHeight: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  bannerCloseText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.amber,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.4,
    color: colors.textMuted,
    textTransform: 'uppercase',
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  summaryCard: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    padding: spacing.md,
    marginTop: spacing.md,
    ...shadow.sm,
  },
  summaryTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13.5,
    color: colors.accent,
    marginBottom: 4,
  },
  summaryText: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  textarea: {
    minHeight: 116,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    padding: spacing.md,
    fontFamily: fonts.body,
    fontSize: 13.5,
    lineHeight: 20,
    color: colors.text,
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    minHeight: 44,
    marginTop: spacing.lg,
  },
  switchLabel: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13.5,
    color: colors.textSecondary,
  },
  errorText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.red,
    marginTop: spacing.md,
  },
  primaryBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    minHeight: 50,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: spacing.lg,
    ...shadow.md,
  },
  primaryBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14.5,
    color: '#FFFFFF',
  },
  overlayBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(15,43,91,0.55)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  traceCard: {
    width: '100%',
    maxHeight: '70%',
    backgroundColor: colors.traceBg,
    borderRadius: radius.card,
    padding: spacing.lg,
    ...shadow.md,
  },
  traceHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
  },
  traceHeaderText: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.6,
    color: colors.traceMuted,
  },
  traceBody: {
    flexGrow: 0,
  },
  traceStep: {
    flexDirection: 'row',
    gap: 9,
    paddingVertical: 5,
    alignItems: 'flex-start',
  },
  traceIcon: {
    fontSize: 15,
    lineHeight: 20,
  },
  traceStepBody: {
    flex: 1,
  },
  traceTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13.5,
    color: '#FFFFFF',
  },
  traceDetail: {
    fontFamily: fonts.heading,
    fontSize: 12,
    color: colors.traceMuted,
    marginTop: 1,
  },
});
