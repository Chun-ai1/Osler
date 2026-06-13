/**
 * Settings — LLM provider/key (persisted via expo-secure-store through
 * AppContext.setLlm), backend info, and an About blurb.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { API_BASE, type LlmSettings } from '@/api/osler';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

type Provider = NonNullable<LlmSettings['provider']>;

const PROVIDERS: { value: Provider; label: string }[] = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Gemini' },
];

export default function SettingsScreen() {
  const { llm, setLlm } = useApp();
  const [provider, setProvider] = useState<Provider>(llm.provider ?? 'openai');
  const [apiKey, setApiKey] = useState(llm.apiKey ?? '');
  const [saved, setSaved] = useState(false);
  const savedTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync local form once the persisted settings finish loading.
  useEffect(() => {
    setProvider(llm.provider ?? 'openai');
    setApiKey(llm.apiKey ?? '');
  }, [llm.provider, llm.apiKey]);

  useEffect(
    () => () => {
      if (savedTimer.current) clearTimeout(savedTimer.current);
    },
    [],
  );

  const save = async () => {
    const trimmed = apiKey.trim();
    await setLlm({ apiKey: trimmed || undefined, provider });
    setSaved(true);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    savedTimer.current = setTimeout(() => setSaved(false), 2000);
  };

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView
        style={styles.flex}
        contentContainerStyle={styles.content}
        keyboardShouldPersistTaps="handled">
        {/* ── LLM ─────────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>LLM (FOR CHAT & FREE-TEXT PARSING)</Text>
        <View style={styles.card}>
          <View style={styles.toggleRow}>
            {PROVIDERS.map((p) => {
              const active = provider === p.value;
              return (
                <Pressable
                  key={p.value}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  onPress={() => setProvider(p.value)}
                  style={({ pressed }) => [
                    styles.toggle,
                    active && styles.toggleActive,
                    pressed && { opacity: 0.8 },
                  ]}>
                  <Text style={[styles.toggleText, active && styles.toggleTextActive]}>
                    {p.label}
                  </Text>
                </Pressable>
              );
            })}
          </View>
          <TextInput
            style={styles.input}
            value={apiKey}
            onChangeText={setApiKey}
            placeholder="sk-..."
            placeholderTextColor={colors.textMuted}
            secureTextEntry
            autoCapitalize="none"
            autoCorrect={false}
            accessibilityLabel="API key"
          />
          <View style={styles.saveRow}>
            <Pressable
              accessibilityRole="button"
              onPress={save}
              style={({ pressed }) => [styles.saveButton, pressed && { opacity: 0.85 }]}>
              <Text style={styles.saveButtonText}>Save</Text>
            </Pressable>
            {saved && <Text style={styles.savedText}>Saved ✓</Text>}
          </View>
          <Text style={styles.note}>
            Stored securely on-device (expo-secure-store). Without a key the demo
            still works: rule-based parsing + symbolic engine; only chat falls back
            to a notice.
          </Text>
        </View>

        {/* ── Backend ─────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>BACKEND</Text>
        <View style={styles.card}>
          <Text style={styles.apiBase}>{API_BASE}</Text>
          <Text style={styles.note}>
            iOS simulator shares the Mac&apos;s localhost. Start the backend with:{' '}
            <Text style={styles.code}>python demo/demo_app.py</Text> (from the Osler
            repo root, branch demo-rx). For a physical device set{' '}
            <Text style={styles.code}>EXPO_PUBLIC_API_BASE</Text> to your Mac&apos;s
            LAN IP.
          </Text>
        </View>

        {/* ── About ───────────────────────────────────────────── */}
        <Text style={styles.sectionLabel}>ABOUT</Text>
        <View style={styles.card}>
          <Text style={styles.aboutTitle}>
            Oslian<Text style={{ color: colors.orange }}>·Rx</Text>
          </Text>
          <Text style={styles.note}>
            Oslian·Rx — clinician-facing drug recommendation demo. The symbolic
            engine makes every recommendation; the LLM only parses cases and
            explains results. Decision support only — not a clinically validated
            tool.
          </Text>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  content: {
    padding: spacing.lg,
    paddingBottom: 48,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.6,
    color: colors.textMuted,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    padding: spacing.lg,
    gap: spacing.md,
    ...shadow.sm,
  },
  toggleRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  toggle: {
    flex: 1,
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleActive: {
    backgroundColor: colors.accent,
  },
  toggleText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.accent,
  },
  toggleTextActive: {
    color: '#FFFFFF',
  },
  input: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    paddingHorizontal: 14,
    minHeight: 44,
    paddingVertical: 0,
  },
  saveRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  saveButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 28,
    minHeight: 44,
    justifyContent: 'center',
  },
  saveButtonText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  savedText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    color: colors.green,
  },
  note: {
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 18,
    color: colors.textSecondary,
  },
  code: {
    fontFamily: fonts.bodyMedium,
    backgroundColor: colors.accentLight,
    color: colors.accent,
  },
  apiBase: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: colors.text,
  },
  aboutTitle: {
    fontFamily: fonts.headingBold,
    fontSize: 17,
    color: colors.accent,
  },
});
