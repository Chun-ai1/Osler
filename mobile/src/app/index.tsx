/**
 * Patients screen — mirrors the web demo's left-column roster (.pt cards).
 * Lists analyzed patients; tapping one selects it and jumps to Reasoning.
 */
import { useRouter } from 'expo-router';
import React from 'react';
import {
  FlatList,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { useApp, type Patient } from '@/state/AppContext';
import { colors, colorForSafety, fonts, radius, shadow, spacing } from '@/theme/tokens';

export default function PatientsScreen() {
  const router = useRouter();
  const { patients, currentId, selectPatient } = useApp();

  const openPatient = (id: string) => {
    selectPatient(id);
    router.push('/reasoning');
  };

  return (
    <View style={styles.screen}>
      <FlatList
        data={patients}
        keyExtractor={(p) => p.id}
        contentContainerStyle={styles.listContent}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.logo}>
              Osl<Text style={styles.logoOrange}>ian</Text>
            </Text>
            <Text style={styles.tagline}>Drug Recommendation Agent</Text>
            {patients.length > 0 && (
              <Text style={styles.sectionLabel}>ANALYZED PATIENTS</Text>
            )}
          </View>
        }
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyTitle}>No patients yet</Text>
            <Text style={styles.emptyText}>
              Import a case to get drug recommendations with full mechanistic
              reasoning.
            </Text>
            <Pressable
              accessibilityRole="button"
              onPress={() => router.push('/analyze')}
              style={({ pressed }) => [styles.pillBtn, pressed && { opacity: 0.85 }]}>
              <Text style={styles.pillBtnText}>＋ Analyze a case</Text>
            </Pressable>
          </View>
        }
        renderItem={({ item }) => (
          <PatientCard
            patient={item}
            active={item.id === currentId}
            onPress={() => openPatient(item.id)}
          />
        )}
      />
    </View>
  );
}

function PatientCard({
  patient,
  active,
  onPress,
}: {
  patient: Patient;
  active: boolean;
  onPress: () => void;
}) {
  const { result } = patient.bundle;
  const n = result.candidates.length;
  const age = result.patient.age != null ? `${result.patient.age}y` : null;
  const sex = result.patient.sex ?? null;
  const demo = [age, sex].filter(Boolean).join(' · ');
  const topDecision = result.candidates[0]?.safety.decision;

  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ selected: active }}
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        active && styles.cardActive,
        pressed && { opacity: 0.9 },
      ]}>
      <View style={[styles.dot, { backgroundColor: colorForSafety(topDecision) }]} />
      <View style={styles.cardBody}>
        <Text style={styles.cardTitle} numberOfLines={1}>
          {patient.title}
        </Text>
        <Text style={styles.cardSub} numberOfLines={1}>
          {result.indication_label ?? result.indication ?? 'Case'}
          {demo ? ` · ${demo}` : ''}
        </Text>
      </View>
      <Text style={styles.cardCount}>
        {n} drug{n === 1 ? '' : 's'}
      </Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  listContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
    flexGrow: 1,
  },
  header: {
    marginBottom: spacing.md,
  },
  logo: {
    fontFamily: fonts.headingBold,
    fontSize: 28,
    color: colors.accent,
    letterSpacing: -0.5,
  },
  logoOrange: {
    color: colors.orange,
  },
  tagline: {
    fontFamily: fonts.body,
    fontSize: 13,
    color: colors.textMuted,
    marginTop: 2,
  },
  sectionLabel: {
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.4,
    color: colors.textMuted,
    marginTop: spacing.xl,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
  },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.card,
    paddingHorizontal: 14,
    paddingVertical: 13,
    marginBottom: spacing.sm,
    minHeight: 60,
    ...shadow.sm,
  },
  cardActive: {
    borderColor: colors.accent,
    shadowColor: colors.accent,
    shadowOpacity: 0.15,
    shadowRadius: 4,
    shadowOffset: { width: 0, height: 0 },
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  cardBody: {
    flex: 1,
  },
  cardTitle: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14.5,
    color: colors.accent,
  },
  cardSub: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.textMuted,
    marginTop: 2,
  },
  cardCount: {
    fontFamily: fonts.bodyMedium,
    fontSize: 11.5,
    color: colors.accent,
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
    overflow: 'hidden',
  },
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    paddingBottom: 60,
  },
  emptyTitle: {
    fontFamily: fonts.heading,
    fontSize: 18,
    color: colors.accent,
    marginBottom: spacing.sm,
  },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    lineHeight: 21,
    color: colors.textSecondary,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  pillBtn: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 26,
    paddingVertical: 13,
    minHeight: 48,
    justifyContent: 'center',
    ...shadow.md,
  },
  pillBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14.5,
    color: '#FFFFFF',
  },
});
