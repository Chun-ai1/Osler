/**
 * DiseaseModelPanel — the disease "world model": name, source, description,
 * perturbations (variable ↑/↓ with cause) and symptoms as pills.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

import type { DiseaseModel } from '@/api/osler';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

interface Props {
  model: DiseaseModel;
}

export default function DiseaseModelPanel({ model }: Props) {
  const perturbations = model.perturbations ?? [];
  const symptoms = model.symptoms ?? [];

  return (
    <View style={styles.card}>
      <Text style={styles.disease}>{model.disease ?? 'Disease model'}</Text>
      {(model.source || model.source_file) && (
        <Text style={styles.source}>
          {[model.source, model.source_file].filter(Boolean).join(' · ')}
        </Text>
      )}

      {!!model.description && (
        <Text style={styles.description}>{model.description}</Text>
      )}

      {perturbations.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Perturbations</Text>
          {perturbations.map((p, i) => {
            const up = p.direction === 'high';
            return (
              <View key={`${p.variable}-${i}`} style={styles.perturbRow}>
                <Text
                  style={[
                    styles.arrow,
                    { color: up ? colors.red : colors.teal },
                  ]}>
                  {up ? '↑' : '↓'}
                </Text>
                <View style={styles.perturbBody}>
                  <Text style={styles.variable}>{p.variable}</Text>
                  {!!p.cause && <Text style={styles.cause}>{p.cause}</Text>}
                </View>
              </View>
            );
          })}
        </View>
      )}

      {symptoms.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Symptoms</Text>
          <View style={styles.pillWrap}>
            {symptoms.map((s, i) => (
              <View key={`${s}-${i}`} style={styles.pill}>
                <Text style={styles.pillText}>{s}</Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderSolid,
    padding: spacing.lg,
    ...shadow.sm,
  },
  disease: {
    fontFamily: fonts.headingBold,
    fontSize: 19,
    color: colors.text,
    textTransform: 'capitalize',
  },
  source: {
    fontFamily: fonts.body,
    fontSize: 11.5,
    color: colors.textMuted,
    marginTop: 2,
  },
  description: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 21,
    color: colors.textSecondary,
    marginTop: spacing.md,
  },
  section: {
    marginTop: spacing.lg,
  },
  sectionTitle: {
    fontFamily: fonts.heading,
    fontSize: 13,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    color: colors.silver,
    marginBottom: spacing.sm,
  },
  perturbRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.sm,
    paddingVertical: 6,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.borderSolid,
  },
  arrow: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 16,
    lineHeight: 20,
    width: 18,
    textAlign: 'center',
  },
  perturbBody: {
    flex: 1,
  },
  variable: {
    fontFamily: fonts.bodyMedium,
    fontSize: 14,
    color: colors.text,
  },
  cause: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.textSecondary,
    marginTop: 1,
  },
  pillWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  pill: {
    backgroundColor: colors.bgWarm,
    borderWidth: 1,
    borderColor: colors.silverLight,
    borderRadius: radius.pill,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  pillText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.textSecondary,
  },
});
