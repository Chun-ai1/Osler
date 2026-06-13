/**
 * DrugCard — one card per DrugCandidate, optimized for clinical readability.
 * White card with a 4px left border in the safety color; the final_answer is
 * the prominent element clinicians read first. Mechanism details collapse.
 */
import Ionicons from '@expo/vector-icons/Ionicons';
import React, { useState } from 'react';
import {
  LayoutAnimation,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  UIManager,
  View,
} from 'react-native';

import type { DiseaseModel, DrugCandidate } from '@/api/osler';
import DrugSources from '@/components/DrugSources';
import { colorForSafety, colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

if (
  Platform.OS === 'android' &&
  UIManager.setLayoutAnimationEnabledExperimental
) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

interface Props {
  candidate: DrugCandidate;
  rank: number;
  /** when true, paint the highlight pulse (accent-light bg + accent border) */
  highlighted?: boolean;
  /** result.mechanism_only — no clinical labels loaded at all */
  mechanismOnly?: boolean;
  /** disease world-model, for the disease-source citation */
  disease?: DiseaseModel | null;
}

export default function DrugCard({ candidate: c, rank, highlighted, mechanismOnly, disease }: Props) {
  const [showMech, setShowMech] = useState(false);
  const safety = colorForSafety(c.safety?.decision);
  const reasons = (c.safety?.reasons ?? []).filter((r) => r.message);
  const isBad = ['avoid', 'block'].includes(
    (c.safety?.decision ?? '').toLowerCase(),
  );

  const toggleMech = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setShowMech((s) => !s);
  };

  return (
    <View
      style={[
        styles.card,
        { borderLeftColor: safety },
        highlighted && styles.cardHighlighted,
      ]}>
      {/* Row 1: rank + name + safety badge */}
      <View style={styles.headerRow}>
        <Text style={styles.rank}>{rank}</Text>
        <Text style={styles.name} numberOfLines={2}>
          {c.drug}
        </Text>
        <View style={[styles.badge, { backgroundColor: safety }]}>
          <Text style={styles.badgeText}>{c.safety?.decision ?? '—'}</Text>
        </View>
      </View>

      {!!c.clinical_role?.label && (
        <Text style={styles.role}>{c.clinical_role.label}</Text>
      )}

      {!!c.final_answer && (
        <Text style={styles.finalAnswer}>{c.final_answer}</Text>
      )}

      {c.dose?.verbatim && c.dose.patient_specific_allowed ? (
        <View style={styles.doseBox}>
          <Text style={styles.doseLabel}>DOSE (CONFIRM WITH CLINICIAN)</Text>
          <Text style={styles.doseText}>{c.dose.verbatim}</Text>
        </View>
      ) : (
        // Mirrors the web demo's `.dose.blocked` box: the safety gate withheld
        // patient-specific dosing — say so explicitly instead of hiding it.
        <View style={styles.doseBoxBlocked}>
          <Text style={styles.doseLabelBlocked}>DOSE</Text>
          <Text style={styles.doseTextBlocked}>
            not shown —{' '}
            {mechanismOnly
              ? 'no label loaded'
              : 'blocked by safety gate / validation'}
          </Text>
        </View>
      )}

      {!!c.rationale && <Text style={styles.rationale}>{c.rationale}</Text>}

      {reasons.length > 0 && (
        <View style={styles.reasons}>
          {reasons.map((r, i) => (
            <View key={i} style={styles.reasonRow}>
              <Text
                style={[
                  styles.reasonIcon,
                  { color: isBad ? colors.red : colors.amber },
                ]}>
                ⚠
              </Text>
              <Text
                style={[
                  styles.reasonText,
                  { color: isBad ? colors.red : colors.amber },
                ]}>
                {r.message}
              </Text>
            </View>
          ))}
        </View>
      )}

      {/* Collapsible mechanism details */}
      <Pressable
        onPress={toggleMech}
        style={styles.mechToggle}
        hitSlop={6}
        accessibilityRole="button"
        accessibilityLabel="Toggle mechanism details">
        <Ionicons
          name={showMech ? 'chevron-down' : 'chevron-forward'}
          size={14}
          color={colors.textSecondary}
        />
        <Text style={styles.mechToggleText}>Mechanism details</Text>
        {typeof c.mechanism_score === 'number' && (
          <Text style={styles.mechScore}>
            score {c.mechanism_score.toFixed(2)}
          </Text>
        )}
      </Pressable>

      {showMech && (
        <View style={styles.mechBody}>
          {!!c.mechanism_chain && (
            <Text style={styles.mechChain}>{c.mechanism_chain}</Text>
          )}
          {!!c.matched_targets?.length && (
            <View style={styles.targetWrap}>
              {c.matched_targets.map((t, i) => (
                <View key={`${t.target}-${i}`} style={styles.targetPill}>
                  <Text style={styles.targetText}>
                    {t.target}
                    {t.effect_type ? ` · ${t.effect_type}` : ''}
                  </Text>
                </View>
              ))}
            </View>
          )}
        </View>
      )}

      <DrugSources candidate={c} disease={disease} />
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.bgCard,
    borderRadius: radius.card,
    borderLeftWidth: 4,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderSolid,
    padding: spacing.lg,
    marginBottom: spacing.md,
    ...shadow.sm,
  },
  cardHighlighted: {
    backgroundColor: colors.accentLight,
    borderWidth: 1.5,
    borderColor: colors.accent,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  rank: {
    fontFamily: fonts.headingBold,
    fontSize: 14,
    color: colors.silver,
    minWidth: 18,
  },
  name: {
    flex: 1,
    fontFamily: fonts.headingBold,
    fontSize: 18,
    color: colors.text,
    textTransform: 'capitalize',
  },
  badge: {
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  badgeText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 10,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
    color: '#FFFFFF',
  },
  role: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.textMuted,
    marginTop: 2,
    marginLeft: 18 + spacing.sm,
  },
  finalAnswer: {
    fontFamily: fonts.bodyMedium,
    fontSize: 15,
    lineHeight: 22,
    color: colors.text,
    marginTop: spacing.md,
  },
  // green dose box mirrors web `.dose` (#F0FDF4 bg / #BBF7D0 border)
  doseBox: {
    backgroundColor: '#F0FDF4',
    borderWidth: 1,
    borderColor: '#BBF7D0',
    borderRadius: radius.sm,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  doseLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 10,
    letterSpacing: 0.8,
    color: colors.green,
    marginBottom: spacing.xs,
  },
  doseText: {
    fontFamily: Platform.select({ ios: 'Menlo', android: 'monospace' }),
    fontSize: 15,
    lineHeight: 21,
    color: colors.text,
  },
  doseBoxBlocked: {
    backgroundColor: colors.bgWarm,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.sm,
    padding: spacing.md,
    marginTop: spacing.md,
  },
  doseLabelBlocked: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 10,
    letterSpacing: 0.8,
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  doseTextBlocked: {
    fontFamily: fonts.body,
    fontSize: 13,
    color: colors.textMuted,
  },
  rationale: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 21,
    color: colors.textSecondary,
    marginTop: spacing.sm,
  },
  reasons: {
    marginTop: spacing.sm,
    gap: spacing.xs,
  },
  reasonRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 6,
  },
  reasonIcon: {
    fontSize: 13,
    lineHeight: 19,
  },
  reasonText: {
    flex: 1,
    fontFamily: fonts.bodyMedium,
    fontSize: 13,
    lineHeight: 19,
  },
  mechToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.md,
    paddingTop: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.borderSolid,
  },
  mechToggleText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 12.5,
    color: colors.textSecondary,
  },
  mechScore: {
    marginLeft: 'auto',
    fontFamily: fonts.body,
    fontSize: 11.5,
    color: colors.textMuted,
  },
  mechBody: {
    marginTop: spacing.sm,
    gap: spacing.sm,
  },
  mechChain: {
    fontFamily: fonts.body,
    fontSize: 13,
    lineHeight: 19,
    color: colors.textSecondary,
  },
  targetWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  targetPill: {
    backgroundColor: colors.bgWarm,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  targetText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 11.5,
    color: colors.teal,
  },
});
