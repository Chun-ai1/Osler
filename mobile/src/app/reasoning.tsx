/**
 * Reasoning screen — segmented Drugs | Graph | Disease views over the
 * current patient's analysis bundle (mirrors the web demo's right column).
 *
 * Tapping a drug node in the Graph posts back through GraphWebView →
 * setHighlightDrug(context) → this screen auto-switches to Drugs, scrolls to
 * the card and pulses a highlight for ~2.5s.
 */
import { useRouter } from 'expo-router';
import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  FlatList,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import type { DrugCandidate } from '@/api/osler';
import DiseaseModelPanel from '@/components/DiseaseModel';
import DrugCard from '@/components/DrugCard';
import GraphWebView from '@/components/GraphWebView';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

type Segment = 'drugs' | 'graph' | 'disease';

const SEGMENTS: { key: Segment; label: string }[] = [
  { key: 'drugs', label: 'Drugs' },
  { key: 'graph', label: 'Graph' },
  { key: 'disease', label: 'Disease' },
];

export default function ReasoningScreen() {
  const router = useRouter();
  const { current, highlightDrug, setHighlightDrug } = useApp();
  const [segment, setSegment] = useState<Segment>('drugs');
  const [pulseDrug, setPulseDrug] = useState<string | null>(null);
  const listRef = useRef<FlatList<DrugCandidate>>(null);

  const bundle = current?.bundle ?? null;
  const candidates = useMemo(
    () => bundle?.result?.candidates ?? [],
    [bundle],
  );

  // Graph node tap (or any setHighlightDrug) → jump to Drugs, scroll, pulse.
  useEffect(() => {
    if (!highlightDrug) return;
    const target = highlightDrug.toLowerCase();
    setSegment('drugs');
    setPulseDrug(target);

    const index = candidates.findIndex(
      (c) => c.drug?.toLowerCase() === target,
    );
    // Wait a tick so the Drugs FlatList is mounted before scrolling.
    const scrollTimer = setTimeout(() => {
      if (index >= 0) {
        listRef.current?.scrollToIndex({
          index,
          animated: true,
          viewPosition: 0.1,
        });
      }
    }, 250);

    const clearTimer = setTimeout(() => {
      setPulseDrug(null);
      setHighlightDrug(null);
    }, 2500);

    return () => {
      clearTimeout(scrollTimer);
      clearTimeout(clearTimer);
    };
  }, [highlightDrug, candidates, setHighlightDrug]);

  if (!bundle) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyTitle}>No analysis yet</Text>
        <Text style={styles.emptyText}>Analyze a case first</Text>
        <Pressable
          style={styles.emptyBtn}
          onPress={() => router.push('/analyze')}
          accessibilityRole="button">
          <Text style={styles.emptyBtnText}>Go to Analyze</Text>
        </Pressable>
      </View>
    );
  }

  const patient = bundle.result?.patient ?? {};
  const flags = patient.flags ?? [];
  const demo = [
    patient.age != null ? `${patient.age} y` : null,
    patient.sex || null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <View style={styles.screen}>
      {/* Patient summary strip */}
      <View style={styles.summary}>
        <View style={styles.summaryText}>
          <Text style={styles.summaryTitle} numberOfLines={1}>
            {bundle.result?.indication_label ?? bundle.indication}
          </Text>
          {!!demo && <Text style={styles.summaryDemo}>{demo}</Text>}
        </View>
        {flags.length > 0 && (
          <View style={styles.flagWrap}>
            {flags.map((f, i) => (
              <View key={`${f}-${i}`} style={styles.flagPill}>
                <Text style={styles.flagText}>{f}</Text>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Segmented control */}
      <View style={styles.segmentBar}>
        {SEGMENTS.map((s) => {
          const active = segment === s.key;
          return (
            <Pressable
              key={s.key}
              onPress={() => setSegment(s.key)}
              style={[styles.segmentBtn, active && styles.segmentBtnActive]}
              accessibilityRole="button"
              accessibilityState={{ selected: active }}>
              <Text
                style={[
                  styles.segmentLabel,
                  active && styles.segmentLabelActive,
                ]}>
                {s.label}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* Content */}
      {segment === 'drugs' && (
        <FlatList
          ref={listRef}
          data={candidates}
          keyExtractor={(item, i) => `${item.drug}-${i}`}
          contentContainerStyle={styles.listContent}
          onScrollToIndexFailed={(info) => {
            setTimeout(() => {
              listRef.current?.scrollToIndex({
                index: info.index,
                animated: true,
                viewPosition: 0.1,
              });
            }, 300);
          }}
          renderItem={({ item, index }) => (
            <DrugCard
              candidate={item}
              rank={index + 1}
              highlighted={pulseDrug === item.drug?.toLowerCase()}
              mechanismOnly={!!bundle?.result?.mechanism_only}
            />
          )}
          ListEmptyComponent={
            <Text style={styles.noData}>
              No drug candidates in this analysis.
            </Text>
          }
        />
      )}

      {segment === 'graph' && (
        <View style={styles.graphWrap}>
          <GraphWebView
            graph={bundle.graph}
            onDrugTap={(drug) => setHighlightDrug(drug)}
          />
        </View>
      )}

      {segment === 'disease' && (
        <ScrollView contentContainerStyle={styles.listContent}>
          <DiseaseModelPanel model={bundle.disease_model} />
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  // Empty state
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.bg,
    padding: spacing.xl,
    gap: spacing.sm,
  },
  emptyTitle: {
    fontFamily: fonts.headingBold,
    fontSize: 19,
    color: colors.text,
  },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.textSecondary,
  },
  emptyBtn: {
    marginTop: spacing.md,
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: spacing.xl,
    paddingVertical: 12,
  },
  emptyBtnText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  // Summary strip
  summary: {
    backgroundColor: colors.bgCard,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.borderSolid,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    gap: spacing.sm,
  },
  summaryText: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: spacing.sm,
  },
  summaryTitle: {
    flexShrink: 1,
    fontFamily: fonts.heading,
    fontSize: 15,
    color: colors.text,
  },
  summaryDemo: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    color: colors.textMuted,
  },
  flagWrap: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  flagPill: {
    backgroundColor: 'rgba(220,38,38,0.08)',
    borderWidth: 1,
    borderColor: 'rgba(220,38,38,0.25)',
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 3,
  },
  flagText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 11.5,
    color: colors.red,
  },
  // Segmented control (mirrors web .reasoning-tabs)
  segmentBar: {
    flexDirection: 'row',
    backgroundColor: colors.bgWarm,
    borderRadius: radius.sm + 2,
    margin: spacing.md,
    padding: 3,
    gap: 3,
  },
  segmentBtn: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 8,
    borderRadius: radius.sm,
  },
  segmentBtnActive: {
    backgroundColor: colors.bgCard,
    ...shadow.sm,
  },
  segmentLabel: {
    fontFamily: fonts.bodyMedium,
    fontSize: 13.5,
    color: colors.textSecondary,
  },
  segmentLabelActive: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
  },
  // Content
  listContent: {
    paddingHorizontal: spacing.md,
    paddingBottom: spacing.xl,
  },
  noData: {
    fontFamily: fonts.body,
    fontSize: 13.5,
    color: colors.textMuted,
    textAlign: 'center',
    marginTop: spacing.xl,
  },
  graphWrap: {
    flex: 1,
    marginHorizontal: spacing.md,
    marginBottom: spacing.md,
    borderRadius: radius.card,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderSolid,
    backgroundColor: colors.bg,
  },
});
