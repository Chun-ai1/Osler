/**
 * Collapsible agent-reasoning card — mirrors the web demo's navy `.trace`
 * block: uppercase Outfit header with chevron, rows of icon + title + detail.
 */
import Ionicons from '@expo/vector-icons/Ionicons';
import React, { useState } from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type { TraceStep } from '@/api/osler';
import { colors, fonts, radius } from '@/theme/tokens';

interface Props {
  steps: TraceStep[];
}

export default function TraceCard({ steps }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!steps || steps.length === 0) return null;

  return (
    <View style={styles.card}>
      <Pressable
        accessibilityRole="button"
        accessibilityState={{ expanded }}
        onPress={() => setExpanded((e) => !e)}
        style={({ pressed }) => [styles.header, pressed && { opacity: 0.85 }]}>
        <Text style={styles.headerText}>
          AGENT REASONING · {steps.length} {steps.length === 1 ? 'STEP' : 'STEPS'}
        </Text>
        <Ionicons
          name={expanded ? 'chevron-down' : 'chevron-forward'}
          size={15}
          color={colors.traceMuted}
        />
      </Pressable>
      {expanded && (
        <View style={styles.body}>
          {steps.map((s, i) => (
            <View key={`${s.title}-${i}`} style={styles.step}>
              <Text style={styles.icon}>{s.icon}</Text>
              <View style={styles.stepText}>
                <Text style={styles.title}>{s.title}</Text>
                {!!s.detail && <Text style={styles.detail}>{s.detail}</Text>}
              </View>
            </View>
          ))}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.traceBg,
    borderRadius: radius.card,
    paddingHorizontal: 16,
    paddingVertical: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    minHeight: 44,
  },
  headerText: {
    flex: 1,
    fontFamily: fonts.heading,
    fontSize: 11,
    letterSpacing: 1.6,
    color: colors.traceMuted,
  },
  body: {
    paddingBottom: 12,
    paddingTop: 2,
  },
  step: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 9,
    paddingVertical: 5,
  },
  icon: {
    fontSize: 14,
    lineHeight: 19,
  },
  stepText: {
    flex: 1,
  },
  title: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 13,
    lineHeight: 18,
    color: colors.traceText,
  },
  detail: {
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 17,
    color: colors.traceMuted,
    marginTop: 1,
  },
});
