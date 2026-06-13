/**
 * Preset chips row — mirrors the web demo's `.preset` pills:
 * accent-light background, navy text, 50px-radius pills, horizontally
 * scrollable. Active chip flips to solid navy.
 */
import React from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
} from 'react-native';

import type { SampleCase } from '@/api/osler';
import { colors, fonts, radius } from '@/theme/tokens';

interface Props {
  cases: SampleCase[];
  selectedId: string | null;
  onSelect: (c: SampleCase) => void;
}

export default function CasePresetChips({ cases, selectedId, onSelect }: Props) {
  if (cases.length === 0) return null;
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}>
      {cases.map((c) => {
        const active = c.id === selectedId;
        return (
          <Pressable
            key={c.id}
            accessibilityRole="button"
            accessibilityState={{ selected: active }}
            onPress={() => onSelect(c)}
            style={({ pressed }) => [
              styles.chip,
              active && styles.chipActive,
              pressed && { opacity: 0.8 },
            ]}>
            <Text style={[styles.chipText, active && styles.chipTextActive]}>
              {c.title}
            </Text>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  row: {
    gap: 7,
    paddingVertical: 2,
  },
  chip: {
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 12, // ≥44px touch target with text line height
    justifyContent: 'center',
    minHeight: 44,
  },
  chipActive: {
    backgroundColor: colors.accent,
  },
  chipText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.accent,
  },
  chipTextActive: {
    color: '#FFFFFF',
  },
});
