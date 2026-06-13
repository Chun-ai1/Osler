/**
 * DrugSources — per-drug "sources · evidence" block, ported from the web demo's
 * citationsHtml() in demo/case_demo.html. Tells the clinician where each claim
 * comes from and links out to the authoritative database to verify:
 *   • Mechanism   → DrugBank (pharmacology authority)
 *   • Disease     → clinical reference for the modeled disease
 *   • FDA label   → DailyMed SPL (official FDA source; direct link when set_id known)
 *   • Adverse ev. → FAERS signals + a side-effect reference
 * Links open in the system browser on tap (user-initiated).
 */
import React from 'react';
import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';

import type { DiseaseModel, DrugCandidate } from '@/api/osler';
import { colors, fonts, radius, spacing } from '@/theme/tokens';

interface Cite {
  label: string;
  text: string;
  ext: string;
  url: string;
}

function buildCitations(c: DrugCandidate, dm?: DiseaseModel | null): Cite[] {
  const q = encodeURIComponent(c.drug);
  const ev = c.evidence?.[0] ?? null;
  const setid = ev?.set_id;
  const realLabel = !!ev && /openfda|dailymed|spl/i.test(ev.source ?? '');
  const dmUrl =
    realLabel && setid
      ? `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=${setid}`
      : `https://dailymed.nlm.nih.gov/dailymed/search.cfm?query=${q}`;

  const cites: Cite[] = [];

  // Mechanism — readable rationale, verify on DrugBank (distinct from the FDA label)
  cites.push({
    label: 'Mechanism',
    text: c.rationale || `modeled to ${c.mechanism_chain || 'act on the target'}`,
    ext: 'Modeled effect (drugs_pkpd.json, unreviewed); verify on DrugBank ↗',
    url: `https://go.drugbank.com/unearth/q?searcher=drugs&query=${q}`,
  });

  // Disease model — its own description + a clinical reference to verify
  if (dm) {
    const desc =
      dm.description?.slice(0, 110) ||
      'pushes ' +
        (dm.perturbations ?? [])
          .slice(0, 3)
          .map((p) => p.variable.replace(/_/g, ' ') + (p.direction === 'high' ? '↑' : '↓'))
          .join(', ');
    cites.push({
      label: 'Disease model',
      text: desc,
      ext: `From ${dm.source_file ? 'data/' + dm.source_file : 'inferred'} (unreviewed); reference ↗`,
      url: `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(
        (dm.disease || '').replace(/\//g, ' '),
      )}`,
    });
  }

  // FDA label — DailyMed SPL (official FDA source)
  cites.push({
    label: 'FDA label',
    text: realLabel
      ? `DailyMed SPL${setid ? ' ' + String(setid).slice(0, 8) : ''}${
          ev?.retrieved_at ? ' · ' + ev.retrieved_at : ''
        }`
      : `look up “${c.drug}” on DailyMed (official FDA source)`,
    ext: '↗',
    url: dmUrl,
  });

  // Adverse events — FAERS-backed side-effect reference
  if (c.faers_signals?.length) {
    cites.push({
      label: 'Adverse events',
      text: 'side-effect reference (Drugs.com)',
      ext: '↗',
      url: `https://www.drugs.com/sfx/${encodeURIComponent(
        c.drug.toLowerCase().replace(/\s+/g, '-'),
      )}-side-effects.html`,
    });
  }

  return cites;
}

export default function DrugSources({
  candidate,
  disease,
}: {
  candidate: DrugCandidate;
  disease?: DiseaseModel | null;
}) {
  const cites = buildCitations(candidate, disease);
  const faers = candidate.faers_signals?.slice(0, 3) ?? [];

  const open = (url: string) => {
    Linking.openURL(url).catch(() => {});
  };

  return (
    <View style={styles.wrap}>
      {faers.length > 0 && (
        <View style={styles.faers}>
          <Text style={styles.faersText}>
            ⚠ FAERS:{' '}
            {faers
              .map((s) => `${s.event} (${s.report_count.toLocaleString()})`)
              .join(', ')}
          </Text>
        </View>
      )}

      <Text style={styles.sectionLabel}>SOURCES · EVIDENCE</Text>
      {cites.map((cite, i) => (
        <Pressable
          key={i}
          onPress={() => open(cite.url)}
          style={({ pressed }) => [styles.cite, pressed && styles.citePressed]}
          accessibilityRole="link"
          accessibilityLabel={`${cite.label}: open source`}>
          <Text style={styles.citeText}>
            <Text style={styles.citeLabel}>{cite.label}</Text>
            <Text> — {cite.text}. </Text>
            <Text style={styles.citeExt}>{cite.ext}</Text>
          </Text>
        </Pressable>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginTop: spacing.md,
    paddingTop: spacing.sm,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: colors.borderSolid,
    gap: spacing.xs,
  },
  faers: {
    backgroundColor: '#FFFBEB',
    borderWidth: 1,
    borderColor: '#FDE68A',
    borderRadius: radius.sm,
    padding: spacing.sm,
    marginBottom: spacing.xs,
  },
  faersText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12,
    lineHeight: 18,
    color: colors.amber,
  },
  sectionLabel: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 10,
    letterSpacing: 0.8,
    color: colors.textMuted,
    marginBottom: 2,
  },
  cite: {
    backgroundColor: colors.bgWarm,
    borderRadius: radius.sm,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  citePressed: {
    backgroundColor: colors.accentLight,
  },
  citeText: {
    fontFamily: fonts.body,
    fontSize: 12.5,
    lineHeight: 18,
    color: colors.textSecondary,
  },
  citeLabel: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
  },
  citeExt: {
    fontFamily: fonts.bodyMedium,
    color: colors.teal,
  },
});
