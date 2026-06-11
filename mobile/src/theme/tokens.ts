/**
 * Design tokens ported 1:1 from the web demo (demo/case_demo.html :root).
 * Keep in sync with the web demo so both clients read as one product.
 */
export const colors = {
  bg: '#F8F9FC',
  bgWarm: '#EEF1F7',
  bgCard: '#FFFFFF',
  accent: '#0F2B5B', // navy — primary text, buttons, user bubbles
  accentLight: '#E8EDF5',
  accentBright: '#1A3F7A',
  silver: '#8C9BB5',
  silverLight: '#C5CEDB',
  text: '#0F2B5B',
  textSecondary: '#4A5A7A',
  textMuted: '#8C9BB5',
  border: 'rgba(15,43,91,0.08)',
  borderSolid: '#E2E7F0',
  orange: '#E8724A', // logo "ian", disease node, alerts
  green: '#059669',
  amber: '#D97706',
  red: '#DC2626',
  gray: '#6B7280',
  teal: '#0D9488',
  traceBg: '#0F2B5B', // agent-trace card background
  traceText: '#E8EDF5',
  traceMuted: '#9DB2D6',
} as const;

/** safety.decision → border/badge color (mirrors web drug node coloring) */
export const safetyColor: Record<string, string> = {
  ok: colors.green,
  show: colors.green,
  caution: colors.amber,
  warn: colors.amber,
  adjust: colors.teal,
  avoid: colors.red,
  block: colors.red,
};

export function colorForSafety(decision?: string): string {
  if (!decision) return colors.gray;
  return safetyColor[decision.toLowerCase()] ?? colors.gray;
}

export const radius = {
  card: 14,
  sm: 9,
  pill: 50,
} as const;

export const fonts = {
  /** headings / logo / section labels */
  heading: 'Outfit_600SemiBold',
  headingBold: 'Outfit_700Bold',
  /** body text */
  body: 'Sora_400Regular',
  bodyMedium: 'Sora_500Medium',
  bodySemiBold: 'Sora_600SemiBold',
} as const;

export const shadow = {
  sm: {
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowRadius: 3,
    shadowOffset: { width: 0, height: 1 },
    elevation: 1,
  },
  md: {
    shadowColor: '#000',
    shadowOpacity: 0.06,
    shadowRadius: 30,
    shadowOffset: { width: 0, height: 8 },
    elevation: 4,
  },
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 18,
  xl: 24,
} as const;
