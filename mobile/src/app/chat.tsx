/**
 * Chat screen — ask the agent about the CURRENT patient's result.
 * Mirrors the web demo's center column: trace card up top, user/AI bubbles,
 * suggested-question chips, pinned composer.
 */
import Ionicons from '@expo/vector-icons/Ionicons';
import { Link, useRouter } from 'expo-router';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Animated,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { chat, type ChatMessage } from '@/api/osler';
import MessageBubble from '@/components/MessageBubble';
import TraceCard from '@/components/TraceCard';
import { useApp } from '@/state/AppContext';
import { colors, fonts, radius, shadow, spacing } from '@/theme/tokens';

function TypingBubble() {
  const pulse = useRef(new Animated.Value(0.35)).current;
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 500, useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0.35, duration: 500, useNativeDriver: true }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [pulse]);
  return (
    <View style={styles.typingBubble}>
      <Animated.Text style={[styles.typingDots, { opacity: pulse }]}>…</Animated.Text>
    </View>
  );
}

export default function ChatScreen() {
  const router = useRouter();
  const { current, appendChat, llm } = useApp();
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [showSettingsHint, setShowSettingsHint] = useState(false);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const listRef = useRef<FlatList<ChatMessage>>(null);

  const chatLen = current?.chat.length ?? 0;
  useEffect(() => {
    if (chatLen > 0) {
      // small delay so the new row has rendered before we scroll
      const t = setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 80);
      return () => clearTimeout(t);
    }
    return undefined;
  }, [chatLen, sending]);

  const suggestions = useMemo(() => {
    if (!current) return [];
    const cands = current.bundle.result?.candidates ?? [];
    const out: string[] = [];
    const first = cands[0];
    if (first) out.push(`Why is ${first.drug} first?`);
    const avoided = cands.find((c) => {
      const d = (c.safety?.decision ?? '').toLowerCase();
      return d === 'avoid' || d === 'block';
    });
    if (avoided) out.push(`Why is ${avoided.drug} avoided?`);
    out.push('What is the disease doing to the body?');
    return out;
  }, [current]);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || !current || sending) return;
      setInput('');
      setSending(true);
      const userMsg: ChatMessage = { role: 'user', content };
      appendChat(current.id, userMsg);
      try {
        const res = await chat(
          current.id,
          [...current.chat, userMsg],
          current.bundle,
          llm,
        );
        appendChat(current.id, { role: 'assistant', content: res.reply });
        if (res.ok === false) setShowSettingsHint(true);
      } catch (e) {
        appendChat(current.id, {
          role: 'assistant',
          content: `Could not reach the backend (${e instanceof Error ? e.message : 'network error'}). Is \`python demo/demo_app.py\` running?`,
        });
      } finally {
        setSending(false);
      }
    },
    [current, sending, appendChat, llm],
  );

  if (!current) {
    return (
      <View style={styles.empty}>
        <Ionicons name="chatbubble-ellipses-outline" size={40} color={colors.silverLight} />
        <Text style={styles.emptyText}>
          Analyze a case first so I can explain its result.
        </Text>
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push('/analyze')}
          style={({ pressed }) => [styles.emptyButton, pressed && { opacity: 0.85 }]}>
          <Text style={styles.emptyButtonText}>Go to Analyze</Text>
        </Pressable>
      </View>
    );
  }

  const showChips = current.chat.length === 0 && !sending;

  return (
    <KeyboardAvoidingView
      style={styles.flex}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}>
      {!llm.apiKey && !bannerDismissed && (
        <View style={styles.banner}>
          <Text style={styles.bannerText}>
            No LLM key set — chat will return a fallback notice. Add one in{' '}
            <Link href="/settings" style={styles.bannerLink}>
              Settings
            </Link>
            .
          </Text>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Dismiss"
            onPress={() => setBannerDismissed(true)}
            hitSlop={12}
            style={styles.bannerClose}>
            <Ionicons name="close" size={16} color={colors.textSecondary} />
          </Pressable>
        </View>
      )}

      <FlatList
        ref={listRef}
        data={current.chat}
        keyExtractor={(_, i) => String(i)}
        renderItem={({ item }) => <MessageBubble message={item} />}
        contentContainerStyle={styles.listContent}
        keyboardShouldPersistTaps="handled"
        ListHeaderComponent={
          <View style={styles.listHeader}>
            <TraceCard steps={current.bundle.trace} />
            {showChips && (
              <View style={styles.chips}>
                {suggestions.map((q) => (
                  <Pressable
                    key={q}
                    accessibilityRole="button"
                    onPress={() => send(q)}
                    style={({ pressed }) => [styles.chip, pressed && { opacity: 0.8 }]}>
                    <Text style={styles.chipText}>{q}</Text>
                  </Pressable>
                ))}
              </View>
            )}
          </View>
        }
        ListFooterComponent={
          <View>
            {sending && <TypingBubble />}
            {showSettingsHint && (
              <Link href="/settings" style={styles.hint}>
                Add an API key in Settings to enable full chat →
              </Link>
            )}
          </View>
        }
      />

      <View style={styles.composer}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder={`Ask about ${current.title}…`}
          placeholderTextColor={colors.textMuted}
          returnKeyType="send"
          onSubmitEditing={() => send(input)}
          editable={!sending}
          multiline={false}
        />
        <Pressable
          accessibilityRole="button"
          accessibilityLabel="Send message"
          onPress={() => send(input)}
          disabled={sending || !input.trim()}
          style={({ pressed }) => [
            styles.sendButton,
            (sending || !input.trim()) && { opacity: 0.5 },
            pressed && { opacity: 0.8 },
          ]}>
          <Ionicons name="send" size={18} color="#FFFFFF" />
        </Pressable>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1, backgroundColor: colors.bg },
  empty: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.lg,
    padding: spacing.xl,
    backgroundColor: colors.bg,
  },
  emptyText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 21,
    color: colors.textSecondary,
    textAlign: 'center',
    maxWidth: 280,
  },
  emptyButton: {
    backgroundColor: colors.accent,
    borderRadius: radius.pill,
    paddingHorizontal: 24,
    minHeight: 44,
    justifyContent: 'center',
  },
  emptyButtonText: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 14,
    color: '#FFFFFF',
  },
  banner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.accentLight,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  bannerText: {
    flex: 1,
    fontFamily: fonts.body,
    fontSize: 12,
    lineHeight: 17,
    color: colors.textSecondary,
  },
  bannerLink: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
    textDecorationLine: 'underline',
  },
  bannerClose: {
    minWidth: 28,
    minHeight: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  listContent: {
    padding: spacing.lg,
    paddingBottom: spacing.xl,
  },
  listHeader: {
    marginBottom: spacing.sm,
    gap: spacing.md,
  },
  chips: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 7,
  },
  chip: {
    backgroundColor: colors.accentLight,
    borderRadius: radius.pill,
    paddingHorizontal: 14,
    paddingVertical: 12,
    minHeight: 44,
    justifyContent: 'center',
  },
  chipText: {
    fontFamily: fonts.bodyMedium,
    fontSize: 12.5,
    color: colors.accent,
  },
  typingBubble: {
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    borderBottomRightRadius: 12,
    borderBottomLeftRadius: 0,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginVertical: 6,
    alignSelf: 'flex-start',
    ...shadow.sm,
  },
  typingDots: {
    fontFamily: fonts.bodySemiBold,
    fontSize: 18,
    color: colors.textMuted,
    letterSpacing: 2,
  },
  hint: {
    fontFamily: fonts.body,
    fontSize: 12,
    color: colors.accent,
    textDecorationLine: 'underline',
    marginTop: spacing.xs,
    marginLeft: 2,
  },
  composer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 9,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.borderSolid,
    backgroundColor: colors.bgCard,
  },
  input: {
    flex: 1,
    fontFamily: fonts.body,
    fontSize: 14,
    color: colors.text,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.borderSolid,
    borderRadius: radius.pill,
    paddingHorizontal: 16,
    paddingVertical: 0,
    minHeight: 44,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
