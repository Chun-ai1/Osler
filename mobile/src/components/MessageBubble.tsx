/**
 * Chat message bubble — mirrors the web demo's `.msg-user` / `.msg-ai`:
 * user → navy bubble, right-aligned, radius 12/12/0/12;
 * assistant → white card with border + sm shadow, radius 12/12/12/0,
 * markdown-rendered content.
 */
import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import Markdown from 'react-native-markdown-display';

import type { ChatMessage } from '@/api/osler';
import { colors, fonts, shadow } from '@/theme/tokens';

interface Props {
  message: ChatMessage;
}

const markdownStyles = {
  body: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 22,
    color: colors.text,
  },
  strong: {
    fontFamily: fonts.bodySemiBold,
    color: colors.accent,
  },
  bullet_list: { marginVertical: 4 },
  ordered_list: { marginVertical: 4 },
  list_item: { marginVertical: 2 },
  paragraph: { marginTop: 4, marginBottom: 4 },
  code_inline: {
    backgroundColor: colors.accentLight,
    color: colors.text,
    borderRadius: 4,
    paddingHorizontal: 5,
    fontSize: 13,
  },
  code_block: {
    backgroundColor: colors.accentLight,
    color: colors.text,
    borderRadius: 6,
    padding: 10,
    fontSize: 13,
    borderWidth: 0,
  },
  fence: {
    backgroundColor: colors.accentLight,
    color: colors.text,
    borderRadius: 6,
    padding: 10,
    fontSize: 13,
    borderWidth: 0,
  },
} as const;

export default function MessageBubble({ message }: Props) {
  if (message.role === 'user') {
    return (
      <View style={styles.userBubble}>
        <Text style={styles.userText}>{message.content}</Text>
      </View>
    );
  }
  return (
    <View style={styles.aiBubble}>
      <Markdown style={markdownStyles}>{message.content}</Markdown>
    </View>
  );
}

const styles = StyleSheet.create({
  userBubble: {
    backgroundColor: colors.accent,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    borderBottomRightRadius: 0,
    borderBottomLeftRadius: 12,
    marginVertical: 6,
    maxWidth: '75%',
    alignSelf: 'flex-end',
  },
  userText: {
    fontFamily: fonts.body,
    fontSize: 14,
    lineHeight: 21,
    color: '#FFFFFF',
  },
  aiBubble: {
    backgroundColor: colors.bgCard,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopLeftRadius: 12,
    borderTopRightRadius: 12,
    borderBottomRightRadius: 12,
    borderBottomLeftRadius: 0,
    marginVertical: 6,
    alignSelf: 'flex-start',
    maxWidth: '92%',
    borderWidth: 1,
    borderColor: colors.borderSolid,
    ...shadow.sm,
  },
});
