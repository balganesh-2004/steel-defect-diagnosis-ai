import React from 'react';
import { ActivityIndicator, Pressable, StyleSheet, Text, ViewStyle } from 'react-native';
import { colors, spacing } from '../theme';

export function PrimaryButton({
  label,
  onPress,
  loading,
  style,
}: {
  label: string;
  onPress: () => void;
  loading?: boolean;
  style?: ViewStyle;
}) {
  return (
    <Pressable onPress={onPress} style={({ pressed }) => [styles.button, pressed && styles.pressed, style]}>
      {loading ? <ActivityIndicator color={colors.backgroundTop} /> : <Text style={styles.text}>{label}</Text>}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  button: {
    minHeight: 52,
    borderRadius: 16,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.lg,
  },
  pressed: {
    transform: [{ scale: 0.98 }],
    opacity: 0.95,
  },
  text: {
    color: colors.backgroundTop,
    fontWeight: '800',
    fontSize: 16,
    letterSpacing: 0.3,
  },
});
