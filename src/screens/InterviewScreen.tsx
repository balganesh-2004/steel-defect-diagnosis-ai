import React, { useMemo, useState } from 'react';
import { FlatList, KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useInterview } from '../state/InterviewContext';
import { colors, spacing } from '../theme';
import { SectionCard } from '../components/SectionCard';
import { PrimaryButton } from '../components/PrimaryButton';
import { MetricPill } from '../components/MetricPill';

export function InterviewScreen({ navigation }: { navigation: any }) {
  const {
    profile,
    questions,
    currentQuestionIndex,
    messages,
    sessionRecords,
    isVoiceMode,
    latestReviewText,
    sendAnswer,
    speakQuestion,
    toggleVoiceMode,
    startInterview,
  } = useInterview();
  const [answer, setAnswer] = useState('');

  const currentQuestion = questions[currentQuestionIndex] ?? null;
  const scoreAverage = useMemo(() => {
    if (sessionRecords.length === 0) {
      return 'N/A';
    }
    const total = sessionRecords.reduce((sum, record) => sum + record.review.score, 0);
    return `${Math.round(total / sessionRecords.length)}/10`;
  }, [sessionRecords]);

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : undefined} style={styles.flex}>
      <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
        <SectionCard>
          <Text style={styles.kicker}>Live Interview</Text>
          <Text style={styles.title}>{currentQuestion?.question ?? 'Upload a resume to generate interview questions.'}</Text>
          <Text style={styles.subtitle}>{profile?.summary ?? 'The interview coach will tailor the questions to your resume keywords.'}</Text>
          <View style={styles.metricsRow}>
            <MetricPill label="Mode" value={isVoiceMode ? 'Voice' : 'Chat'} />
            <MetricPill label="Score" value={scoreAverage} />
          </View>
          <View style={styles.buttonRow}>
            <PrimaryButton label="Start Interview" onPress={startInterview} style={styles.flexButton} />
            <PrimaryButton label="Speak Question" onPress={speakQuestion} style={styles.flexButton} />
          </View>
          <View style={styles.spacer} />
          <PrimaryButton label={isVoiceMode ? 'Switch to Chat Mode' : 'Switch to Voice Mode'} onPress={toggleVoiceMode} />
        </SectionCard>

        <SectionCard>
          <Text style={styles.sectionTitle}>Your Response</Text>
          <TextInput
            style={styles.input}
            multiline
            placeholder="Type your answer here or use voice mode as your practice guide..."
            placeholderTextColor={colors.muted}
            value={answer}
            onChangeText={setAnswer}
          />
          <View style={styles.spacer} />
          <PrimaryButton
            label="Submit Answer"
            onPress={() => {
              sendAnswer(answer);
              setAnswer('');
            }}
          />
          <Text style={styles.note}>{latestReviewText}</Text>
        </SectionCard>

        <SectionCard>
          <Text style={styles.sectionTitle}>Chat Coach</Text>
          <FlatList
            data={messages}
            scrollEnabled={false}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <View style={[styles.chatBubble, item.role === 'user' ? styles.userBubble : styles.coachBubble]}>
                <Text style={styles.chatRole}>{item.role === 'user' ? 'You' : 'Coach'}</Text>
                <Text style={styles.chatText}>{item.text}</Text>
              </View>
            )}
          />
        </SectionCard>

        <SectionCard>
          <Text style={styles.sectionTitle}>Voice Meeting Notes</Text>
          <Text style={styles.sectionText}>
            The app uses speech playback for the interviewer and can be extended with speech-to-text later.
          </Text>
          <Text style={styles.sectionText}>
            For the college demo, this gives you a strong voice-coach experience without needing an external API key.
          </Text>
          <PrimaryButton label="See Feedback" onPress={() => navigation.navigate('Feedback')} />
        </SectionCard>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  flex: {
    flex: 1,
  },
  container: {
    padding: spacing.lg,
    paddingTop: 64,
    paddingBottom: 56,
  },
  kicker: {
    color: colors.accent,
    textTransform: 'uppercase',
    letterSpacing: 2,
    fontWeight: '800',
    marginBottom: 10,
  },
  title: {
    color: colors.text,
    fontSize: 28,
    lineHeight: 36,
    fontWeight: '900',
    marginBottom: spacing.md,
  },
  subtitle: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    marginBottom: spacing.md,
  },
  metricsRow: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: spacing.md,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  flexButton: {
    flex: 1,
  },
  spacer: {
    height: spacing.md,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 10,
  },
  input: {
    minHeight: 132,
    borderRadius: 18,
    backgroundColor: colors.panelStrong,
    color: colors.text,
    padding: 14,
    textAlignVertical: 'top',
    borderWidth: 1,
    borderColor: colors.panelBorder,
  },
  note: {
    color: colors.muted,
    marginTop: 12,
    lineHeight: 20,
  },
  chatBubble: {
    borderRadius: 18,
    padding: 14,
    marginBottom: 12,
    borderWidth: 1,
  },
  userBubble: {
    backgroundColor: 'rgba(120, 255, 214, 0.12)',
    borderColor: 'rgba(120, 255, 214, 0.2)',
    alignSelf: 'flex-end',
  },
  coachBubble: {
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderColor: colors.panelBorder,
    alignSelf: 'flex-start',
  },
  chatRole: {
    color: colors.accent,
    fontSize: 12,
    fontWeight: '800',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  chatText: {
    color: colors.text,
    lineHeight: 21,
  },
  sectionText: {
    color: colors.muted,
    lineHeight: 22,
    marginBottom: 12,
  },
});
