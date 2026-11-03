import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { useInterview } from '../state/InterviewContext';
import { colors, spacing } from '../theme';
import { SectionCard } from '../components/SectionCard';
import { MetricPill } from '../components/MetricPill';
import { PrimaryButton } from '../components/PrimaryButton';

export function FeedbackScreen({ navigation }: { navigation: any }) {
  const { sessionRecords, profile } = useInterview();

  const summary = useMemo(() => {
    if (sessionRecords.length === 0) {
      return {
        average: 'N/A',
        bestPractice: 'Answer one question in full before moving to the next.',
        focusArea: 'Upload a resume and complete at least one answer to unlock feedback.',
      };
    }
    const averageScore = Math.round(
      sessionRecords.reduce((sum, record) => sum + record.review.score, 0) / sessionRecords.length
    );
    const strongest = sessionRecords[0]?.review.strengths[0] ?? 'You provided a clear answer.';
    const improvement = sessionRecords[0]?.review.improvements[0] ?? 'Add more measurable impact.';
    return {
      average: `${averageScore}/10`,
      bestPractice: strongest,
      focusArea: improvement,
    };
  }, [sessionRecords]);

  return (
    <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
      <SectionCard>
        <Text style={styles.kicker}>Feedback Report</Text>
        <Text style={styles.title}>AI Interview Coach Review</Text>
        <Text style={styles.subtitle}>{profile?.summary ?? 'Complete an interview to generate personalized feedback.'}</Text>
        <View style={styles.metricsRow}>
          <MetricPill label="Average" value={summary.average} />
          <MetricPill label="Answers" value={String(sessionRecords.length)} />
        </View>
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Strengths</Text>
        <Text style={styles.sectionText}>{summary.bestPractice}</Text>
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Improvement Focus</Text>
        <Text style={styles.sectionText}>{summary.focusArea}</Text>
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Answer Upgrade Formula</Text>
        <Text style={styles.sectionText}>1. Start with the situation</Text>
        <Text style={styles.sectionText}>2. Explain the action you took</Text>
        <Text style={styles.sectionText}>3. End with a measurable result</Text>
        <Text style={styles.sectionText}>4. Add keywords from your resume</Text>
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Suggested Answer Style</Text>
        <Text style={styles.sectionText}>
          Use a short STAR structure, mention one or two keywords from your resume, and finish with a result or outcome.
        </Text>
        <PrimaryButton label="Back to Interview" onPress={() => navigation.navigate('Interview')} />
      </SectionCard>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
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
    fontSize: 30,
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
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 10,
  },
  sectionText: {
    color: colors.muted,
    lineHeight: 22,
    marginBottom: 10,
  },
});
