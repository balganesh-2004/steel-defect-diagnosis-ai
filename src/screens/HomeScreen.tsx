import React, { useMemo, useState } from 'react';
import { FlatList, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { useInterview } from '../state/InterviewContext';
import { colors, spacing } from '../theme';
import { SectionCard } from '../components/SectionCard';
import { PrimaryButton } from '../components/PrimaryButton';
import { MetricPill } from '../components/MetricPill';

export function HomeScreen({ navigation }: { navigation: any }) {
  const { profile, questions, uploadResume, loadResumeFromText, isUploading, latestReviewText, resetSession } = useInterview();
  const [resumeText, setResumeText] = useState('');

  const derivedKeywords = useMemo(() => profile?.keywords ?? [], [profile]);

  const applyTypedResume = () => {
    if (!resumeText.trim()) {
      return;
    }
    loadResumeFromText(resumeText, 'Pasted Resume');
    navigation.navigate('Interview');
  };

  return (
    <ScrollView contentContainerStyle={styles.container} showsVerticalScrollIndicator={false}>
      <View style={styles.hero}>
        <Text style={styles.kicker}>AI Interview Coach</Text>
        <Text style={styles.title}>Practice interviews from your resume with chat and voice coaching.</Text>
        <Text style={styles.subtitle}>
          Upload a resume, generate targeted questions, answer like a real candidate, and get instant feedback.
        </Text>
      </View>

      <SectionCard>
        <Text style={styles.sectionTitle}>Resume Intake</Text>
        <Text style={styles.sectionText}>
          The app works best with a pasted resume or a text-based document. PDF support is demo-friendly and can be extended later.
        </Text>
        <PrimaryButton label={isUploading ? 'Uploading...' : 'Upload Resume'} onPress={uploadResume} loading={isUploading} />
        <View style={styles.spacer} />
        <TextInput
          style={styles.input}
          multiline
          placeholder="Paste resume text here for a quicker demo..."
          placeholderTextColor={colors.muted}
          value={resumeText}
          onChangeText={setResumeText}
        />
        <View style={styles.spacer} />
        <PrimaryButton label="Use Pasted Resume" onPress={applyTypedResume} />
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Live Profile</Text>
        <View style={styles.metricsRow}>
          <MetricPill label="Questions" value={String(questions.length || 0)} />
          <MetricPill label="Keywords" value={String(derivedKeywords.length || 0)} />
        </View>
        <View style={styles.spacer} />
        <Text style={styles.sectionText}>{profile?.summary ?? 'No resume processed yet.'}</Text>
        {derivedKeywords.length > 0 ? (
          <FlatList
            data={derivedKeywords}
            keyExtractor={(item) => item}
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.keywordList}
            renderItem={({ item }) => <View style={styles.keywordChip}><Text style={styles.keywordText}>{item}</Text></View>}
          />
        ) : null}
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>Start Interview</Text>
        <Text style={styles.sectionText}>{latestReviewText}</Text>
        <View style={styles.buttonRow}>
          <PrimaryButton label="Go to Interview" onPress={() => navigation.navigate('Interview')} style={styles.flexButton} />
          <PrimaryButton label="Reset" onPress={resetSession} style={styles.flexButton} />
        </View>
      </SectionCard>

      <SectionCard>
        <Text style={styles.sectionTitle}>How it works</Text>
        <Text style={styles.sectionText}>
          The coach uses local resume parsing and answer scoring to generate questions, review responses, and suggest stronger wording.
        </Text>
        <Text style={styles.sectionText}>1. Upload resume or paste text</Text>
        <Text style={styles.sectionText}>2. Answer dynamic interview questions</Text>
        <Text style={styles.sectionText}>3. Read feedback and improve</Text>
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
  hero: {
    marginBottom: spacing.lg,
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
    fontSize: 34,
    lineHeight: 40,
    fontWeight: '900',
    marginBottom: spacing.md,
  },
  subtitle: {
    color: colors.muted,
    fontSize: 16,
    lineHeight: 24,
  },
  sectionTitle: {
    color: colors.text,
    fontSize: 20,
    fontWeight: '800',
    marginBottom: 10,
  },
  sectionText: {
    color: colors.muted,
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 12,
  },
  spacer: {
    height: spacing.md,
  },
  input: {
    minHeight: 140,
    borderRadius: 18,
    backgroundColor: colors.panelStrong,
    color: colors.text,
    padding: 14,
    textAlignVertical: 'top',
    borderWidth: 1,
    borderColor: colors.panelBorder,
  },
  metricsRow: {
    flexDirection: 'row',
    gap: 12,
  },
  keywordList: {
    gap: 10,
    paddingTop: 12,
  },
  keywordChip: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 999,
    paddingHorizontal: 14,
    paddingVertical: 8,
    marginRight: 10,
  },
  keywordText: {
    color: colors.text,
    fontWeight: '700',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  flexButton: {
    flex: 1,
  },
});
