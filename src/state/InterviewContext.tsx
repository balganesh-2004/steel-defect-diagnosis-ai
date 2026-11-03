import React, { createContext, useContext, useMemo, useState } from 'react';
import { Alert } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import * as Speech from 'expo-speech';
import { buildQuestions, evaluateAnswer, parseResumeText } from '../services/interviewEngine';
import { ChatMessage, InterviewQuestion, ResumeProfile, SessionRecord } from '../types';

type InterviewContextValue = {
  profile: ResumeProfile | null;
  questions: InterviewQuestion[];
  currentQuestionIndex: number;
  messages: ChatMessage[];
  sessionRecords: SessionRecord[];
  isUploading: boolean;
  isVoiceMode: boolean;
  latestReviewText: string;
  uploadResume: () => Promise<void>;
  loadResumeFromText: (rawText: string, fileName?: string) => void;
  startInterview: () => void;
  speakQuestion: () => void;
  sendAnswer: (answer: string) => void;
  toggleVoiceMode: () => void;
  resetSession: () => void;
};

const InterviewContext = createContext<InterviewContextValue | undefined>(undefined);

function createCoachMessage(text: string): ChatMessage {
  return {
    id: `coach-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role: 'coach',
    text,
    timestamp: Date.now(),
  };
}

export function InterviewProvider({ children }: { children: React.ReactNode }) {
  const [profile, setProfile] = useState<ResumeProfile | null>(null);
  const [questions, setQuestions] = useState<InterviewQuestion[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [messages, setMessages] = useState<ChatMessage[]>([
    createCoachMessage('Upload a resume or paste its text to begin your interview practice.'),
  ]);
  const [sessionRecords, setSessionRecords] = useState<SessionRecord[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isVoiceMode, setIsVoiceMode] = useState(true);
  const [latestReviewText, setLatestReviewText] = useState('Your feedback will appear after the first answer.');

  const currentQuestion = questions[currentQuestionIndex] ?? null;

  const setResumeProfile = (rawText: string, fileName?: string) => {
    const parsed = parseResumeText(rawText, fileName);
    const generatedQuestions = buildQuestions(parsed);
    setProfile(parsed);
    setQuestions(generatedQuestions);
    setCurrentQuestionIndex(0);
    setSessionRecords([]);
    setLatestReviewText('Resume processed. The first interview question is ready.');
    setMessages([
      createCoachMessage(`Resume loaded: ${parsed.title}. I found ${parsed.keywords.length} useful keywords.`),
      createCoachMessage(`Ready to start with: ${generatedQuestions[0]?.question ?? 'No question available yet.'}`),
    ]);
  };

  const uploadResume = async () => {
    try {
      setIsUploading(true);
      const result = await DocumentPicker.getDocumentAsync({
        type: ['text/plain', 'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        copyToCacheDirectory: true,
        multiple: false,
      });

      if (result.canceled || !result.assets?.[0]) {
        return;
      }

      const asset = result.assets[0];
      let rawText = '';

      if (asset.mimeType?.startsWith('text/') && asset.uri) {
        rawText = await FileSystem.readAsStringAsync(asset.uri);
      }

      if (!rawText.trim()) {
        rawText = [
          `File name: ${asset.name ?? 'Resume file'}`,
          'This document type cannot be fully parsed inside the demo app yet.',
          'Paste the resume text into the screen after upload so the AI coach can generate better questions.',
        ].join('\n');
      }

      setResumeProfile(rawText, asset.name);
    } catch (error) {
      Alert.alert('Upload failed', 'Please try again with a text resume or a smaller document.');
    } finally {
      setIsUploading(false);
    }
  };

  const loadResumeFromText = (rawText: string, fileName?: string) => {
    if (!rawText.trim()) {
      Alert.alert('Add resume text', 'Paste a short resume summary before continuing.');
      return;
    }
    setResumeProfile(rawText, fileName);
  };

  const startInterview = () => {
    if (!currentQuestion) {
      Alert.alert('Add a resume first', 'Upload a resume or paste resume text to generate interview questions.');
      return;
    }
    setMessages((current) => [
      ...current,
      createCoachMessage(`Interview mode started. First question: ${currentQuestion.question}`),
    ]);
  };

  const speakQuestion = () => {
    if (!currentQuestion) {
      Alert.alert('No question yet', 'Upload a resume first.');
      return;
    }
    Speech.speak(currentQuestion.question, {
      language: 'en-US',
      pitch: 1,
      rate: 0.95,
    });
  };

  const sendAnswer = (answer: string) => {
    const trimmed = answer.trim();
    if (!currentQuestion) {
      Alert.alert('No question yet', 'Upload a resume first.');
      return;
    }
    if (!trimmed) {
      Alert.alert('Type an answer', 'Write a short answer so the coach can review it.');
      return;
    }

    const review = evaluateAnswer(currentQuestion, trimmed, profile ?? parseResumeText(''));
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      role: 'user',
      text: trimmed,
      timestamp: Date.now(),
    };
    const coachMessage = createCoachMessage(
      `Score ${review.score}/10. Strengths: ${review.strengths[0]} Improvement: ${review.improvements[0]}`
    );

    setMessages((current) => [...current, userMessage, coachMessage]);
    setSessionRecords((current) => [
      ...current,
      {
        question: currentQuestion,
        answer: trimmed,
        review,
      },
    ]);
    setLatestReviewText(`${review.score}/10 - ${review.improvements[0]}`);
    if (currentQuestionIndex < questions.length - 1) {
      const nextIndex = currentQuestionIndex + 1;
      setCurrentQuestionIndex(nextIndex);
      setMessages((current) => [
        ...current,
        createCoachMessage(`Next question: ${questions[nextIndex].question}`),
      ]);
    }
  };

  const toggleVoiceMode = () => setIsVoiceMode((current) => !current);

  const resetSession = () => {
    setProfile(null);
    setQuestions([]);
    setCurrentQuestionIndex(0);
    setMessages([createCoachMessage('Upload a resume or paste its text to begin your interview practice.')]);
    setSessionRecords([]);
    setLatestReviewText('Your feedback will appear after the first answer.');
  };

  const value = useMemo<InterviewContextValue>(() => ({
    profile,
    questions,
    currentQuestionIndex,
    messages,
    sessionRecords,
    isUploading,
    isVoiceMode,
    latestReviewText,
    uploadResume,
    loadResumeFromText,
    startInterview,
    speakQuestion,
    sendAnswer,
    toggleVoiceMode,
    resetSession,
  }), [profile, questions, currentQuestionIndex, messages, sessionRecords, isUploading, isVoiceMode, latestReviewText]);

  return <InterviewContext.Provider value={value}>{children}</InterviewContext.Provider>;
}

export function useInterview() {
  const context = useContext(InterviewContext);
  if (!context) {
    throw new Error('useInterview must be used within InterviewProvider');
  }
  return context;
}
