export type ResumeProfile = {
  title: string;
  summary: string;
  keywords: string[];
  highlights: string[];
  fileName?: string;
  rawText: string;
};

export type InterviewQuestion = {
  id: string;
  question: string;
  focus: string;
};

export type Review = {
  score: number;
  strengths: string[];
  improvements: string[];
  suggestedAnswer: string;
};

export type ChatMessage = {
  id: string;
  role: 'coach' | 'user';
  text: string;
  timestamp: number;
};

export type SessionRecord = {
  question: InterviewQuestion;
  answer: string;
  review: Review;
  audioUri?: string | null;
};
