import { InterviewQuestion, ResumeProfile, Review } from '../types';

const rolePrompts = [
  'Tell me about yourself and why this role fits your background.',
  'Describe a project from your resume that best shows your strengths.',
  'What was a difficult problem you solved, and how did you approach it?',
  'How do you work with teammates, stakeholders, or clients under pressure?',
  'Why do you want this role, and what makes you a strong candidate?',
];

const actionWords = [
  'built',
  'created',
  'improved',
  'led',
  'designed',
  'managed',
  'developed',
  'optimized',
  'presented',
  'collaborated',
];

const stopWords = new Set([
  'the', 'and', 'with', 'your', 'from', 'that', 'this', 'for', 'you', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'their', 'they', 'them', 'into', 'about', 'using', 'used', 'been', 'more', 'than', 'can', 'all', 'our', 'its', 'also', 'but', 'not', 'when', 'who', 'what', 'how', 'why', 'role', 'work', 'team', 'project', 'experience', 'resume', 'skill', 'skills', 'help', 'like'
]);

function normalize(text: string) {
  return text.replace(/\s+/g, ' ').trim();
}

function tokenise(text: string) {
  return normalize(text)
    .toLowerCase()
    .split(/[^a-z0-9+.-]+/)
    .filter(Boolean)
    .filter((token) => token.length > 2 && !stopWords.has(token));
}

function unique(values: string[]) {
  return [...new Set(values)];
}

export function parseResumeText(rawText: string, fileName?: string): ResumeProfile {
  const normalized = normalize(rawText);
  const tokens = tokenise(normalized);
  const keywordCounts = tokens.reduce<Record<string, number>>((acc, token) => {
    acc[token] = (acc[token] ?? 0) + 1;
    return acc;
  }, {});

  const keywords = Object.entries(keywordCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([keyword]) => keyword);

  const highlights = unique(
    normalized
      .split(/[.\n]/)
      .map((part) => part.trim())
      .filter((part) => part.length > 24)
      .slice(0, 4)
  );

  const title = fileName ? fileName.replace(/\.[^.]+$/, '') : 'Resume Profile';
  const summary = keywords.length
    ? `Detected strengths around ${keywords.slice(0, 3).join(', ')}.`
    : 'No strong keywords detected yet. Add more resume text for richer interview questions.';

  return {
    title,
    summary,
    keywords,
    highlights,
    fileName,
    rawText: normalized,
  };
}

export function buildQuestions(profile: ResumeProfile): InterviewQuestion[] {
  const keywordFocus = profile.keywords.length > 0 ? profile.keywords : ['experience', 'leadership', 'problem solving'];
  const dynamicQuestions = keywordFocus.slice(0, 4).map((keyword, index) => ({
    id: `dynamic-${index}`,
    question: `Explain how you used ${keyword} in a real project or achievement.`,
    focus: keyword,
  }));

  const staticQuestions: InterviewQuestion[] = rolePrompts.map((question, index) => ({
    id: `static-${index}`,
    question,
    focus: ['intro', 'project', 'challenge', 'communication', 'motivation'][index] ?? 'general',
  }));

  return uniqueQuestions([...dynamicQuestions, ...staticQuestions]).slice(0, 6);
}

function uniqueQuestions(questions: InterviewQuestion[]) {
  const seen = new Set<string>();
  return questions.filter((question) => {
    if (seen.has(question.question)) {
      return false;
    }
    seen.add(question.question);
    return true;
  });
}

export function evaluateAnswer(question: InterviewQuestion, answer: string, profile: ResumeProfile): Review {
  const text = normalize(answer);
  const lower = text.toLowerCase();
  const matchedKeywords = profile.keywords.filter((keyword) => lower.includes(keyword.toLowerCase()));
  const hasActionWords = actionWords.some((word) => lower.includes(word));
  const hasNumbers = /\d/.test(text);
  const wordCount = text.split(/\s+/).filter(Boolean).length;

  let score = 4;
  if (wordCount > 25) score += 2;
  if (wordCount > 60) score += 1;
  if (matchedKeywords.length > 0) score += 2;
  if (hasActionWords) score += 1;
  if (hasNumbers) score += 1;
  if (question.focus !== 'general' && lower.includes(question.focus.toLowerCase())) score += 1;
  score = Math.max(1, Math.min(10, score));

  const strengths: string[] = [];
  if (matchedKeywords.length > 0) {
    strengths.push(`You connected the answer to ${matchedKeywords.slice(0, 2).join(', ')}.`);
  }
  if (hasActionWords) {
    strengths.push('You used impact-oriented verbs that make the answer sound active.');
  }
  if (hasNumbers) {
    strengths.push('You included a measurable detail, which makes the answer stronger.');
  }
  if (strengths.length === 0) {
    strengths.push('The answer is clear and easy to follow.');
  }

  const improvements: string[] = [];
  if (wordCount < 25) {
    improvements.push('Add more context so the interviewer can see the situation, your action, and the outcome.');
  }
  if (matchedKeywords.length === 0) {
    improvements.push('Mention one or two keywords from your resume to show direct relevance.');
  }
  if (!hasNumbers) {
    improvements.push('Add a result, percentage, or count to show impact.');
  }
  if (!hasActionWords) {
    improvements.push('Use stronger action verbs such as built, led, improved, or delivered.');
  }
  if (improvements.length === 0) {
    improvements.push('Try tightening the answer into a strong STAR-style response: situation, task, action, result.');
  }

  const suggestedAnswer = buildSuggestedAnswer(question, profile, matchedKeywords, wordCount);

  return {
    score,
    strengths,
    improvements,
    suggestedAnswer,
  };
}

function buildSuggestedAnswer(question: InterviewQuestion, profile: ResumeProfile, matchedKeywords: string[], wordCount: number) {
  const keywordLine = matchedKeywords.length > 0
    ? `I will reference ${matchedKeywords.slice(0, 2).join(' and ')} from my background.`
    : 'I will connect my background to the role with a specific example.';

  const resultLine = wordCount > 25
    ? 'I will finish with a measurable result so the outcome is clear.'
    : 'I will add the challenge, what I did, and the result so the answer feels complete.';

  const profileLine = profile.highlights[0]
    ? `A useful example from my resume is: ${profile.highlights[0]}`
    : 'A useful example from my resume would show a project, internship, or team result.';

  return [
    `For ${question.focus}, I would answer by giving a concise example from my experience.`,
    keywordLine,
    profileLine,
    resultLine,
  ].join(' ');
}
