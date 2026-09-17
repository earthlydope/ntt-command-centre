/**
 * The questions Ask AI Expert offers before you type.
 *
 * Every one of these has a deterministic answer on the server
 * (`api/semantic/answers.py`) — so a chip never returns "cannot answer right
 * now" when the language service is busy. Change a string here and the same
 * string must change there: the match is on the text.
 *
 * Phrased the way the persona would say it, in plain words. No field names.
 */
import type { PersonaKey } from "../api/types";

export const SUGGESTIONS: Record<PersonaKey, string[]> = {
  ae: [
    "Which of my accounts has the most at stake?",
    "How does my win rate compare with the book?",
    "Where is my pipeline concentrated by portfolio?",
    "Which of my deals have gone quiet?",
    "What should I do first today?",
  ],
  manager: [
    "Which reps in my pod hold the most open pipeline?",
    "How does win rate vary across my pod?",
    "Where is the pod's pipeline by line of business?",
    "Who should I coach first?",
    "How much of the pod's pipeline has stopped moving?",
  ],
  executive: [
    "Which line of business has the weakest margin?",
    "Which accounts are biggest by gross profit?",
    "How has closed-won tracked by month?",
    "Where is open pipeline concentrated by industry?",
    "Are we on track against plan this quarter?",
    "What needs fixing, and what is it worth?",
  ],
};
