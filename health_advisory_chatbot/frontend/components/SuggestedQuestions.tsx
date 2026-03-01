/**
 * Suggested Questions Component
 * 
 * Displays quick-question buttons for common health queries.
 */

'use client';

import React from 'react';
import { HelpCircle, Moon, Activity, Pill } from 'lucide-react';
import { SuggestedQuestion } from '../types';

interface SuggestedQuestionsProps {
  questions: SuggestedQuestion[];
  onQuestionClick: (question: string) => void;
}

const categoryIcons: Record<string, React.ReactNode> = {
  sleep: <Moon className="w-4 h-4" />,
  safety: <Activity className="w-4 h-4" />,
  medications: <Pill className="w-4 h-4" />,
  fall_prevention: <Activity className="w-4 h-4" />,
  cognitive: <HelpCircle className="w-4 h-4" />,
  general_health: <HelpCircle className="w-4 h-4" />,
};

const categoryLabels: Record<string, string> = {
  sleep: 'Sleep',
  safety: 'Safety',
  medications: 'Medications',
  fall_prevention: 'Fall Prevention',
  cognitive: 'Cognitive Health',
  general_health: 'General',
};

export function SuggestedQuestions({ questions, onQuestionClick }: SuggestedQuestionsProps) {
  if (questions.length === 0) {
    questions = [
      { question: 'How did I sleep last night?', category: 'sleep' },
      { question: 'What are my fall risk factors?', category: 'safety' },
      { question: 'How can I improve my mobility?', category: 'fall_prevention' },
    ];
  }

  return (
    <div className="space-y-3">
      <p className="text-gray-600 text-sm font-medium">Common questions:</p>
      <div className="space-y-2">
        {questions.map((item, index) => (
          <button
            key={index}
            onClick={() => onQuestionClick(item.question)}
            className="w-full text-left p-3 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:bg-blue-50 transition-all group"
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600 group-hover:bg-blue-200 transition-colors">
                {categoryIcons[item.category] || <HelpCircle className="w-4 h-4" />}
              </div>
              <div className="flex-1">
                <p className="text-gray-800 font-medium group-hover:text-blue-700 transition-colors">
                  {item.question}
                </p>
                <p className="text-xs text-gray-400">
                  {categoryLabels[item.category] || item.category}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
