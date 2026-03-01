/**
 * Message Bubble Component
 * 
 * Displays chat messages with support for:
 * - Citations and evidence indicators
 * - Risk alerts
 * - Elder-friendly formatting
 */

'use client';

import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, FileText } from 'lucide-react';
import { Message, Citation } from '../types';

interface MessageBubbleProps {
  message: Message;
  isLatest?: boolean;
}

export function MessageBubble({ message, isLatest }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const [showCitations, setShowCitations] = useState(false);

  return (
    <div
      className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}
      role="article"
      aria-label={`${isUser ? 'Your' : 'Advisor'} message`}
    >
      {/* Avatar */}
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-blue-100' : 'bg-green-100'
        }`}
      >
        {isUser ? (
          <User className="w-5 h-5 text-blue-600" />
        ) : (
          <Bot className="w-5 h-5 text-green-600" />
        )}
      </div>

      {/* Message Content */}
      <div className={`max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-lg leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-white border border-gray-200 text-gray-800 rounded-bl-md shadow-sm'
          }`}
        >
          {message.content}
        </div>

        {/* Risk Alerts */}
        {!isUser && message.riskAlerts && message.riskAlerts.length > 0 && (
          <div className="mt-2 bg-red-50 border border-red-200 rounded-lg p-3">
            <p className="text-red-800 font-semibold text-sm flex items-center gap-2">
              <span className="text-red-500">⚠️</span>
              Health Alert
            </p>
            <ul className="mt-1 space-y-1">
              {message.riskAlerts.map((alert, idx) => (
                <li key={idx} className="text-red-700 text-sm">
                  • {alert}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Citations Toggle */}
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setShowCitations(!showCitations)}
              className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-800 transition-colors"
            >
              <FileText className="w-4 h-4" />
              <span>{message.citations.length} sources</span>
              {showCitations ? (
                <ChevronUp className="w-4 h-4" />
              ) : (
                <ChevronDown className="w-4 h-4" />
              )}
            </button>

            {showCitations && (
              <div className="mt-2 space-y-2 bg-gray-50 rounded-lg p-3">
                {message.citations.map((citation, idx) => (
                  <CitationItem key={idx} citation={citation} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Evidence Summary */}
        {!isUser && message.evidenceSummary && (
          <p className="mt-1 text-xs text-gray-500 italic">
            {message.evidenceSummary}
          </p>
        )}

        {/* Timestamp */}
        <p className="mt-1 text-xs text-gray-400">
          {new Date(message.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  );
}

function CitationItem({ citation }: { citation: Citation }) {
  const evidenceColors = {
    systematic_review: 'bg-purple-100 text-purple-800',
    rct: 'bg-green-100 text-green-800',
    cohort_study: 'bg-blue-100 text-blue-800',
    case_control: 'bg-yellow-100 text-yellow-800',
    expert_opinion: 'bg-gray-100 text-gray-800',
    clinical_guideline: 'bg-indigo-100 text-indigo-800',
    manufacturer_data: 'bg-gray-100 text-gray-600',
  };

  const evidenceLabels = {
    systematic_review: 'Systematic Review',
    rct: 'RCT',
    cohort_study: 'Cohort Study',
    case_control: 'Case-Control',
    expert_opinion: 'Expert Opinion',
    clinical_guideline: 'Guideline',
    manufacturer_data: 'Manufacturer',
  };

  return (
    <div className="text-sm border-l-2 border-blue-300 pl-3 py-1">
      <p className="font-medium text-gray-800 line-clamp-2">{citation.title}</p>
      <div className="flex items-center gap-2 mt-1">
        <span className={`text-xs px-2 py-0.5 rounded-full ${evidenceColors[citation.evidenceLevel]}`}>
          {evidenceLabels[citation.evidenceLevel]}
        </span>
        {citation.year && (
          <span className="text-xs text-gray-500">{citation.year}</span>
        )}
        {citation.confidenceScore && (
          <span className="text-xs text-gray-500">
            {Math.round(citation.confidenceScore)}% confidence
          </span>
        )}
      </div>
      {citation.url && (
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-blue-600 hover:underline mt-1 inline-block"
        >
          View source →
        </a>
      )}
    </div>
  );
}
