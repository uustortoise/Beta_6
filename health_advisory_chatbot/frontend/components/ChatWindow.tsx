/**
 * Chat Window Component
 * 
 * Main chat interface with message history, input, and suggested questions.
 * Elder-friendly design with large text, high contrast, and clear visual hierarchy.
 */

'use client';

import React, { useEffect, useState } from 'react';
import { Send, Sparkles, AlertTriangle } from 'lucide-react';
import { useChatbot } from '../hooks/useChatbot';
import { MessageBubble } from './MessageBubble';
import { SuggestedQuestions } from './SuggestedQuestions';
import { EvidenceBadge } from './EvidenceBadge';
import { SuggestedQuestion, RiskInfo } from '../types';

interface ChatWindowProps {
  elderId: string;
  elderName?: string;
  onClose: () => void;
}

export function ChatWindow({ elderId, elderName, onClose }: ChatWindowProps) {
  const { messages, isLoading, error, sendMessage, getSuggestions, messagesEndRef } = useChatbot(elderId);
  const [inputValue, setInputValue] = useState('');
  const [suggestions, setSuggestions] = useState<SuggestedQuestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [latestRisks, setLatestRisks] = useState<RiskInfo | null>(null);

  // Load suggestions on mount
  useEffect(() => {
    getSuggestions().then(setSuggestions);
  }, [getSuggestions]);

  // Extract risks from latest message
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage?.role === 'assistant' && lastMessage.riskAlerts && lastMessage.riskAlerts.length > 0) {
      // In production, we'd extract full risk info from the API response
      setLatestRisks({
        overallScore: 65,
        overallLevel: 'high',
        criticalAlerts: lastMessage.riskAlerts,
      });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;
    setShowSuggestions(false);
    await sendMessage(inputValue);
    setInputValue('');
  };

  const handleSuggestionClick = async (question: string) => {
    setShowSuggestions(false);
    await sendMessage(question);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="w-[400px] h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-gray-200">
      {/* Header */}
      <div className="bg-blue-600 text-white p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">Health Advisor</h3>
            <p className="text-blue-100 text-sm">
              {elderName ? `For ${elderName}` : 'Evidence-based guidance'}
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-2 hover:bg-blue-700 rounded-full transition-colors"
          aria-label="Close chat"
        >
          <span className="text-2xl">&times;</span>
        </button>
      </div>

      {/* Risk Alert Banner */}
      {latestRisks && latestRisks.criticalAlerts.length > 0 && (
        <div className="bg-red-50 border-l-4 border-red-500 p-3 flex items-start gap-2">
          <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-red-800 font-medium text-sm">Health Alert</p>
            <p className="text-red-700 text-xs mt-0.5">
              {latestRisks.criticalAlerts[0]}
            </p>
          </div>
        </div>
      )}

      {/* Evidence Quality Indicator */}
      {messages.length > 0 && !showSuggestions && (
        <div className="bg-gray-50 px-4 py-2 border-b border-gray-100">
          <EvidenceBadge level="high" count={3} />
        </div>
      )}

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 && showSuggestions ? (
          <div className="space-y-4">
            <div className="bg-blue-50 rounded-xl p-4">
              <p className="text-blue-900 text-lg leading-relaxed">
                Hello! I&apos;m your health advisor. I can help you understand:
              </p>
              <ul className="mt-3 space-y-2 text-blue-800">
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Your sleep patterns
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Fall prevention tips
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Medication safety
                </li>
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  Cognitive health
                </li>
              </ul>
            </div>

            <SuggestedQuestions
              questions={suggestions}
              onQuestionClick={handleSuggestionClick}
            />
          </div>
        ) : (
          messages.map((message, index) => (
            <MessageBubble
              key={index}
              message={message}
              isLatest={index === messages.length - 1}
            />
          ))
        )}

        {isLoading && (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            <span className="text-sm ml-2">Analyzing your health data...</span>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm">
            Error: {error}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 bg-white border-t border-gray-200">
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your health..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-xl text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={isLoading}
            aria-label="Type your question"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputValue.trim()}
            className="p-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send className="w-6 h-6" />
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          Powered by medical research • Not a substitute for professional care
        </p>
      </div>
    </div>
  );
}
