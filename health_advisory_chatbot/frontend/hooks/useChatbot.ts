/**
 * Custom hook for chatbot state and API interaction
 */

import { useState, useCallback, useRef } from 'react';
import { Message, ApiChatResponse, ApiCitation, SuggestedQuestion } from '../types';

const API_BASE_URL = process.env.NEXT_PUBLIC_CHATBOT_API_URL || '/api';

function mapApiCitation(citation: ApiCitation) {
  return {
    sourceType: citation.source_type,
    title: citation.title,
    authors: citation.authors,
    journal: citation.journal,
    year: citation.year,
    doi: citation.doi,
    url: citation.url,
    pmid: citation.pmid,
    evidenceLevel: citation.evidence_level,
    confidenceScore: citation.confidence_score,
  };
}

export function useChatbot(elderId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const sendMessage = useCallback(async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Add user message immediately
    const userMessage: Message = {
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          elder_id: elderId,
          message: content.trim(),
          session_id: sessionId,
          include_medical_history: true,
          include_adl_data: true,
          include_icope_data: true,
          include_sleep_data: true,
          require_citations: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ApiChatResponse = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'Failed to get response');
      }

      // Update session ID
      if (data.session_id) {
        setSessionId(data.session_id);
      }

      // Add assistant message
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.message.content,
        timestamp: data.message.timestamp,
        citations: (data.message.citations || []).map(mapApiCitation),
        evidenceSummary: data.message.evidence_summary,
        riskAlerts: data.message.risk_alerts,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An error occurred';
      setError(errorMessage);
      
      // Add error message
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'I apologize, but I encountered an error. Please try again.',
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setIsLoading(false);
      setTimeout(scrollToBottom, 100);
    }
  }, [elderId, sessionId, isLoading, scrollToBottom]);

  const getSuggestions = useCallback(async (): Promise<SuggestedQuestion[]> => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/suggestions?elder_id=${elderId}`);
      if (!response.ok) return [];
      
      const data = await response.json();
      return data.suggestions || [];
    } catch {
      return [];
    }
  }, [elderId]);

  const clearChat = useCallback(() => {
    setMessages([]);
    setSessionId(null);
    setError(null);
  }, []);

  return {
    messages,
    sessionId,
    isLoading,
    error,
    sendMessage,
    getSuggestions,
    clearChat,
    messagesEndRef,
  };
}
