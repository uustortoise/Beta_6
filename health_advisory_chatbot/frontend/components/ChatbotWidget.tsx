/**
 * Chatbot Widget Component
 * 
 * Floating chat button that expands into a chat window.
 * Designed for elderly users with accessibility features.
 */

'use client';

import React, { useState } from 'react';
import { MessageCircle, X } from 'lucide-react';
import { ChatWindow } from './ChatWindow';

interface ChatbotWidgetProps {
  elderId: string;
  elderName?: string;
}

export function ChatbotWidget({ elderId, elderName }: ChatbotWidgetProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
      {/* Chat Window */}
      {isOpen && (
        <div className="mb-4 animate-in slide-in-from-bottom-2 duration-200">
          <ChatWindow
            elderId={elderId}
            elderName={elderName}
            onClose={() => setIsOpen(false)}
          />
        </div>
      )}

      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center justify-center
          w-16 h-16 rounded-full
          shadow-lg transition-all duration-200
          ${isOpen 
            ? 'bg-red-500 hover:bg-red-600 rotate-90' 
            : 'bg-blue-600 hover:bg-blue-700 hover:scale-105'
          }
          focus:outline-none focus:ring-4 focus:ring-blue-300
        `}
        aria-label={isOpen ? 'Close chat' : 'Open health advisor chat'}
      >
        {isOpen ? (
          <X className="w-8 h-8 text-white" />
        ) : (
          <MessageCircle className="w-8 h-8 text-white" />
        )}
      </button>

      {/* Unread indicator (could be dynamic) */}
      {!isOpen && (
        <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-white text-xs font-bold">
          1
        </span>
      )}
    </div>
  );
}
