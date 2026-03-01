/**
 * Evidence Badge Component
 * 
 * Displays evidence quality indicator with visual badge.
 */

'use client';

import React from 'react';
import { Shield, CheckCircle, AlertCircle } from 'lucide-react';

interface EvidenceBadgeProps {
  level: 'high' | 'moderate' | 'low' | 'insufficient';
  count?: number;
}

export function EvidenceBadge({ level, count }: EvidenceBadgeProps) {
  const configs = {
    high: {
      icon: <Shield className="w-4 h-4" />,
      bgColor: 'bg-green-100',
      textColor: 'text-green-800',
      label: 'High Quality Evidence',
    },
    moderate: {
      icon: <CheckCircle className="w-4 h-4" />,
      bgColor: 'bg-blue-100',
      textColor: 'text-blue-800',
      label: 'Moderate Quality Evidence',
    },
    low: {
      icon: <AlertCircle className="w-4 h-4" />,
      bgColor: 'bg-yellow-100',
      textColor: 'text-yellow-800',
      label: 'Limited Evidence',
    },
    insufficient: {
      icon: <AlertCircle className="w-4 h-4" />,
      bgColor: 'bg-gray-100',
      textColor: 'text-gray-600',
      label: 'Insufficient Data',
    },
  };

  const config = configs[level];

  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${config.bgColor} ${config.textColor}`}>
      {config.icon}
      <span>{config.label}</span>
      {count !== undefined && (
        <span className="bg-white/50 px-1.5 py-0.5 rounded">
          {count} sources
        </span>
      )}
    </div>
  );
}
