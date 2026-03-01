'use client';

import { useEffect, useState } from 'react';
import { Brain, AlertCircle, CheckCircle, TrendingUp, TrendingDown, Clock } from 'lucide-react';

interface Anomaly {
  id: number;
  detection_date: string;
  anomaly_type: string;
  anomaly_score: number;
  description: string;
  baseline_value: string;
  observed_value: string;
}

interface RoutineHealthProps {
  elderId: string;
}

const ANOMALY_ICONS: Record<string, any> = {
  late_wakeup: Clock,
  early_wakeup: Clock,
  increased_bathroom_visits: TrendingUp,
  reduced_bathroom_visits: TrendingDown,
  reduced_activity: TrendingDown,
  increased_wandering: TrendingUp,
};

export default function RoutineHealth({ elderId }: RoutineHealthProps) {
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [historyDays, setHistoryDays] = useState(0);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      setLoading(true);
      try {
        const res = await fetch(`/api/anomalies?elderId=${elderId}&days=7`);
        if (!res.ok) throw new Error('Failed to fetch');
        
        const data = await res.json();
        setAnomalies(data.anomalies || []);
        setHistoryDays(data.history_days || 0);
        setMessage(data.message || '');
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    
    fetchData();
  }, [elderId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-24 text-gray-400">
        <div className="animate-spin h-5 w-5 border-2 border-gray-300 border-t-purple-500 rounded-full mr-2" />
        Analyzing patterns...
      </div>
    );
  }

  // Show status based on history availability
  if (historyDays < 7) {
    return (
      <div className="border-2 border-dashed border-gray-200 rounded-lg p-6 text-center">
        <Brain className="h-10 w-10 mx-auto mb-3 text-gray-300" />
        <h4 className="font-medium text-gray-700 mb-1">Building Routine Profile</h4>
        <p className="text-sm text-gray-500">{message}</p>
        <div className="mt-3 flex items-center justify-center gap-2">
          <div className="h-2 w-32 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-purple-500 transition-all"
              style={{ width: `${(historyDays / 7) * 100}%` }}
            />
          </div>
          <span className="text-xs text-gray-500">{historyDays}/7 days</span>
        </div>
      </div>
    );
  }

  // Show anomalies or healthy status
  if (anomalies.length === 0) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-center gap-3 dark:bg-green-900/20 dark:border-green-800">
        <CheckCircle className="h-6 w-6 text-green-500" />
        <div>
          <h4 className="font-medium text-green-800 dark:text-green-200">Routine Normal</h4>
          <p className="text-sm text-green-600 dark:text-green-300">
            All daily patterns within expected ranges
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {anomalies.map((anomaly) => {
        const Icon = ANOMALY_ICONS[anomaly.anomaly_type] || AlertCircle;
        const severity = anomaly.anomaly_score > 0.7 ? 'high' : anomaly.anomaly_score > 0.4 ? 'medium' : 'low';
        
        return (
          <div 
            key={anomaly.id}
            className={`rounded-lg p-4 border ${
              severity === 'high' 
                ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' 
                : severity === 'medium'
                ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800'
                : 'bg-blue-50 border-blue-200 dark:bg-blue-900/20 dark:border-blue-800'
            }`}
          >
            <div className="flex items-start gap-3">
              <Icon className={`h-5 w-5 mt-0.5 ${
                severity === 'high' ? 'text-red-500' : severity === 'medium' ? 'text-yellow-500' : 'text-blue-500'
              }`} />
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {anomaly.description}
                  </h4>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    severity === 'high' 
                      ? 'bg-red-100 text-red-700' 
                      : severity === 'medium'
                      ? 'bg-yellow-100 text-yellow-700'
                      : 'bg-blue-100 text-blue-700'
                  }`}>
                    {Math.round(anomaly.anomaly_score * 100)}% deviation
                  </span>
                </div>
                <div className="mt-1 text-sm text-gray-600 dark:text-gray-400">
                  <span>Observed: <strong>{anomaly.observed_value}</strong></span>
                  <span className="mx-2">•</span>
                  <span>Normal: {anomaly.baseline_value}</span>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  {anomaly.detection_date}
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
