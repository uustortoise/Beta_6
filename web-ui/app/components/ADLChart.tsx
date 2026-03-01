

'use client';

import { useState } from 'react';
import {
    ComposedChart,
    Line,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    BarChart,
    LineChart
} from 'recharts';
import { ADLHistory } from '../lib/data';
import { AlertTriangle } from 'lucide-react';

interface ADLChartProps {
    data: ADLHistory;
}

export default function ADLChart({ data }: ADLChartProps) {
    const [view, setView] = useState<'activity' | 'mobility' | 'night' | 'diversity'>('activity');
    const records = data.daily_records || [];
    const anomalies = data.anomalies || [];

    if (!records || records.length === 0) {
        return (
            <div className="h-full flex items-center justify-center text-gray-400">
                No data available
            </div>
        );
    }

    return (
        <div className="h-full w-full flex flex-col">
            {/* Anomaly Alert Banner */}
            {anomalies.length > 0 && (
                <div className="mb-2 bg-red-50 border border-red-100 rounded px-2 py-1.5 flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-red-500 shrink-0" />
                    <span className="text-xs text-red-700 font-medium truncate">{anomalies[0]}</span>
                    {anomalies.length > 1 && <span className="text-[10px] bg-red-200 text-red-800 px-1 rounded">+{anomalies.length - 1}</span>}
                </div>
            )}

            {/* View Toggles */}
            <div className="flex justify-end mb-2">
                <div className="flex bg-gray-100 dark:bg-gray-700 rounded p-1 gap-0.5">
                    {['activity', 'mobility', 'night', 'diversity'].map((v) => (
                        <button
                            key={v}
                            onClick={() => setView(v as any)}
                            className={`px-2 py-1 text-[10px] uppercase font-semibold rounded transition-colors ${view === v ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 hover:text-gray-900 dark:text-gray-400'}`}
                        >
                            {v}
                        </button>
                    ))}
                </div>
            </div>

            <div className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    {view === 'activity' ? (
                        <BarChart data={records} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })} />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', fontSize: '12px' }} />
                            <Legend iconSize={8} wrapperStyle={{ fontSize: '10px', bottom: 0 }} />
                            <Bar dataKey="count_sleep" name="Sleep" stackId="a" fill="#6366f1" />
                            <Bar dataKey="count_normal" name="Normal Use" stackId="a" fill="#22c55e" />
                            <Bar dataKey="count_cooking" name="Cooking" stackId="a" fill="#f97316" />
                            <Bar dataKey="count_bath" name="Bath" stackId="a" fill="#06b6d4" />
                        </BarChart>
                    ) : view === 'mobility' ? (
                        <LineChart data={records} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })} />
                            <YAxis domain={[0, 1.2]} tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', fontSize: '12px' }} />
                            <Line type="monotone" dataKey="mobility_index" name="Mobility Index" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 3 }} />
                        </LineChart>
                    ) : view === 'night' ? (
                        <BarChart data={records} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })} />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', fontSize: '12px' }} />
                            <Legend iconSize={8} wrapperStyle={{ fontSize: '10px', bottom: 0 }} />
                            <Bar dataKey="night_activity_count" name="Night Events" fill="#6b7280" />
                        </BarChart>
                    ) : (
                        <BarChart data={records} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f3f4f6" />
                            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })} />
                            <YAxis tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', fontSize: '12px' }} />
                            <Bar dataKey="diversity_score" name="Diversity Score" fill="#10b981" />
                        </BarChart>
                    )}
                </ResponsiveContainer>
            </div>
        </div>
    );
}
