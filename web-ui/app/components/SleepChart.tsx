'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { useState } from 'react';

interface SleepChartProps {
    data: {
        sleep_periods: any[];
        seven_day_average?: {
            duration_hours: number;
            efficiency: number;
            score: number;
        };
        daily_history?: any[];
        stage_breakdown?: {
            Light: number;
            Deep: number;
            REM: number;
            Awake: number;
        };
        grade?: string;
    };
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042'];

export function SleepChart({ data }: SleepChartProps) {
    const [view, setView] = useState<'trend' | 'stages'>('trend');

    // Safe access to data
    const history = data.daily_history || [];
    const stages = data.stage_breakdown ? [
        { name: 'Light', value: data.stage_breakdown.Light },
        { name: 'Deep', value: data.stage_breakdown.Deep },
        { name: 'REM', value: data.stage_breakdown.REM },
        { name: 'Awake', value: data.stage_breakdown.Awake },
    ] : [];

    const average = data.seven_day_average;
    const grade = data.grade || 'N/A';

    return (
        <div className="h-full w-full flex flex-col">
            {/* Header / Toggle */}
            <div className="flex justify-between items-start mb-2">
                <div className="flex gap-4 text-xs font-medium">
                    {average && (
                        <div className="bg-indigo-50 dark:bg-indigo-900/30 px-3 py-1.5 rounded-lg">
                            <span className="block text-indigo-400 text-[10px] uppercase tracking-wider">7-Day Avg</span>
                            <span className="text-lg font-bold text-indigo-600 dark:text-indigo-400">{average.duration_hours}h</span>
                        </div>
                    )}
                    {grade && (
                        <div className="bg-purple-50 dark:bg-purple-900/30 px-3 py-1.5 rounded-lg">
                            <span className="block text-purple-400 text-[10px] uppercase tracking-wider">Grade</span>
                            <span className="text-lg font-bold text-purple-600 dark:text-purple-400">{grade}</span>
                        </div>
                    )}
                </div>

                <div className="flex bg-gray-100 dark:bg-gray-700 rounded p-1">
                    <button
                        onClick={() => setView('trend')}
                        className={`px-3 py-1 text-xs rounded transition-colors ${view === 'trend' ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 hover:text-gray-900 dark:text-gray-400'}`}
                    >
                        Trend
                    </button>
                    <button
                        onClick={() => setView('stages')}
                        className={`px-3 py-1 text-xs rounded transition-colors ${view === 'stages' ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm' : 'text-gray-500 hover:text-gray-900 dark:text-gray-400'}`}
                    >
                        Stages
                    </button>
                </div>
            </div>

            {/* Charts */}
            <div className="flex-1 min-h-0 relative flex items-center justify-center">
                <ResponsiveContainer width="100%" height="100%">
                    {view === 'trend' ? (
                        <BarChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                            <XAxis
                                dataKey="date"
                                stroke="#9CA3AF"
                                fontSize={10}
                                tickLine={false}
                                axisLine={false}
                                tickFormatter={(value) => {
                                    const d = new Date(value);
                                    return `${d.getMonth() + 1}/${d.getDate()}`;
                                }}
                            />
                            <YAxis stroke="#9CA3AF" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                labelStyle={{ color: '#6B7280', fontSize: '12px' }}
                                formatter={(value: number) => [value.toFixed(2), 'Duration (h)']}
                            />
                            <Bar dataKey="duration_hours" name="Duration (h)" fill="#6366f1" radius={[4, 4, 0, 0]} barSize={20} />
                        </BarChart>
                    ) : (
                        <PieChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <Pie
                                data={stages}
                                innerRadius={50}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                                label={({ cx, cy, midAngle, innerRadius, outerRadius, percent, index, name }) => {
                                    const RADIAN = Math.PI / 180;
                                    const radius = outerRadius + 20;
                                    const x = cx + radius * Math.cos(-midAngle * RADIAN);
                                    const y = cy + radius * Math.sin(-midAngle * RADIAN);

                                    return (
                                        <text
                                            x={x}
                                            y={y}
                                            fill={COLORS[index % COLORS.length]}
                                            textAnchor={x > cx ? 'start' : 'end'}
                                            dominantBaseline="central"
                                            className="text-[10px] font-medium"
                                        >
                                            {`${name} ${stages[index].value.toFixed(0)}%`}
                                        </text>
                                    );
                                }}
                            >
                                {stages.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                            />
                        </PieChart>
                    )}
                </ResponsiveContainer>
            </div>

            {/* Legend removed in favor of direct labels */}
        </div>
    );
}
