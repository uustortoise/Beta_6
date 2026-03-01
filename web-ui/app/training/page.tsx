'use client';

import { useState, useEffect, useMemo } from 'react';
import { Sidebar } from '../components/Sidebar';
import ActivityTimeline from '../components/ActivityTimeline';
import { Play, CheckCircle, Clock } from 'lucide-react';

interface TrainingStatus {
    status: 'idle' | 'running' | 'success' | 'error';
    progress: number;
    message: string;
    last_updated: string;
}

interface TrainingHistoryItem {
    timestamp: string;
    model_type: string;
    epochs: number;
    accuracy: number;
    status: string;
}

export default function TrainingPage() {
    const [status, setStatus] = useState<TrainingStatus>({ status: 'idle', progress: 0, message: 'Ready', last_updated: '' });
    const [history, setHistory] = useState<TrainingHistoryItem[]>([]);
    const [isPolling, setIsPolling] = useState(false);

    // Candidates State
    const [candidates, setCandidates] = useState<any[]>([]);
    const [selectedDate, setSelectedDate] = useState<string>('all');

    // Initial Fetch
    useEffect(() => {
        fetchStatus();
        fetchCandidates();
    }, []);

    // Polling Effect
    useEffect(() => {
        if (status.status === 'running' || isPolling) {
            const interval = setInterval(fetchStatus, 1000);
            return () => clearInterval(interval);
        }
    }, [status.status, isPolling]);

    const fetchStatus = async () => {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.current) {
                setStatus(data.current);
                if (data.current.status !== 'running') {
                    setIsPolling(false);
                }
            }
            if (data.history) {
                setHistory(data.history);
            }
        } catch (error) {
            console.error("Failed to fetch status:", error);
        }
    };

    const fetchCandidates = async () => {
        try {
            const res = await fetch('/api/candidates');
            const data = await res.json();
            if (data.candidates) {
                setCandidates(data.candidates);
            }
        } catch (error) {
            console.error("Failed to fetch candidates:", error);
        }
    };

    const startTraining = async () => {
        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ epochs: 10 })
            });

            if (res.ok) {
                setIsPolling(true);
                setStatus(prev => ({ ...prev, status: 'running', message: 'Starting...' }));
            } else {
                alert("Failed to start training.");
            }
        } catch (e) {
            alert("Error triggering training.");
        }
    };

    // Filter Logic
    const uniqueDates = useMemo(() => {
        const dates = new Set(candidates.map(c => new Date(c.timestamp).toISOString().split('T')[0]));
        return Array.from(dates).sort().reverse();
    }, [candidates]);

    const filteredCandidates = useMemo(() => {
        if (selectedDate === 'all') return candidates;
        return candidates.filter(c => c.timestamp.startsWith(selectedDate));
    }, [candidates, selectedDate]);

    const downloadCSV = () => {
        if (filteredCandidates.length === 0) return;

        const headers = ["Timestamp", "Resident", "Room", "Activity", "Confidence", "Reason"];
        const rows = filteredCandidates.map(c => [
            c.timestamp,
            c.resident_id,
            c.room,
            c.activity,
            c.confidence.toFixed(2),
            c.reason
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(row => row.join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `review_candidates_${selectedDate}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white mb-8">Model Studio</h1>



                    {/* Data Review Queue */}
                    <div className="mb-8 rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800 overflow-hidden">
                        <div className="border-b border-gray-200 bg-gray-50/50 px-6 py-4 dark:border-gray-700 dark:bg-gray-800 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                            <div>
                                <h3 className="text-base font-semibold leading-6 text-gray-900 dark:text-white">Data Review Queue</h3>
                                <p className="text-xs text-gray-500 mt-1">Low confidence predictions and anomalies needing review.</p>
                            </div>

                            <div className="flex items-center gap-3">
                                {/* Date Selector */}
                                <select
                                    value={selectedDate}
                                    onChange={(e) => setSelectedDate(e.target.value)}
                                    className="block w-40 rounded-md border-0 py-1.5 pl-3 pr-10 text-gray-900 ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-indigo-600 sm:text-sm sm:leading-6 dark:bg-gray-700 dark:text-white dark:ring-gray-600"
                                >
                                    <option value="all">All Dates</option>
                                    {uniqueDates.map(date => (
                                        <option key={date} value={date}>{date}</option>
                                    ))}
                                </select>

                                <button
                                    onClick={downloadCSV}
                                    disabled={filteredCandidates.length === 0}
                                    className="text-sm bg-white border border-gray-300 text-gray-700 px-3 py-1.5 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-gray-700 dark:border-gray-600 dark:text-gray-200"
                                >
                                    Download CSV ({filteredCandidates.length})
                                </button>
                            </div>
                        </div>

                        {/* Visualization of Review Items */}
                        {filteredCandidates.length > 0 && (
                            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
                                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Timeline Distribution</h4>
                                <ActivityTimeline data={filteredCandidates} />
                            </div>
                        )}

                        <div className="max-h-[300px] overflow-y-auto">
                            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                                <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0 z-10">
                                    <tr>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Resident</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Room</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Activity</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Conf.</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Reason</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-900 dark:divide-gray-700">
                                    {filteredCandidates.length > 0 ? (
                                        filteredCandidates.map((c, idx) => (
                                            <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                                <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                                                    {new Date(c.timestamp).toLocaleTimeString()}
                                                </td>
                                                <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                                                    {c.resident_id}
                                                </td>
                                                <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                                                    {c.room}
                                                </td>
                                                <td className="px-6 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                                                    {c.activity}
                                                </td>
                                                <td className="px-6 py-3 whitespace-nowrap text-sm text-gray-500">
                                                    {(c.confidence * 100).toFixed(0)}%
                                                </td>
                                                <td className="px-6 py-3 whitespace-nowrap text-sm">
                                                    <span className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium ring-1 ring-inset ${c.reason === 'Anomaly'
                                                        ? 'bg-red-50 text-red-700 ring-red-600/20 dark:bg-red-900/20 dark:text-red-400'
                                                        : 'bg-yellow-50 text-yellow-800 ring-yellow-600/20 dark:bg-yellow-900/20 dark:text-yellow-400'
                                                        }`}>
                                                        {c.reason}
                                                    </span>
                                                </td>
                                            </tr>
                                        ))
                                    ) : (
                                        <tr>
                                            <td colSpan={6} className="px-6 py-8 text-center text-sm text-gray-500">
                                                No review candidates found for selected date.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* History Table */}
                    <div className="rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800 overflow-hidden">
                        <div className="border-b border-gray-200 bg-gray-50/50 px-6 py-4 dark:border-gray-700 dark:bg-gray-800">
                            <h3 className="text-base font-semibold leading-6 text-gray-900 dark:text-white">Training History</h3>
                        </div>
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                                <thead className="bg-gray-50 dark:bg-gray-800">
                                    <tr>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Epochs</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-900 dark:divide-gray-700">
                                    {history.length > 0 ? (
                                        history.map((item, idx) => (
                                            <tr key={idx} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-300">
                                                    {new Date(item.timestamp).toLocaleString()}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                                                    {item.model_type}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                    {item.epochs}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                    <span className="inline-flex items-center rounded-md bg-green-50 px-2 py-1 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-600/20">
                                                        {(item.accuracy * 100).toFixed(1)}%
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                    <div className="flex items-center gap-1.5">
                                                        <CheckCircle className="h-4 w-4 text-green-500" />
                                                        <span className="text-green-700 dark:text-green-400">Success</span>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))
                                    ) : (
                                        <tr>
                                            <td colSpan={5} className="px-6 py-8 text-center text-sm text-gray-500">
                                                No training history available.
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
