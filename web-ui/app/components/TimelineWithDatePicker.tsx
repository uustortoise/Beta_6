'use client';

import { useState, useEffect, useMemo } from 'react';
import useSWR from 'swr';
import ActivityTimeline from './ActivityTimeline';
import TrajectoryTimeline from './TrajectoryTimeline';
import { Calendar, ChevronDown, RefreshCw } from 'lucide-react';

// Standard fetcher for JSON API endpoints
const fetcher = (url: string) => fetch(url).then(res => {
    if (!res.ok) throw new Error('Failed to fetch');
    return res.json();
});

// SWR configuration for data that needs to stay fresh
const freshDataConfig = {
    revalidateOnFocus: true,        // Refresh when user returns to tab
    revalidateOnReconnect: true,    // Refresh when network reconnects
    refreshInterval: 30000,          // Auto-refresh every 30 seconds
    dedupingInterval: 5000,         // Dedupe requests within 5 seconds
    errorRetryCount: 3,             // Retry failed requests 3 times
};

interface TimelineWithDatePickerProps {
    elderId: string;
    initialDates: string[];
    initialData: any[];
}

export default function TimelineWithDatePicker({
    elderId,
    initialDates,
    initialData
}: TimelineWithDatePickerProps) {
    // Ensure initialDates is always an array
    const safeInitialDates = Array.isArray(initialDates) ? initialDates : [];
    const [selectedDate, setSelectedDate] = useState<string>(safeInitialDates[0] || '');

    // Use SWR for available dates - auto-refreshes to detect new data
    const { data: datesData } = useSWR(
        `/api/timeline?elderId=${elderId}`,
        fetcher,
        {
            fallbackData: { dates: safeInitialDates },
            revalidateOnFocus: true,
            refreshInterval: 60000,  // Check for new dates every minute
        }
    );

    const availableDates = Array.isArray(datesData?.data?.dates) ? datesData.data.dates
        : Array.isArray(datesData?.dates) ? datesData.dates
            : safeInitialDates;

    // Use SWR for timeline data - auto-refreshes for data freshness
    const { data: timelineResponse, isLoading, isValidating, mutate } = useSWR(
        selectedDate ? `/api/timeline?elderId=${elderId}&date=${selectedDate}` : null,
        fetcher,
        {
            ...freshDataConfig,
            fallbackData: selectedDate === initialDates[0] ? { timeline: initialData } : undefined,
        }
    );

    const timelineData = timelineResponse?.data?.timeline || timelineResponse?.timeline || [];

    // Update selected date if a new date becomes available
    useEffect(() => {
        if (availableDates.length > 0 && !availableDates.includes(selectedDate)) {
            setSelectedDate(availableDates[0]);
        }
    }, [availableDates, selectedDate]);

    const formatDateDisplay = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString(undefined, {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    // Show subtle indicator when revalidating in background
    const showRefreshIndicator = isValidating && !isLoading;

    return (
        <div className="space-y-4">
            {/* Date Picker Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Calendar className="h-4 w-4 text-gray-500" />
                    <span className="text-sm text-gray-600 dark:text-gray-400">Select Date:</span>
                    {/* Auto-refresh indicator */}
                    {showRefreshIndicator && (
                        <RefreshCw className="h-3 w-3 text-blue-500 animate-spin" />
                    )}
                </div>

                <div className="flex items-center gap-2">
                    <div className="relative">
                        <select
                            value={selectedDate}
                            onChange={(e) => setSelectedDate(e.target.value)}
                            disabled={isLoading}
                            className="appearance-none bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg px-4 py-2 pr-10 text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer disabled:opacity-50"
                        >
                            {availableDates.map((date: string) => (
                                <option key={date} value={date}>
                                    {formatDateDisplay(date)}
                                </option>
                            ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-500 pointer-events-none" />
                    </div>

                    {/* Manual refresh button */}
                    <button
                        onClick={() => mutate()}
                        disabled={isLoading || isValidating}
                        className="p-2 text-gray-500 hover:text-blue-500 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors disabled:opacity-50"
                        title="Refresh data"
                    >
                        <RefreshCw className={`h-4 w-4 ${isValidating ? 'animate-spin' : ''}`} />
                    </button>
                </div>
            </div>

            {/* Loading State */}
            {isLoading && (
                <div className="flex items-center justify-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    <span className="ml-3 text-sm text-gray-500">Loading timeline...</span>
                </div>
            )}

            {/* Timeline */}
            {!isLoading && <ActivityTimeline data={timelineData} />}

            {/* Movement Trajectories - uses same selected date */}
            {!isLoading && (
                <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
                    <div className="flex items-center gap-2 mb-4">
                        <svg className="h-5 w-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        <h4 className="text-base font-semibold text-gray-900 dark:text-white">
                            Movement Trajectories
                        </h4>
                        <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-purple-100 text-purple-700 rounded-full">
                            Beta 5
                        </span>
                    </div>
                    <TrajectoryTimeline elderId={elderId} date={selectedDate} />
                </div>
            )}
        </div>
    );
}
