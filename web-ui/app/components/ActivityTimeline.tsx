'use client';

import { useState, useMemo, useEffect } from 'react';
import { Filter, ZoomIn, Clock } from 'lucide-react';

interface TimelineSegment {
    timestamp: string;      // start_time
    end_time?: string;      // end_time (for segments)
    room: string;
    activity: string;
    confidence: number;
    duration_minutes?: number;
    event_count?: number;
    is_anomaly?: boolean;
}

interface ActivityTimelineProps {
    data: TimelineSegment[];
}

export default function ActivityTimeline({ data }: ActivityTimelineProps) {
    // 1. Get Unique Activities & Rooms (with null safety)
    const safeData = Array.isArray(data) ? data : [];
    const allActivities = useMemo(() => Array.from(new Set(safeData.map(d => d.activity))).sort(), [safeData]);
    const rooms = useMemo(() => Array.from(new Set(safeData.map(d => d.room))).sort(), [safeData]);

    // 2. State for Filters - Default to showing ALL activities EXCEPT 'inactive'
    const [selectedActivities, setSelectedActivities] = useState<Set<string>>(() => {
        const initial = new Set(allActivities);
        initial.delete('inactive');   // Hide 'inactive' (person present, idle)
        initial.delete('unoccupied'); // Hide 'unoccupied' (room empty)
        return initial;
    });

    // 3. Colors for activities
    const activityColors: Record<string, string> = {
        'sleep': '#6366f1',      // Indigo
        'nap': '#8b5cf6',        // Purple
        'inactive': '#9ca3af',   // Gray
        'room_normal_use': '#22c55e', // Green
        'livingroom_normal_use': '#14b8a6', // Teal
        'bathroom_normal_use': '#0ea5e9', // Sky
        'kitchen normal use': '#f59e0b', // Amber
        'cooking': '#ef4444',    // Red
        'shower': '#06b6d4',     // Cyan
        'low_confidence': '#fbbf24', // Yellow
    };

    const getActivityColor = (activity: string) => {
        if (!activity) return '#9ca3af'; // Default gray for null/undefined
        const key = activity.toLowerCase();
        return activityColors[key] || activityColors[activity] || '#8884d8';
    };

    // 4. Process and filter data
    const filteredSegments = useMemo(() => {
        return safeData
            .filter(d => selectedActivities.has(d.activity))
            .map(d => ({
                ...d,
                startTime: new Date(d.timestamp).getTime(),
                endTime: d.end_time ? new Date(d.end_time).getTime() : new Date(d.timestamp).getTime() + 600000, // Default 10min
                roomIndex: rooms.indexOf(d.room),
            }))
            .sort((a, b) => a.startTime - b.startTime);
    }, [safeData, selectedActivities, rooms]);

    // 5. Calculate time bounds - ALWAYS show full 24 hours
    const timeBounds = useMemo(() => {
        if (filteredSegments.length === 0) return { min: 0, max: 0 };

        // Get the date from the first segment
        const firstSegmentDate = new Date(filteredSegments[0].startTime);

        // Set to start of day (00:00:00)
        const startOfDay = new Date(firstSegmentDate);
        startOfDay.setHours(0, 0, 0, 0);

        // Set to end of day (23:59:59)
        const endOfDay = new Date(firstSegmentDate);
        endOfDay.setHours(23, 59, 59, 999);

        return {
            min: startOfDay.getTime(),
            max: endOfDay.getTime()
        };
    }, [filteredSegments]);

    const totalDuration = timeBounds.max - timeBounds.min;

    // 6. Monitoring (Production Recommendation)
    useEffect(() => {
        if (totalDuration > 0 && filteredSegments.length > 0) {
            const durationHours = totalDuration / 3600000;
            const segmentDensity = filteredSegments.length / durationHours;

            if (segmentDensity > 20) {
                console.warn(`⚠️ High segment density detected: ${segmentDensity.toFixed(1)} segments/hour. Consider adjusting denoising or merging logic.`);
            }
        }
    }, [filteredSegments, totalDuration]);

    // Handlers
    const toggleActivity = (activity: string) => {
        const next = new Set(selectedActivities);
        if (next.has(activity)) {
            next.delete(activity);
        } else {
            next.add(activity);
        }
        setSelectedActivities(next);
    };

    const toggleAll = () => {
        if (selectedActivities.size === allActivities.length) {
            setSelectedActivities(new Set());
        } else {
            setSelectedActivities(new Set(allActivities));
        }
    };

    const formatTime = (timestamp: number) => {
        return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    if (!safeData || safeData.length === 0) {
        return (
            <div className="flex h-full items-center justify-center text-gray-500">
                No activity timeline data available
            </div>
        );
    }

    return (
        <div className="w-full space-y-4">
            {/* Control Panel */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-3 bg-gray-50 rounded-md border border-gray-100 dark:bg-gray-800 dark:border-gray-700">
                <div className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-200 min-w-max">
                    <Filter className="h-4 w-4" />
                    <span>Filters:</span>
                </div>

                <div className="flex flex-wrap gap-2 flex-1">
                    <button
                        onClick={toggleAll}
                        className="px-3 py-1 text-xs font-semibold rounded-full border border-gray-300 bg-white text-gray-600 hover:bg-gray-100 dark:bg-gray-700 dark:text-gray-200 dark:border-gray-500"
                    >
                        {selectedActivities.size === allActivities.length ? 'Clear All' : 'Select All'}
                    </button>
                    {allActivities.map((act) => (
                        <button
                            key={act}
                            onClick={() => toggleActivity(act)}
                            className={`px-3 py-1 text-xs font-semibold rounded-full border transition-all flex items-center gap-1
                                ${selectedActivities.has(act)
                                    ? 'bg-white shadow-sm border-gray-300 text-gray-800 dark:bg-gray-700 dark:text-gray-100 dark:border-gray-500'
                                    : 'bg-gray-100 text-gray-400 border-transparent dark:bg-gray-900 dark:text-gray-600 opacity-60'
                                }`}
                        >
                            <span className="w-2 h-2 rounded-full" style={{ backgroundColor: getActivityColor(act) }}></span>
                            {act}
                        </button>
                    ))}
                </div>

                <div className="hidden sm:flex items-center gap-2 text-xs text-gray-400 min-w-max">
                    <Clock className="h-3 w-3" />
                    <span>{filteredSegments.length} segments</span>
                </div>
            </div>

            {/* Gantt-Style Timeline */}
            <div className="relative bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-700 overflow-visible">
                {/* Time Axis Header */}
                <div className="flex border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-xs text-gray-500">
                    <div className="w-24 shrink-0 px-2 py-2 font-semibold">Room</div>
                    <div className="flex-1 flex justify-between px-2 py-2">
                        <span>{formatTime(timeBounds.min)}</span>
                        <span>{formatTime(timeBounds.min + totalDuration * 0.25)}</span>
                        <span>{formatTime(timeBounds.min + totalDuration * 0.5)}</span>
                        <span>{formatTime(timeBounds.min + totalDuration * 0.75)}</span>
                        <span>{formatTime(timeBounds.max)}</span>
                    </div>
                </div>

                {/* Room Rows */}
                {rooms.map((room, roomIdx) => {
                    const roomSegments = filteredSegments.filter(s => s.room === room);

                    return (
                        <div key={room} className="flex border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                            {/* Room Label */}
                            <div className="w-24 shrink-0 px-2 py-3 text-xs font-medium text-gray-700 dark:text-gray-300 capitalize truncate">
                                {room.replace(/_/g, ' ')}
                            </div>

                            {/* Timeline Track */}
                            <div className="flex-1 relative h-10 bg-gray-50/50 dark:bg-gray-900/50">
                                {/* Grid lines */}
                                <div className="absolute inset-0 flex">
                                    {[0.25, 0.5, 0.75].map(p => (
                                        <div key={p} className="absolute h-full border-l border-gray-200 dark:border-gray-700" style={{ left: `${p * 100}%` }} />
                                    ))}
                                </div>

                                {/* Activity Bars */}
                                {roomSegments.map((seg, idx) => {
                                    // 1. Clamp end time to max time bounds (handle midnight spans)
                                    const constrainedEndTime = Math.min(seg.endTime, timeBounds.max);
                                    const constrainedStartTime = Math.max(seg.startTime, timeBounds.min);

                                    // Skip invalid segments
                                    if (constrainedStartTime >= constrainedEndTime) {
                                        console.warn(`[Timeline] Skipped invalid segment: ${seg.activity} @ ${new Date(seg.startTime).toISOString()}`);
                                        return null;
                                    }

                                    // 2. Calculate position percentages
                                    let left = ((constrainedStartTime - timeBounds.min) / totalDuration) * 100;
                                    let width = ((constrainedEndTime - constrainedStartTime) / totalDuration) * 100;

                                    // 3. Visual bounds checking
                                    left = Math.max(0, Math.min(100, left));
                                    width = Math.max(0.1, Math.min(100 - left, width)); // Min 0.1% width for visibility

                                    return (
                                        <div
                                            key={`${seg.activity}-${idx}`}
                                            className="absolute group"
                                            style={{
                                                left: `${left}%`,
                                                width: `${width}%`,
                                                top: '0.25rem',
                                                bottom: '0.25rem'
                                            }}
                                        >
                                            {/* Activity Bar */}
                                            <div
                                                className="absolute inset-0 rounded-sm shadow-sm group-hover:shadow-md transition-shadow cursor-pointer"
                                                style={{
                                                    backgroundColor: getActivityColor(seg.activity),
                                                    opacity: seg.confidence || 0.8
                                                }}
                                                title={`${seg.activity}: ${formatTime(seg.startTime)} - ${formatTime(seg.endTime)} (${seg.duration_minutes?.toFixed(0) || '?'} min)`}
                                            />

                                            {/* Tooltip - positioned as sibling with very high z-index */}
                                            <div className="absolute hidden group-hover:block left-1/2 -translate-x-1/2 pointer-events-none" style={{ top: 'calc(100% + 8px)', zIndex: 9999 }}>
                                                <div className="bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap shadow-lg">
                                                    <div className="font-semibold">{seg.activity || 'unknown'}</div>
                                                    <div>{formatTime(seg.startTime)} - {formatTime(seg.endTime)}</div>
                                                    <div className="text-gray-300">{seg.duration_minutes?.toFixed(0)} min</div>
                                                    {seg.endTime > timeBounds.max && (
                                                        <div className="text-red-300 text-[10px]">(Continues next day)</div>
                                                    )}
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-3 text-xs text-gray-500 dark:text-gray-400">
                {allActivities.filter(a => selectedActivities.has(a)).map(act => (
                    <div key={act} className="flex items-center gap-1">
                        <span className="w-3 h-3 rounded" style={{ backgroundColor: getActivityColor(act) }}></span>
                        <span className="capitalize">{act.replace(/_/g, ' ')}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
