'use client';

import useSWR from 'swr';

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

// Hook for Activity Timeline data
export function useActivityTimeline(elderId: string, date: string | null) {
    const { data, error, isLoading, mutate } = useSWR(
        date ? `/api/timeline?elderId=${elderId}&date=${date}` : null,
        fetcher,
        freshDataConfig
    );

    return {
        timeline: data?.timeline || [],
        isLoading,
        isError: !!error,
        refresh: mutate
    };
}

// Hook for available timeline dates
export function useTimelineDates(elderId: string) {
    const { data, error, isLoading } = useSWR(
        `/api/timeline?elderId=${elderId}`,
        fetcher,
        {
            revalidateOnFocus: true,
            refreshInterval: 60000,  // Check for new dates every minute
        }
    );

    return {
        dates: data?.dates || [],
        isLoading,
        isError: !!error
    };
}

// Hook for Trajectory data
export function useTrajectories(elderId: string, date: string | null) {
    const { data, error, isLoading, mutate } = useSWR(
        date ? `/api/trajectories?elderId=${elderId}&date=${date}` : null,
        fetcher,
        freshDataConfig
    );

    return {
        trajectories: data?.trajectories || [],
        isLoading,
        isError: !!error,
        refresh: mutate
    };
}

// Hook for Context episodes data
export function useContextEpisodes(elderId: string, date: string | null) {
    const { data, error, isLoading, mutate } = useSWR(
        date ? `/api/context?elderId=${elderId}&date=${date}` : null,
        fetcher,
        freshDataConfig
    );

    return {
        episodes: data?.episodes || [],
        date: data?.date,
        isLoading,
        isError: !!error,
        refresh: mutate
    };
}
