'use client';

import { useEffect, useState } from 'react';
import { Navigation, ArrowRight, Clock } from 'lucide-react';

interface RoomVisit {
  room: string;
  start: string;
  end: string;
  activity: string;
  duration_min: number;
}

interface Trajectory {
  id: number;
  start_time: string;
  end_time: string;
  path: string;
  primary_activity: string;
  room_sequence: RoomVisit[];
  duration_minutes: number;
  confidence: number;
}

interface TrajectoryTimelineProps {
  elderId: string;
  date?: string;
}

const ROOM_COLORS: Record<string, string> = {
  bedroom: 'bg-indigo-100 text-indigo-800 border-indigo-300',
  living_room: 'bg-green-100 text-green-800 border-green-300',
  bathroom: 'bg-cyan-100 text-cyan-800 border-cyan-300',
  kitchen: 'bg-orange-100 text-orange-800 border-orange-300',
  entrance: 'bg-gray-100 text-gray-800 border-gray-300',
  hallway: 'bg-yellow-100 text-yellow-800 border-yellow-300',
};

const ACTIVITY_COLORS: Record<string, string> = {
  sleep: 'bg-purple-500',
  nap: 'bg-purple-400',
  shower: 'bg-blue-500',
  cooking: 'bg-orange-500',
  toilet: 'bg-cyan-500',
  out: 'bg-gray-500',
};

export default function TrajectoryTimeline({ elderId, date }: TrajectoryTimelineProps) {
  const [trajectories, setTrajectories] = useState<Trajectory[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataDate, setDataDate] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTrajectories() {
      setLoading(true);
      try {
        const params = new URLSearchParams({ elderId });
        if (date) params.append('date', date);
        
        const res = await fetch(`/api/trajectories?${params}`);
        if (!res.ok) throw new Error('Failed to fetch trajectories');
        
        const data = await res.json();
        setTrajectories(data.trajectories || []);
        setDataDate(data.date);
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    }
    
    fetchTrajectories();
  }, [elderId, date]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-400">
        <div className="animate-spin h-5 w-5 border-2 border-gray-300 border-t-blue-500 rounded-full mr-2" />
        Loading trajectories...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 text-center py-4">
        Error: {error}
      </div>
    );
  }

  if (trajectories.length === 0) {
    return (
      <div className="text-gray-500 text-center py-8 border-2 border-dashed border-gray-200 rounded-lg">
        <Navigation className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No movement trajectories detected</p>
        <p className="text-xs mt-1">Run the Intelligence pipeline to generate trajectory data</p>
      </div>
    );
  }

  const formatTime = (ts: string) => {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  };

  const getRoomColor = (room: string) => {
    return ROOM_COLORS[room.toLowerCase()] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getActivityColor = (activity: string) => {
    return ACTIVITY_COLORS[activity.toLowerCase()] || 'bg-gray-400';
  };

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center gap-4 text-xs text-gray-500 mb-4">
        <span className="font-medium text-gray-700">
          {trajectories.length} movement path{trajectories.length !== 1 ? 's' : ''} detected
        </span>
        <span>•</span>
        <span>
          Total: {Math.round(trajectories.reduce((sum, t) => sum + t.duration_minutes, 0))} min
        </span>
        {dataDate && (
          <>
            <span>•</span>
            <span>{dataDate}</span>
          </>
        )}
      </div>

      {/* Trajectory cards */}
      {trajectories.map((traj) => (
        <div 
          key={traj.id} 
          className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow dark:border-gray-700 dark:bg-gray-800/50"
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-400" />
              <span className="text-sm font-medium text-gray-900 dark:text-white">
                {formatTime(traj.start_time)} - {formatTime(traj.end_time)}
              </span>
              <span className="text-xs text-gray-500">
                ({Math.round(traj.duration_minutes)} min)
              </span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-2 py-0.5 text-xs font-medium rounded-full text-white ${getActivityColor(traj.primary_activity)}`}>
                {traj.primary_activity}
              </span>
              <span className="text-xs text-gray-400">
                {Math.round(traj.confidence * 100)}%
              </span>
            </div>
          </div>

          {/* Path visualization */}
          <div className="flex items-center gap-1 flex-wrap">
            {traj.room_sequence.map((visit, idx) => (
              <div key={idx} className="flex items-center">
                <div className={`px-2 py-1 rounded border text-xs font-medium ${getRoomColor(visit.room)}`}>
                  {visit.room.replace('_', ' ')}
                  <span className="ml-1 opacity-70">({visit.duration_min}m)</span>
                </div>
                {idx < traj.room_sequence.length - 1 && (
                  <ArrowRight className="h-3 w-3 mx-1 text-gray-400" />
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
