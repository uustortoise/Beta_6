'use client';

import { useEffect, useState } from 'react';
import { Home, Users, UserMinus, Moon, Zap, HelpCircle } from 'lucide-react';

interface Episode {
  start_time: string;
  end_time: string;
  context_label: string;
  confidence: number;
}

interface ContextTimelineProps {
  elderId: string;
}

const CONTEXT_CONFIG: Record<string, { color: string, icon: any, label: string }> = {
  'Empty': { color: 'bg-gray-400', icon: UserMinus, label: 'Empty Home' },
  'Home_Alone': { color: 'bg-blue-500', icon: Home, label: 'Home Alone' },
  'Guest_Present': { color: 'bg-purple-500', icon: Users, label: 'Guest Present' },
  'High_Activity': { color: 'bg-orange-500', icon: Zap, label: 'High Activity' },
  'Sleep_Quiet': { color: 'bg-indigo-500', icon: Moon, label: 'Sleep/Quiet' },
  'Normal_Activity': { color: 'bg-teal-500', icon: Home, label: 'Normal Activity' },
};

export default function ContextTimeline({ elderId }: ContextTimelineProps) {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [loading, setLoading] = useState(true);
  const [date, setDate] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`/api/context?elderId=${elderId}`);
        if (res.ok) {
          const data = await res.json();
          setEpisodes(data.episodes || []);
          setDate(data.date);
        }
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, [elderId]);

  if (loading) return <div className="h-16 animate-pulse bg-gray-100 rounded-lg" />;
  
  if (episodes.length === 0) {
    return (
        <div className="text-center py-6 text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
            <p>No household context data available.</p>
            <p className="text-xs mt-1">Run Part 2 Analysis to generate context.</p>
        </div>
    );
  }

  // Calculate percentages for width
  const startOfDay = new Date(episodes[0].start_time).setHours(0,0,0,0);
  const endOfDay = new Date(episodes[0].start_time).setHours(23,59,59,999);
  const totalMs = endOfDay - startOfDay;

  const getLeft = (time: string) => {
    const t = new Date(time).getTime();
    return Math.max(0, ((t - startOfDay) / totalMs) * 100);
  };
  
  const getWidth = (start: string, end: string) => {
    const s = new Date(start).getTime();
    const e = new Date(end).getTime();
    return Math.min(100, ((e - s) / totalMs) * 100);
  };

  return (
    <div className="space-y-4">
        {date && (
            <div className="text-xs text-center text-gray-500 mb-2">
                Analysis Date: {date}
            </div>
        )}

      <div className="relative h-12 bg-gray-100 rounded-lg overflow-hidden flex w-full">
        {episodes.map((ep, idx) => {
          const config = CONTEXT_CONFIG[ep.context_label] || { color: 'bg-gray-500', icon: HelpCircle, label: ep.context_label };
          const left = getLeft(ep.start_time);
          const width = getWidth(ep.start_time, ep.end_time);
          
          return (
            <div
              key={idx}
              className={`absolute top-0 bottom-0 ${config.color} border-r border-white/20 hover:opacity-90 transition-opacity group`}
              style={{ left: `${left}%`, width: `${width}%` }}
            >
                {/* Tooltip */}
                <div className="hidden group-hover:block absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-gray-900 text-white text-xs p-2 rounded whitespace-nowrap z-10">
                    <div className="font-semibold">{config.label}</div>
                    <div>{new Date(ep.start_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} - {new Date(ep.end_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</div>
                </div>
            </div>
          );
        })}
        
        {/* Hour markers */}
        {[0, 6, 12, 18].map(h => (
            <div key={h} className="absolute top-0 bottom-0 w-px bg-gray-300 pointer-events-none" style={{ left: `${(h/24)*100}%` }}>
                <span className="absolute top-1 left-1 text-[10px] text-gray-400 font-mono">{h}:00</span>
            </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 justify-center text-xs text-gray-600">
        {Object.entries(CONTEXT_CONFIG).map(([key, conf]) => (
            <div key={key} className="flex items-center gap-1.5">
                <div className={`w-3 h-3 rounded-full ${conf.color}`} />
                <span>{conf.label}</span>
            </div>
        ))}
      </div>
    </div>
  );
}
