import { Sidebar } from './components/Sidebar';
import { StatCard } from './components/StatCard';
import { Users, AlertTriangle, Activity, HeartPulse } from 'lucide-react';
import { getDashboardStats, getRecentSystemActivities } from './lib/data';

export const dynamic = 'force-dynamic'; // Ensure real-time data

export default async function Home() {
  const stats = await getDashboardStats();
  const recentActivities = await getRecentSystemActivities();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar />

      <main className="p-4 sm:ml-64">
        <div className="p-4 mt-14">
          {/* Header */}
          <div className="mb-8 flex items-end justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Overview</h1>
              <p className="mt-2 text-gray-500 dark:text-gray-400">Welcome to the CareMonitor Pro Dashboard.</p>
            </div>
            <div className="flex items-center space-x-2">
              <span className="flex h-3 w-3 relative">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
              </span>
              <span className="text-sm font-medium text-gray-600 dark:text-gray-300">System Online</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 gap-4 mb-8 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              title="Total Residents"
              value={stats.totalResidents.toString()}
              subtitle="Pilot Program Active"
              icon={<Users className="h-6 w-6" />}
              trend="neutral"
              trendValue="Active"
            />
            <StatCard
              title="Active Alerts"
              value={stats.activeAlerts.toString()}
              subtitle="Requires Attention"
              icon={<AlertTriangle className="h-6 w-6 text-red-600" />}
              className={stats.activeAlerts > 0 ? "border-l-4 border-l-red-500" : ""}
              trend={stats.activeAlerts > 0 ? "up" : "neutral"}
              trendValue={stats.activeAlerts > 0 ? "Critical" : "None"}
            />
            <StatCard
              title="Daily Activities"
              value={`${stats.complianceRate}%`}
              subtitle="Compliance Rate"
              icon={<Activity className="h-6 w-6" />}
              trend="up"
              trendValue="Stable"
            />
            <StatCard
              title="Vitality Score (Avg)"
              value={stats.avgVitality.toString()}
              subtitle="Cohort Average"
              icon={<HeartPulse className="h-6 w-6" />}
              trend={stats.avgVitalityTrend === 'stable' ? 'neutral' : stats.avgVitalityTrend}
              trendValue="Stable"
            />
          </div>

          {/* Main Content Area */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 h-96 overflow-y-auto">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Recent Activity Logs</h3>
              <div className="space-y-4">
                {recentActivities.length > 0 ? (
                  recentActivities.map((activity, i) => (
                    <div key={`${activity.id}-${i}`} className="flex items-center justify-between border-b border-gray-100 pb-2 last:border-0 dark:border-gray-700">
                      <div className="flex items-center space-x-3">
                        <div className={`h-2 w-2 rounded-full ${activity.description.toLowerCase().includes('fall') ? 'bg-red-500' : 'bg-blue-500'}`}></div>
                        <span className="text-sm text-gray-600 dark:text-gray-300">
                          <span className="font-medium text-gray-900 dark:text-white">{activity.id}</span> - {activity.description} ({activity.room})
                        </span>
                      </div>
                      <span className="text-xs text-gray-400 whitespace-nowrap ml-2">
                        {new Date(activity.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No recent activity detected.</p>
                )}
              </div>
            </div>

            {/* Placeholder for future Analytics Chart */}
            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 h-96 flex items-center justify-center border-dashed">
              <p className="text-gray-500 dark:text-gray-400">Cohort Analytics Chart (Coming Soon)</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
