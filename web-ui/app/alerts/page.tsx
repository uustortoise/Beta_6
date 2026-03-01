import { Sidebar } from '../components/Sidebar';
import { Bell, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { getAllAlerts } from '../lib/data';
import Link from 'next/link';

export const dynamic = 'force-dynamic';

export default async function AlertsPage() {
    const alerts = await getAllAlerts();

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-3">
                            <Bell className="h-8 w-8 text-red-600" />
                            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Alerts</h1>
                        </div>
                        <span className="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs font-semibold">
                            {alerts.length} Total
                        </span>
                    </div>

                    <div className="space-y-4">
                        {alerts.map((alert) => (
                            <div key={alert.id} className={`p-6 rounded-lg border shadow-sm transition-all hover:shadow-md
                                ${alert.severity === 'high' ? 'bg-red-50 border-red-100 dark:bg-red-900/10 dark:border-red-900' :
                                    alert.severity === 'medium' ? 'bg-yellow-50 border-yellow-100 dark:bg-yellow-900/10 dark:border-yellow-900' :
                                        'bg-white border-gray-200 dark:bg-gray-800 dark:border-gray-700'}`}>
                                <div className="flex items-start gap-4">
                                    <div className={`mt-1 p-2 rounded-full shrink-0
                                        ${alert.severity === 'high' ? 'bg-red-100 text-red-600' :
                                            alert.severity === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                                                'bg-blue-100 text-blue-600'}`}>
                                        {alert.severity === 'high' ? <AlertTriangle className="h-5 w-5" /> :
                                            alert.severity === 'medium' ? <Info className="h-5 w-5" /> :
                                                <CheckCircle className="h-5 w-5" />}
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex items-center justify-between">
                                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">{alert.title}</h3>
                                            <span className="text-xs text-gray-500 whitespace-nowrap">
                                                {new Date(alert.alert_date).toLocaleString()}
                                            </span>
                                        </div>
                                        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                                            {alert.message}
                                        </p>

                                        {alert.recommendations && alert.recommendations.length > 0 && (
                                            <div className="mt-4 bg-white/50 dark:bg-black/20 rounded p-3">
                                                <h4 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">Recommendations</h4>
                                                <ul className="list-disc pl-5 space-y-1">
                                                    {alert.recommendations.map((rec, i) => (
                                                        <li key={i} className="text-sm text-gray-700 dark:text-gray-400">{rec}</li>
                                                    ))}
                                                </ul>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ))}

                        {alerts.length === 0 && (
                            <div className="text-center py-12 bg-white rounded-lg border border-dashed border-gray-300 dark:bg-gray-800 dark:border-gray-700">
                                <h3 className="text-gray-900 font-medium dark:text-white">No active alerts</h3>
                                <p className="text-gray-500 text-sm">Everything is running smoothly.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
