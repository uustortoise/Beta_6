import { Sidebar } from '../components/Sidebar';
import { BarChart2 } from 'lucide-react';

export default function AnalyticsPage() {
    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    <div className="flex items-center gap-3 mb-8">
                        <BarChart2 className="h-8 w-8 text-blue-600" />
                        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Analytics</h1>
                    </div>

                    <div className="rounded-lg border border-gray-200 bg-white p-12 text-center shadow-sm dark:border-gray-700 dark:bg-gray-800">
                        <h3 className="text-lg font-medium text-gray-900 dark:text-white">Coming Soon</h3>
                        <p className="mt-2 text-gray-500">
                            Aggregated population statistics and facility-wide trends will appear here.
                        </p>
                    </div>
                </div>
            </main>
        </div>
    );
}
