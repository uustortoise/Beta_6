import Link from 'next/link';
import { Sidebar } from '../components/Sidebar';
import { getResidents } from '../lib/data';
import { User, AlertCircle, Activity } from 'lucide-react';
import NewResidentButton from './NewResidentButton';

export const dynamic = 'force-dynamic'; // Force re-render on data change

export default async function ResidentsPage() {
    const residents = await getResidents();

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    <div className="mb-8 flex justify-between items-start">
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">Residents</h1>
                            <p className="mt-2 text-gray-500 dark:text-gray-400">Manage and monitor elder profiles.</p>
                        </div>
                        <NewResidentButton />
                    </div>

                    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
                        {residents.map((resident) => (
                            <Link href={`/residents/${resident.id}`} key={resident.id}>
                                <div className="group relative overflow-hidden rounded-lg border border-gray-200 bg-white p-6 shadow-sm transition-all hover:shadow-md dark:border-gray-700 dark:bg-gray-800">
                                    <div className="flex items-center space-x-4">
                                        <div className="h-12 w-12 rounded-full bg-gray-100 flex items-center justify-center text-gray-400 dark:bg-gray-700 dark:text-gray-300">
                                            <User className="h-6 w-6" />
                                        </div>
                                        <div>
                                            <h3 className="text-lg font-medium text-gray-900 group-hover:text-blue-600 dark:text-white dark:group-hover:text-blue-400">
                                                {resident.name}
                                            </h3>
                                            <p className="text-sm text-gray-500 dark:text-gray-400">ID: {resident.id} • Age: {resident.age}</p>
                                        </div>
                                    </div>

                                    <div className="mt-4 flex items-center justify-between border-t border-gray-100 pt-4 dark:border-gray-700">
                                        <div className="flex items-center space-x-2">
                                            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium 
                                        ${resident.risk_level === 'high' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300' :
                                                    resident.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300' :
                                                        'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'}`}>
                                                {resident.risk_level.toUpperCase()}
                                            </span>
                                        </div>
                                        <span className="text-xs text-gray-400">Updated: {new Date(resident.last_updated).toLocaleDateString()}</span>
                                    </div>
                                </div>
                            </Link>
                        ))}

                        {residents.length === 0 && (
                            <div className="col-span-full rounded-lg border border-dashed border-gray-300 p-12 text-center dark:border-gray-700">
                                <AlertCircle className="mx-auto h-12 w-12 text-gray-400" />
                                <h3 className="mt-2 text-sm font-semibold text-gray-900 dark:text-white">No residents found</h3>
                                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">Make sure data processing has run.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
}
