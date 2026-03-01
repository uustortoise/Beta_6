import { Sidebar } from '../../components/Sidebar';
import { getResidentById, getSleepData, getADLHistory, getICOPEStatus, getActivityTimeline, getAvailableTimelineDates } from '../../lib/data';
import { SleepChart } from '../../components/SleepChart';
import ADLChart from '../../components/ADLChart';
import ICOPEChart from '../../components/ICOPEChart';
import TimelineWithDatePicker from '../../components/TimelineWithDatePicker';
import RoutineHealth from '../../components/RoutineHealth';
import ContextTimeline from '../../components/ContextTimeline';
import TrajectoryTimeline from '../../components/TrajectoryTimeline';
import { Home, User, Activity, Moon, Clipboard, AlertTriangle, HeartPulse, Clock, TrendingUp, TrendingDown, Minus, Navigation, Brain } from 'lucide-react';
import Link from 'next/link';

export const revalidate = 60;  // Server cache OK - SWR handles client-side freshness

export default async function ResidentDetailPage({ params }: { params: { id: string } }) {
    const { id } = await params;
    const profile = await getResidentById(id);
    const sleepData = await getSleepData(id);
    const adlData = await getADLHistory(id);
    const icopeData = await getICOPEStatus(id);
    const timelineData = await getActivityTimeline(id);
    const availableDates = await getAvailableTimelineDates(id);

    const icopeChartData = icopeData ? Object.entries(icopeData.domains).map(([key, value]) => ({
        domain: key.charAt(0).toUpperCase() + key.slice(1),
        score: value.score
    })) : [];

    if (!profile) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center dark:bg-gray-900">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Resident Not Found</h1>
                    <Link href="/residents" className="text-blue-600 hover:underline">Back to List</Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    {/* Header */}
                    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                            <div className="flex items-center gap-3">
                                <Link href="/residents" className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">← Residents</Link>
                            </div>
                            <h1 className="mt-2 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">{profile.personal_info.full_name}</h1>
                            <div className="mt-2 flex items-center gap-4 text-sm text-gray-500 dark:text-gray-400">
                                <span>ID: {profile.id}</span>
                                <span>•</span>
                                <span>Age: {profile.personal_info.age}</span>
                                <span>•</span>
                                <span>{profile.personal_info.gender}</span>
                            </div>
                        </div>
                        <div className="flex gap-3">
                            <Link
                                href={`/residents/${profile.id}/edit`}
                                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500"
                            >
                                Edit Profile
                            </Link>
                            <button className="rounded-md bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50 dark:bg-gray-800 dark:text-white dark:ring-gray-700 dark:hover:bg-gray-700">Export Rpt</button>
                        </div>
                    </div>

                    {/* Content Grid: Top Row */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">

                        {/* Left Column: Profile & Info */}
                        <div className="space-y-6 flex flex-col">
                            {/* Risk Status */}
                            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                                <h3 className="text-base font-semibold leading-7 text-gray-900 dark:text-white">Current Status</h3>
                                <div className={`mt-4 rounded-md p-4 flex items-start gap-3 
                            ${profile.risk_level === 'high' ? 'bg-red-50 text-red-900 dark:bg-red-900/20 dark:text-red-200' :
                                        profile.risk_level === 'medium' ? 'bg-yellow-50 text-yellow-900 dark:bg-yellow-900/20 dark:text-yellow-200' :
                                            'bg-green-50 text-green-900 dark:bg-green-900/20 dark:text-green-200'}`}>
                                    <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
                                    <div>
                                        <h4 className="font-semibold capitalize">{profile.risk_level} Risk</h4>
                                        <p className="mt-1 text-sm opacity-90">Auto-assessed based on activity and sleep patterns.</p>
                                    </div>
                                </div>
                            </div>

                            {/* Enhanced Medical Info */}
                            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 flex-1">
                                <h3 className="flex items-center gap-2 text-base font-semibold leading-7 text-gray-900 dark:text-white">
                                    <Clipboard className="h-5 w-5 text-gray-400" /> Medical Profile
                                </h3>
                                <div className="mt-4 space-y-4">
                                    <div>
                                        <h4 className="text-xs font-semibold text-gray-500 uppercase">Chronic Conditions</h4>
                                        {profile.medical_history.chronic_conditions.length > 0 ? (
                                            <ul className="mt-1 list-disc pl-5 space-y-1 text-sm text-gray-600 dark:text-gray-300">
                                                {profile.medical_history.chronic_conditions.map((item, i) => (
                                                    <li key={i}>{item}</li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <p className="text-xs text-gray-500 italic">None listed</p>
                                        )}
                                    </div>

                                    <div>
                                        <h4 className="text-xs font-semibold text-gray-500 uppercase">Medications</h4>
                                        {profile.medical_history.medications.length > 0 ? (
                                            <ul className="mt-1 space-y-1 text-sm text-gray-600 dark:text-gray-300">
                                                {profile.medical_history.medications.map((med, i) => (
                                                    <li key={i} className="flex justify-between">
                                                        <span>{med.name}</span>
                                                        <span className="text-xs text-gray-500">{med.dosage}</span>
                                                    </li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <p className="text-xs text-gray-500 italic">None listed</p>
                                        )}
                                    </div>

                                    <div>
                                        <h4 className="text-xs font-semibold text-gray-500 uppercase">Emergency Contact</h4>
                                        {profile.emergency_contacts.length > 0 ? (
                                            <div className="mt-1 text-sm text-gray-600 dark:text-gray-300">
                                                <p className="font-medium">{profile.emergency_contacts[0].name} ({profile.emergency_contacts[0].relationship})</p>
                                                <p>{profile.emergency_contacts[0].phone}</p>
                                            </div>
                                        ) : (
                                            <p className="text-xs text-gray-500 italic">Not set</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Right Column: ICOPE */}
                        <div className="flex flex-col">
                            <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 h-full">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2">
                                        <HeartPulse className="h-5 w-5 text-purple-500" />
                                        <h3 className="text-base font-semibold text-gray-900 dark:text-white">7-Day Assessment</h3>
                                    </div>
                                    {icopeData && (
                                        <div className="flex items-center gap-2">
                                            {icopeData.trend === 'improving' && <TrendingUp className="h-4 w-4 text-green-500" />}
                                            {icopeData.trend === 'declining' && <TrendingDown className="h-4 w-4 text-red-500" />}
                                            {icopeData.trend === 'stable' && <Minus className="h-4 w-4 text-gray-400" />}

                                            <span className={`text-xs font-semibold px-2 py-1 rounded
                                                    ${icopeData.overall_score >= 80 ? 'bg-green-100 text-green-700' :
                                                    icopeData.overall_score >= 60 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                                                {icopeData.overall_score}/100
                                            </span>
                                        </div>
                                    )}
                                </div>
                                {icopeData ? (
                                    <div className="flex flex-col h-[calc(100%-4rem)]">
                                        <div className="flex-1 min-h-[200px]">
                                            <ICOPEChart data={icopeChartData} />
                                        </div>
                                        <div className="mt-4 space-y-4 overflow-y-auto max-h-64 pr-2">
                                            {icopeData.assessment_summary && (
                                                <div className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-sans">
                                                    {icopeData.assessment_summary.replace(/\*\*/g, '')}
                                                </div>
                                            )}

                                            {icopeData.priority_actions && icopeData.priority_actions.length > 0 && (
                                                <div className="border-t border-gray-100 pt-2">
                                                    <h4 className="text-xs font-semibold text-gray-900 dark:text-white mb-2">Priority Actions</h4>
                                                    <ul className="list-disc pl-4 space-y-1">
                                                        {icopeData.priority_actions.map((action, i) => (
                                                            <li key={i} className="text-xs text-gray-600 dark:text-gray-400">{action}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            )}

                                            <div className="grid grid-cols-2 gap-2 text-xs text-gray-500 pt-2 border-t border-gray-100">
                                                {Object.entries(icopeData.domains).map(([key, val]) => (
                                                    <div key={key} className="flex justify-between pb-1">
                                                        <span className="capitalize">{key}</span>
                                                        <span className="font-semibold">{val.rating}</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="h-full flex items-center justify-center text-gray-400">
                                        No assessment data
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Full Width: Activity Timeline */}
                    <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                        <div className="flex items-center gap-2 mb-4">
                            <Clock className="h-5 w-5 text-teal-500" />
                            <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                                Activity Timeline
                            </h3>
                        </div>
                        <TimelineWithDatePicker
                            elderId={id}
                            initialDates={availableDates}
                            initialData={timelineData}
                        />
                    </div>


                    {/* NEW: Routine Health Section (Beta 5 Intelligence) */}
                    <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                        <div className="flex items-center gap-2 mb-4">
                            <Brain className="h-5 w-5 text-orange-500" />
                            <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                                Routine Health
                            </h3>
                            <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-orange-100 text-orange-700 rounded-full">
                                Beta 5
                            </span>
                        </div>
                        <RoutineHealth elderId={id} />
                    </div>


                    {/* NEW: Household Context (Beta 5 Part 2) */}
                    <div className="mb-6 rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                        <div className="flex items-center gap-2 mb-4">
                            <Home className="h-5 w-5 text-blue-500" />
                            <h3 className="text-base font-semibold text-gray-900 dark:text-white">
                                Household Context (Home State)
                            </h3>
                            <span className="ml-2 px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-700 rounded-full">
                                Beta 5 Part 2
                            </span>
                        </div>
                        <ContextTimeline elderId={id} />
                    </div>


                    {/* Right Column: Key Modules (Sleep, ADL, etc) */}
                    <div className="lg:col-span-3 grid grid-cols-1 lg:grid-cols-2 gap-6">

                        {/* ADL Trends Chart */}
                        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 bg-white h-[500px] flex flex-col">
                            <div className="flex items-center gap-2 mb-4">
                                <Activity className="h-5 w-5 text-blue-500" />
                                <h3 className="text-base font-semibold text-gray-900 dark:text-white">ADL Analysis</h3>
                            </div>
                            {adlData && adlData.daily_records.length > 0 ? (
                                <div className="flex-1 min-h-0">
                                    <ADLChart data={adlData} />
                                </div>
                            ) : (
                                <div className="h-64 flex flex-col items-center justify-center text-gray-500 border-2 border-dashed border-gray-200 rounded-lg">
                                    <p>No ADL trend data available.</p>
                                </div>
                            )}
                        </div>

                        {/* Sleep Chart */}
                        <div className="rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 h-[500px] flex flex-col">
                            <div className="flex items-center justify-between mb-4">
                                <div className="flex items-center gap-2">
                                    <Moon className="h-5 w-5 text-indigo-500" />
                                    <h3 className="text-base font-semibold text-gray-900 dark:text-white">Sleep Analysis</h3>
                                </div>

                            </div>

                            {sleepData && sleepData.sleep_periods.length > 0 ? (
                                <div className="flex flex-col h-full">
                                    <div className="flex-1 min-h-0">
                                        <SleepChart data={sleepData} />
                                    </div>
                                    {sleepData.insights && sleepData.insights.length > 0 ? (
                                        <div className="mt-4 border-t border-gray-100 pt-2">
                                            <h4 className="text-xs font-semibold text-gray-900 dark:text-white mb-2">Insights & Recommendations</h4>
                                            <ul className="list-disc pl-4 space-y-1">
                                                {sleepData.insights.map((insight, i) => (
                                                    <li key={i} className="text-xs text-gray-600 dark:text-gray-400">{insight}</li>
                                                ))}
                                            </ul>
                                        </div>
                                    ) : (
                                        <div className="mt-4 pt-2 text-center text-xs text-gray-500">
                                            No specific insights triggered (Good Sleep Quality)
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <div className="h-full flex flex-col items-center justify-center text-gray-500">
                                    <p>No sleep data available for today.</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
