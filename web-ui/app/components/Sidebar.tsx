import Link from 'next/link';
import { Home, Users, BarChart2, Bell, Settings, LogOut, Database } from 'lucide-react';

export function Sidebar() {
  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 -translate-x-full border-r border-gray-200 bg-white transition-transform sm:translate-x-0 dark:border-gray-700 dark:bg-gray-800">
      <div className="h-full overflow-y-auto px-3 py-4">
        <Link href="/" className="mb-5 flex items-center pl-2.5">
          <span className="self-center whitespace-nowrap text-xl font-semibold dark:text-white">
            CareMonitor <span className="text-blue-600">Pro</span>
          </span>
        </Link>
        <ul className="space-y-2 font-medium">
          <li>
            <Link href="/" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
              <Home className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
              <span className="ml-3">Dashboard</span>
            </Link>
          </li>
          <li>
            <Link href="/residents" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
              <Users className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
              <span className="ml-3">Residents</span>
            </Link>
          </li>
          <li>
            <Link href="/analytics" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
              <BarChart2 className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
              <span className="ml-3">Analytics</span>
            </Link>
          </li>
          <li>
            <Link href="/alerts" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
              <Bell className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
              <span className="flex-1 ml-3 whitespace-nowrap">Alerts</span>
              <span className="ml-3 inline-flex h-3 w-3 items-center justify-center rounded-full bg-red-500 p-3 text-sm font-medium text-white">3</span>
            </Link>
          </li>
          <li>
            <Link href="/training" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
              <Database className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
              <span className="ml-3">Model Studio</span>
            </Link>
          </li>
        </ul>
        <div className="mt-auto pt-10">
          <ul className="space-y-2 font-medium border-t border-gray-200 dark:border-gray-700 pt-4">
            <li>
              <Link href="/rules-v2" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
                <Settings className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
                <span className="ml-3">Alert Rules</span>
              </Link>
            </li>
            <li>
              <Link href="#" className="group flex items-center rounded-lg p-2 text-gray-900 hover:bg-gray-100 dark:text-white dark:hover:bg-gray-700">
                <LogOut className="h-5 w-5 text-gray-500 transition duration-75 group-hover:text-gray-900 dark:text-gray-400 dark:group-hover:text-white" />
                <span className="ml-3">Sign Out</span>
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </aside>
  );
}
