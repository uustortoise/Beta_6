import React from 'react';

interface CardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    trend?: 'up' | 'down' | 'neutral';
    trendValue?: string;
    icon?: React.ReactNode;
    className?: string; // Allow custom classes
}

export function StatCard({ title, value, subtitle, trend, trendValue, icon, className }: CardProps) {
    return (
        <div className={`rounded-lg border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800 ${className || ''}`}>
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
                </div>
                {icon && (
                    <div className="rounded-full bg-blue-100 p-3 text-blue-600 dark:bg-blue-900 dark:text-blue-300">
                        {icon}
                    </div>
                )}
            </div>
            {(subtitle || trend) && (
                <div className="mt-4 flex items-center">
                    {trend && (
                        <span className={`flex items-center text-sm font-medium ${trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-gray-500'}`}>
                            {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '•'} {trendValue}
                        </span>
                    )}
                    <span className="ml-2 text-sm text-gray-500 dark:text-gray-400">{subtitle}</span>
                </div>
            )}
        </div>
    );
}
