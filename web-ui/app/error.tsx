
'use client';

import { useEffect } from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';

export default function Error({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    useEffect(() => {
        // Log the error to an error reporting service
        console.error(error);
    }, [error]);

    return (
        <div className="flex min-h-screen w-full flex-col items-center justify-center gap-4 bg-gray-50 p-8 text-center dark:bg-neutral-900">
            <div className="rounded-full bg-red-100 p-4 dark:bg-red-900/30">
                <AlertCircle className="h-8 w-8 text-red-600 dark:text-red-400" />
            </div>
            <div className="space-y-2">
                <h2 className="text-xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
                    Something went wrong!
                </h2>
                <p className="max-w-md text-sm text-gray-500 dark:text-gray-400">
                    {error.message || "An unexpected error occurred."}
                </p>
                {error.digest && (
                    <p className="text-xs text-gray-400 font-mono mt-2">
                        Error ID: {error.digest}
                    </p>
                )}
            </div>
            <button
                onClick={
                    // Attempt to recover by trying to re-render the segment
                    () => reset()
                }
                className="inline-flex items-center gap-2 rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-900 focus:ring-offset-2 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-200"
            >
                <RotateCcw className="h-4 w-4" />
                Try again
            </button>
        </div>
    );
}
