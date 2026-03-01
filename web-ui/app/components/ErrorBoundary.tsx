
'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';

interface Props {
    children?: ReactNode;
    fallback?: ReactNode;
}

interface State {
    hasError: boolean;
    error: Error | null;
}

class ErrorBoundary extends Component<Props, State> {
    public state: State = {
        hasError: false,
        error: null
    };

    public static getDerivedStateFromError(error: Error): State {
        return { hasError: true, error };
    }

    public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
        console.error("Uncaught error:", error, errorInfo);
    }

    public render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="flex h-[50vh] w-full flex-col items-center justify-center gap-4 rounded-lg border border-red-200 bg-red-50 p-8 text-center text-red-900 dark:border-red-900/50 dark:bg-red-900/20 dark:text-red-200">
                    <div className="rounded-full bg-red-100 p-3 dark:bg-red-900/30">
                        <AlertCircle className="h-6 w-6 text-red-600 dark:text-red-400" />
                    </div>
                    <div className="space-y-2">
                        <h3 className="text-lg font-semibold">Something went wrong</h3>
                        <p className="max-w-xs text-sm text-red-800/80 dark:text-red-300/80">
                            {this.state.error?.message || "An unexpected error occurred while loading this component."}
                        </p>
                    </div>
                    <button
                        onClick={() => this.setState({ hasError: false, error: null })}
                        className="inline-flex items-center gap-2 rounded-md bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 dark:hover:bg-red-500"
                    >
                        <RotateCcw className="h-4 w-4" />
                        Try again
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
