
import { toast } from "sonner";
import { z } from "zod";

interface ApiResponse<T> {
    success: boolean;
    data?: T;
    message?: string;
    error?: string;
    correlationId?: string;
}

export class ApiError extends Error {
    constructor(public message: string, public status?: number, public correlationId?: string) {
        super(message);
        this.name = 'ApiError';
    }
}

async function handleResponse<T>(response: Response): Promise<T> {
    const correlationId = response.headers.get('X-Correlation-ID') || undefined;

    if (!response.ok) {
        let errorMessage = `Request failed with status ${response.status}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorData.error || errorMessage;
        } catch (e) {
            // Failed to parse error JSON, use default status text
            errorMessage = response.statusText || errorMessage;
        }

        // Log for debugging
        console.error(`[API Error] ${errorMessage} (ID: ${correlationId})`);

        // Toast notification
        toast.error(errorMessage, {
            description: correlationId ? `Reference ID: ${correlationId}` : undefined,
        });

        throw new ApiError(errorMessage, response.status, correlationId);
    }

    // Parse success response
    try {
        // Check for empty response (e.g. 204)
        if (response.status === 204) {
            return {} as T;
        }

        const json = await response.json();

        // If backend standardization wraps data in { success: true, data: ... }
        if (json && typeof json === 'object' && 'success' in json) {
            if (!json.success) {
                const msg = json.message || 'Operation failed';
                toast.error(msg);
                throw new ApiError(msg, response.status, correlationId);
            }
            return json.data as T;
        }

        return json as T;
    } catch (e) {
        if (e instanceof ApiError) throw e;
        const msg = "Failed to parse response from server";
        toast.error(msg);
        throw new ApiError(msg, response.status, correlationId);
    }
}

export const api = {
    get: async <T>(url: string, options?: RequestInit): Promise<T> => {
        try {
            const res = await fetch(url, { ...options, method: 'GET' });
            return handleResponse<T>(res);
        } catch (error) {
            if (error instanceof ApiError) throw error;
            const msg = error instanceof Error ? error.message : "Network error";
            toast.error("Network Error", { description: msg });
            throw new ApiError(msg);
        }
    },

    post: async <T>(url: string, body: any, options?: RequestInit): Promise<T> => {
        try {
            // Handle FormData vs JSON
            const isFormData = body instanceof FormData;
            const headers = new Headers(options?.headers);
            if (!isFormData && !headers.has('Content-Type')) {
                headers.set('Content-Type', 'application/json');
            }

            const res = await fetch(url, {
                ...options,
                method: 'POST',
                headers,
                body: isFormData ? body : JSON.stringify(body)
            });
            return handleResponse<T>(res);
        } catch (error) {
            if (error instanceof ApiError) throw error;
            const msg = error instanceof Error ? error.message : "Network error";
            toast.error("Network Error", { description: msg });
            throw new ApiError(msg);
        }
    },

    // Add put/delete as needed
};
