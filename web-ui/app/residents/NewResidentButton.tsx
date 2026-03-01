
'use client';

import { useRouter } from 'next/navigation';
import { Plus } from 'lucide-react';
import { useState } from 'react';
import { api } from '../lib/api';
import { toast } from 'sonner';

export default function NewResidentButton() {
    const router = useRouter();
    const [isCreating, setIsCreating] = useState(false);

    const handleCreate = async () => {
        setIsCreating(true);
        try {
            // Use standardized API client which handles errors and toasts automatically
            const data = await api.post<{ id: string }>('/api/residents', {});

            toast.success('Resident profile created');
            router.push(`/residents/${data.id}/edit`);
            router.refresh();
        } catch (error) {
            // Error handled by api client (toast displayed)
            console.error("Creation failed", error);
        } finally {
            setIsCreating(false);
        }
    };

    return (
        <button
            onClick={handleCreate}
            disabled={isCreating}
            className="flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 disabled:opacity-70 disabled:cursor-not-allowed transition-all"
        >
            <Plus className="h-4 w-4" />
            {isCreating ? 'Creating...' : 'New Resident'}
        </button>
    );
}
