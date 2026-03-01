'use client';

import { useState, useEffect } from 'react';
import { Sidebar } from '../../../components/Sidebar';
import { useRouter, useParams } from 'next/navigation';
import { COMMON_MEDICAL_CONDITIONS } from '../../../lib/constants';

export default function EditResidentPage() {
    const params = useParams();
    const id = Array.isArray(params.id) ? params.id[0] : params.id as string;
    const router = useRouter();

    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);

    // Initial State Structure matching Enhanced Profile
    const [formData, setFormData] = useState({
        personal_info: {
            full_name: '',
            date_of_birth: '',
            age: '',
            gender: 'Unknown',
            contact_info: { phone: '', email: '', address: '' }
        },
        medical_history: {
            chronic_conditions: [] as string[],
            medications: [] as any[],
            allergies: [] as any[]
        },
        health_metrics: {
            height_cm: '',
            weight_kg: '',
            blood_type: ''
        },
        emergency_contacts: [] as any[]
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch(`/api/residents/${id}`);
                if (res.ok) {
                    const data = await res.json();

                    // Merge fetched data with structure, handling legacy/missing fields
                    setFormData(prev => ({
                        ...prev,
                        personal_info: {
                            ...prev.personal_info,
                            ...data.personal_info,
                            full_name: data.personal_info?.full_name || data.name || '',
                            age: data.personal_info?.age || data.age || ''
                        },
                        medical_history: {
                            ...prev.medical_history,
                            ...data.medical_history,
                            chronic_conditions: data.medical_history?.chronic_conditions || (Array.isArray(data.medical_history) ? data.medical_history : [])
                        },
                        health_metrics: { ...prev.health_metrics, ...data.health_metrics },
                        emergency_contacts: data.emergency_contacts || []
                    }));
                }
            } catch (error) {
                console.error("Failed to load resident", error);
            } finally {
                setIsLoading(false);
            }
        };
        fetchData();
    }, [id]);

    const handleChange = (section: string, field: string, value: any, subField?: string) => {
        setFormData(prev => {
            if (section === 'root') return { ...prev, [field]: value };

            // Handle deeper nesting like personal_info.contact_info
            if (subField && section === 'personal_info' && field === 'contact_info') {
                return {
                    ...prev,
                    personal_info: {
                        ...prev.personal_info,
                        contact_info: {
                            ...prev.personal_info.contact_info,
                            [subField]: value
                        }
                    }
                };
            }

            return {
                ...prev,
                [section]: {
                    ...prev[section as keyof typeof prev],
                    [field]: value
                }
            };
        });
    };

    const handleListChange = (section: string, field: string, index: number, value: any, subKey?: string) => {
        // simplified for string[] (conditions) vs object[] (meds) handling would go here
        // For prototype, implementing simple Add/Remove for Chronic Conditions strings
        if (section === 'medical_history' && field === 'chronic_conditions') {
            const newList = [...formData.medical_history.chronic_conditions];
            newList[index] = value;
            setFormData(prev => ({
                ...prev,
                medical_history: { ...prev.medical_history, chronic_conditions: newList }
            }));
        }
    };

    const addListItem = (section: string, field: string, initialValue: any) => {
        if (section === 'medical_history' && field === 'chronic_conditions') {
            setFormData(prev => ({
                ...prev,
                medical_history: {
                    ...prev.medical_history,
                    chronic_conditions: [...prev.medical_history.chronic_conditions, initialValue]
                }
            }));
        }
        // Implement others similarly
    };

    const removeListItem = (section: string, field: string, index: number) => {
        if (section === 'medical_history' && field === 'chronic_conditions') {
            const newList = formData.medical_history.chronic_conditions.filter((_, i) => i !== index);
            setFormData(prev => ({
                ...prev,
                medical_history: { ...prev.medical_history, chronic_conditions: newList }
            }));
        }
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsSaving(true);
        try {
            const res = await fetch(`/api/residents/${id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (res.ok) {
                alert("Profile Updated Successfully");
                router.push(`/residents/${id}`);
            } else {
                alert("Failed to save profile");
            }
        } catch (e) {
            console.error(e);
            alert("Error saving profile");
        } finally {
            setIsSaving(false);
        }
    };

    if (isLoading) return <div className="p-8">Loading...</div>;

    return (
        <div className="flex bg-gray-50 dark:bg-gray-900 min-h-screen">
            <Sidebar />
            <main className="flex-1 p-8">
                <div className="mx-auto max-w-4xl">
                    <div className="mb-6 flex items-center justify-between">
                        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Edit Profile: {formData.personal_info.full_name}</h1>
                        <button
                            type="button"
                            onClick={() => router.push(`/residents/${id}`)}
                            className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400"
                        >
                            Cancel
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-8">

                        {/* Personal Info */}
                        <div className="bg-white p-6 shadow-sm rounded-lg dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                            <h2 className="text-lg font-semibold mb-4 dark:text-white">Personal Information</h2>
                            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Full Name</label>
                                    <input
                                        type="text"
                                        value={formData.personal_info.full_name}
                                        onChange={(e) => handleChange('personal_info', 'full_name', e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Gender</label>
                                    <select
                                        value={formData.personal_info.gender}
                                        onChange={(e) => handleChange('personal_info', 'gender', e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                    >
                                        <option value="Male">Male</option>
                                        <option value="Female">Female</option>
                                        <option value="Unknown">Unknown</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Phone</label>
                                    <input
                                        type="text"
                                        value={formData.personal_info.contact_info.phone}
                                        onChange={(e) => handleChange('personal_info', 'contact_info', e.target.value, 'phone')}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Medical History with Datalist */}
                        <div className="bg-white p-6 shadow-sm rounded-lg dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                            <div className="flex justify-between items-center mb-4">
                                <h2 className="text-lg font-semibold dark:text-white">Medical History</h2>
                            </div>

                            <datalist id="conditions-list">
                                {COMMON_MEDICAL_CONDITIONS.map((c) => (
                                    <option key={c} value={c} />
                                ))}
                            </datalist>

                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Chronic Conditions</label>
                                {formData.medical_history.chronic_conditions.map((condition, index) => (
                                    <div key={index} className="flex gap-2 mb-2">
                                        <input
                                            type="text"
                                            list="conditions-list"
                                            value={condition}
                                            onChange={(e) => handleListChange('medical_history', 'chronic_conditions', index, e.target.value)}
                                            className="flex-1 rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                            placeholder="Select or type..."
                                        />
                                        <button
                                            type="button"
                                            onClick={() => removeListItem('medical_history', 'chronic_conditions', index)}
                                            className="px-2 text-red-600 hover:text-red-800"
                                        >×</button>
                                    </div>
                                ))}
                                <button
                                    type="button"
                                    onClick={() => addListItem('medical_history', 'chronic_conditions', '')}
                                    className="mt-2 text-sm text-indigo-600 hover:text-indigo-500"
                                >
                                    + Add Condition
                                </button>
                            </div>
                        </div>

                        {/* Health Metrics */}
                        <div className="bg-white p-6 shadow-sm rounded-lg dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
                            <h2 className="text-lg font-semibold mb-4 dark:text-white">Health Metrics</h2>
                            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Height (cm)</label>
                                    <input
                                        type="number"
                                        value={formData.health_metrics.height_cm || ''}
                                        onChange={(e) => handleChange('health_metrics', 'height_cm', e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Weight (kg)</label>
                                    <input
                                        type="number"
                                        value={formData.health_metrics.weight_kg || ''}
                                        onChange={(e) => handleChange('health_metrics', 'weight_kg', e.target.value)}
                                        className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white sm:text-sm"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Danger Zone - Delete Profile */}
                        <div className="bg-red-50 border border-red-200 p-6 shadow-sm rounded-lg dark:bg-red-900/20 dark:border-red-800">
                            <h2 className="text-lg font-semibold text-red-900 dark:text-red-200 mb-2">Danger Zone</h2>
                            <p className="text-sm text-red-700 dark:text-red-300 mb-4">
                                Deleting this profile will permanently remove all associated data including medical history, alerts, and activity records. This action cannot be undone.
                            </p>
                            <button
                                type="button"
                                onClick={async () => {
                                    const confirmName = prompt(`To confirm deletion, please type the resident's name: "${formData.personal_info.full_name}"`);
                                    if (confirmName === formData.personal_info.full_name) {
                                        if (confirm('Are you absolutely sure? This action cannot be undone.')) {
                                            try {
                                                const res = await fetch(`/api/residents/${id}`, { method: 'DELETE' });
                                                if (res.ok) {
                                                    alert('Profile deleted successfully');
                                                    router.push('/residents');
                                                } else {
                                                    alert('Failed to delete profile');
                                                }
                                            } catch (e) {
                                                alert('Error deleting profile');
                                            }
                                        }
                                    } else if (confirmName !== null) {
                                        alert('Name did not match. Deletion cancelled.');
                                    }
                                }}
                                className="px-4 py-2 bg-red-600 text-white text-sm font-semibold rounded-md hover:bg-red-700 focus:ring-2 focus:ring-red-500 focus:ring-offset-2"
                            >
                                Delete Profile Permanently
                            </button>
                        </div>

                        <div className="flex justify-end gap-3 pt-6">
                            <button
                                type="button"
                                onClick={() => router.push(`/residents/${id}`)}
                                className="rounded-md bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={isSaving}
                                className="rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 disabled:opacity-50"
                            >
                                {isSaving ? 'Saving...' : 'Save Changes'}
                            </button>
                        </div>
                    </form>
                </div>
            </main>
        </div>
    );
}
