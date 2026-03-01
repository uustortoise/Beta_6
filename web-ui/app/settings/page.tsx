'use client';

import { useState, useEffect } from 'react';
import { Sidebar } from '../components/Sidebar';
import { Save, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react';

interface SettingsData {
    // Model
    DEFAULT_EPOCHS: number;
    DEFAULT_LSTM_UNITS: number;
    DEFAULT_CONV_FILTERS_1: number;
    DEFAULT_CONV_FILTERS_2: number;
    DEFAULT_DROPOUT_RATE: number;
    DEFAULT_VALIDATION_SPLIT: number;
    DEFAULT_CONFIDENCE_THRESHOLD: number;

    // Data
    DEFAULT_DATA_INTERVAL: string;
    DEFAULT_SEQUENCE_WINDOW: string;
    DEFAULT_DENOISING_THRESHOLD: number;
    DEFAULT_DENOISING_WINDOW: number;
    DEFAULT_DENOISING_METHOD: string;

    // Sleep
    SLEEP_STAGE_RATIOS: {
        Light: number;
        Deep: number;
        REM: number;
        Awake: number;
    };

    // ADL
    ADL_CONFIDENCE_THRESHOLD: number;
}

const DEFAULT_SETTINGS: SettingsData = {
    DEFAULT_EPOCHS: 5,
    DEFAULT_LSTM_UNITS: 64,
    DEFAULT_CONV_FILTERS_1: 64,
    DEFAULT_CONV_FILTERS_2: 32,
    DEFAULT_DROPOUT_RATE: 0.3,
    DEFAULT_VALIDATION_SPLIT: 0.2,
    DEFAULT_CONFIDENCE_THRESHOLD: 0.8,
    DEFAULT_DATA_INTERVAL: '10s',
    DEFAULT_SEQUENCE_WINDOW: '10min',
    DEFAULT_DENOISING_THRESHOLD: 4,
    DEFAULT_DENOISING_WINDOW: 3,
    DEFAULT_DENOISING_METHOD: 'hampel',
    SLEEP_STAGE_RATIOS: {
        Light: 0.55,
        Deep: 0.15,
        REM: 0.20,
        Awake: 0.10
    },
    ADL_CONFIDENCE_THRESHOLD: 0.8
};

export default function SettingsPage() {
    const [settings, setSettings] = useState<SettingsData | null>(null);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

    useEffect(() => {
        fetchSettings();
    }, []);

    const fetchSettings = async () => {
        setLoading(true);
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error('Failed to load settings');
            const data = await res.json();
            setSettings({
                ...DEFAULT_SETTINGS,
                ...data,
                SLEEP_STAGE_RATIOS: {
                    ...DEFAULT_SETTINGS.SLEEP_STAGE_RATIOS,
                    ...(data?.SLEEP_STAGE_RATIOS || {})
                }
            });
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to load configuration.' });
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        if (!settings) return;
        setSaving(true);
        setMessage(null);

        // Validate Sleep Ratios sum to 1.0 (approx)
        const ratioSum = Object.values(settings.SLEEP_STAGE_RATIOS || DEFAULT_SETTINGS.SLEEP_STAGE_RATIOS).reduce((a, b) => a + b, 0);
        if (Math.abs(ratioSum - 1.0) > 0.01) {
            setMessage({ type: 'error', text: `Sleep Stage Ratios must sum to 1.0 (Current: ${ratioSum.toFixed(2)})` });
            setSaving(false);
            return;
        }

        try {
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(settings)
            });

            if (!res.ok) throw new Error('Failed to save');

            setMessage({ type: 'success', text: 'Configuration saved successfully. Changes apply to next run.' });

            // Allow user to see success message
            setTimeout(() => setMessage(null), 3000);
        } catch (err) {
            setMessage({ type: 'error', text: 'Failed to save configuration.' });
        } finally {
            setSaving(false);
        }
    };

    const handleChange = (key: keyof SettingsData, value: any) => {
        if (!settings) return;
        setSettings({ ...settings, [key]: value });
    };

    const handleNestedChange = (parent: keyof SettingsData, key: string, value: any) => {
        if (!settings) return;
        // @ts-ignore
        setSettings({
            ...settings,
            [parent]: {
                // @ts-ignore
                ...settings[parent],
                [key]: Number(value)
            }
        });
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center dark:bg-gray-900">
                <RefreshCw className="h-8 w-8 animate-spin text-blue-600" />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />

            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14 max-w-4xl mx-auto">
                    <div className="flex items-center justify-between mb-8">
                        <div>
                            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">System Configuration</h1>
                            <p className="mt-2 text-gray-500 dark:text-gray-400">Manage Machine Learning & Analysis parameters.</p>
                        </div>
                        <button
                            onClick={handleSave}
                            disabled={saving}
                            className={`flex items-center px-4 py-2 rounded-lg text-white font-medium transition-colors
                                ${saving ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700'}`}
                        >
                            {saving ? <RefreshCw className="h-4 w-4 mr-2 animate-spin" /> : <Save className="h-4 w-4 mr-2" />}
                            {saving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>

                    {message && (
                        <div className={`mb-6 p-4 rounded-lg flex items-center ${message.type === 'success' ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}`}>
                            {message.type === 'success' ? <CheckCircle className="h-5 w-5 mr-2" /> : <AlertCircle className="h-5 w-5 mr-2" />}
                            {message.text}
                        </div>
                    )}

                    {settings && (
                        <div className="space-y-6">

                            {/* ML Hyperparameters */}
                            <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 dark:bg-gray-800 dark:border-gray-700">
                                <h2 className="text-xl font-semibold mb-6 text-gray-800 dark:text-white flex items-center">
                                    <span className="w-2 h-8 bg-purple-500 rounded-full mr-3"></span>
                                    Model Hyperparameters
                                </h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput label="Epochs" type="number"
                                        value={settings.DEFAULT_EPOCHS}
                                        onChange={(v) => handleChange('DEFAULT_EPOCHS', Number(v))}
                                        desc="Number of training iterations." />
                                    <FormInput label="LSTM Units" type="number"
                                        value={settings.DEFAULT_LSTM_UNITS}
                                        onChange={(v) => handleChange('DEFAULT_LSTM_UNITS', Number(v))}
                                        desc="Size of the LSTM layer." />
                                    <FormInput label="Dropout Rate" type="number" step="0.1"
                                        value={settings.DEFAULT_DROPOUT_RATE}
                                        onChange={(v) => handleChange('DEFAULT_DROPOUT_RATE', Number(v))}
                                        desc="Regularization rate (0.0 - 1.0)." />
                                    <FormInput label="Validation Split" type="number" step="0.1"
                                        value={settings.DEFAULT_VALIDATION_SPLIT}
                                        onChange={(v) => handleChange('DEFAULT_VALIDATION_SPLIT', Number(v))}
                                        desc="Fraction of data used for validation." />
                                    <FormInput label="Conv Filters (Layer 1)" type="number"
                                        value={settings.DEFAULT_CONV_FILTERS_1}
                                        onChange={(v) => handleChange('DEFAULT_CONV_FILTERS_1', Number(v))} />
                                    <FormInput label="Conv Filters (Layer 2)" type="number"
                                        value={settings.DEFAULT_CONV_FILTERS_2}
                                        onChange={(v) => handleChange('DEFAULT_CONV_FILTERS_2', Number(v))} />
                                </div>
                            </section>

                            {/* Thresholds & Data */}
                            <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 dark:bg-gray-800 dark:border-gray-700">
                                <h2 className="text-xl font-semibold mb-6 text-gray-800 dark:text-white flex items-center">
                                    <span className="w-2 h-8 bg-blue-500 rounded-full mr-3"></span>
                                    Processing & Thresholds
                                </h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <FormInput label="ADL Confidence Threshold" type="number" step="0.05"
                                        value={settings.ADL_CONFIDENCE_THRESHOLD}
                                        onChange={(v) => handleChange('ADL_CONFIDENCE_THRESHOLD', Number(v))}
                                        desc="Min confidence to accept prediction." />
                                    <FormInput label="Denoising Threshold" type="number" step="0.5"
                                        value={settings.DEFAULT_DENOISING_THRESHOLD}
                                        onChange={(v) => handleChange('DEFAULT_DENOISING_THRESHOLD', Number(v))}
                                        desc="Sigma threshold for Hampel filter." />
                                    <FormInput label="Data Interval" type="text"
                                        value={settings.DEFAULT_DATA_INTERVAL}
                                        onChange={(v) => handleChange('DEFAULT_DATA_INTERVAL', v)}
                                        desc="Resampling interval string (e.g. '10s')." />
                                    <FormInput label="Sequence Window" type="text"
                                        value={settings.DEFAULT_SEQUENCE_WINDOW}
                                        onChange={(v) => handleChange('DEFAULT_SEQUENCE_WINDOW', v)}
                                        desc="Window size string (e.g. '1min')." />
                                </div>
                            </section>

                            {/* Sleep Analysis */}
                            <section className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 dark:bg-gray-800 dark:border-gray-700">
                                <h2 className="text-xl font-semibold mb-6 text-gray-800 dark:text-white flex items-center">
                                    <span className="w-2 h-8 bg-indigo-500 rounded-full mr-3"></span>
                                    Sleep Analysis Calibration
                                </h2>
                                <p className="text-sm text-gray-500 mb-4 bg-yellow-50 p-3 rounded-md border border-yellow-100 dark:bg-gray-900 dark:border-gray-600">
                                    Adjust the weights for deterministic sleep stage estimation. Must sum to 1.0.
                                </p>
                                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                                    {Object.entries(settings.SLEEP_STAGE_RATIOS).map(([key, val]) => (
                                        <div key={key}>
                                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                                {key} Ratio
                                            </label>
                                            <input
                                                type="number"
                                                step="0.05"
                                                min="0"
                                                max="1"
                                                value={val}
                                                onChange={(e) => handleNestedChange('SLEEP_STAGE_RATIOS', key, e.target.value)}
                                                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white p-2 border"
                                            />
                                        </div>
                                    ))}
                                </div>
                            </section>

                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

function FormInput({ label, type, value, onChange, desc, step }: { label: string, type: string, value: any, onChange: (v: any) => void, desc?: string, step?: string }) {
    return (
        <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                {label}
            </label>
            <input
                type={type}
                step={step}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm dark:bg-gray-700 dark:border-gray-600 dark:text-white p-2 border"
            />
            {desc && <p className="mt-1 text-xs text-gray-500">{desc}</p>}
        </div>
    );
}
