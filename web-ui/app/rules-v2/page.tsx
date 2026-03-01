'use client';

import { useState, useEffect } from 'react';
import { Sidebar } from '../components/Sidebar';
import { Settings, Plus, Trash2, Save, X, AlertTriangle } from 'lucide-react';

interface RuleCondition {
    metric: string;
    operator: string;
    value_type: string;
    value: number;
}

interface AlertRule {
    id?: number;
    rule_name: string;
    enabled: boolean;
    required_condition: string | null;
    conditions: {
        logic: string;
        rules: RuleCondition[];
    };
    alert_severity: string;
    alert_message: string;
}

const METRICS = [
    { value: 'sleep_duration', label: 'Sleep Duration (hrs)', category: 'Sleep' },
    { value: 'deep_sleep_pct', label: 'Deep Sleep %', category: 'Sleep' },
    { value: 'sleep_efficiency', label: 'Sleep Efficiency %', category: 'Sleep' },
    { value: 'night_toilet_visits', label: 'Night Toilet Visits', category: 'ADL' },
    { value: 'night_motion_events', label: 'Night Motion Events', category: 'ADL' },
    { value: 'day_activity_count', label: 'Daily Activity Count', category: 'ADL' },
];

const OPERATORS = [
    { value: 'greater_than', label: '>' },
    { value: 'less_than', label: '<' },
    { value: 'equals', label: '=' },
    { value: 'greater_equal', label: '>=' },
    { value: 'less_equal', label: '<=' },
];

const VALUE_TYPES = [
    { value: 'absolute', label: 'Absolute Value' },
    { value: 'avg_7d', label: '7-day Average +/%' },
    { value: 'avg_30d', label: '30-day Average +/%' },
    { value: 'avg_90d', label: '90-day Average +/%' },
];

const CONDITIONS = [
    { value: null, label: 'Any Patient' },
    { value: 'hypertension', label: 'Hypertension' },
    { value: 'diabetes', label: 'Diabetes' },
    { value: 'heart_disease', label: 'Heart Disease' },
    { value: 'dementia', label: 'Dementia' },
];

export default function RulesV2Page() {
    const [rules, setRules] = useState<AlertRule[]>([]);
    const [loading, setLoading] = useState(true);
    const [mounted, setMounted] = useState(false);
    const [showNewRuleForm, setShowNewRuleForm] = useState(false);
    const [newRule, setNewRule] = useState<AlertRule>({
        rule_name: '',
        enabled: true,
        required_condition: null,
        conditions: { logic: 'AND', rules: [{ metric: 'sleep_duration', operator: 'less_than', value_type: 'absolute', value: 5 }] },
        alert_severity: 'medium',
        alert_message: ''
    });

    useEffect(() => {
        setMounted(true);
        fetchRules();
    }, []);

    async function fetchRules() {
        try {
            const res = await fetch('/api/alert-rules-v2');
            const data = await res.json();
            if (Array.isArray(data)) {
                setRules(data);
            } else {
                console.error('API returned non-array:', data);
                setRules([]);
            }
        } catch (error) {
            console.error('Failed to fetch rules:', error);
        } finally {
            setLoading(false);
        }
    }

    async function createRule() {
        if (!newRule.rule_name) return;
        try {
            await fetch('/api/alert-rules-v2', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newRule)
            });
            setShowNewRuleForm(false);
            setNewRule({
                rule_name: '',
                enabled: true,
                required_condition: null,
                conditions: { logic: 'AND', rules: [{ metric: 'sleep_duration', operator: 'less_than', value_type: 'absolute', value: 5 }] },
                alert_severity: 'medium',
                alert_message: ''
            });
            fetchRules();
        } catch (error) {
            console.error('Failed to create rule:', error);
        }
    }

    async function deleteRule(id: number) {
        try {
            await fetch(`/api/alert-rules-v2?id=${id}`, { method: 'DELETE' });
            fetchRules();
        } catch (error) {
            console.error('Failed to delete rule:', error);
        }
    }

    async function toggleRule(rule: AlertRule) {
        if (!rule.id) return;
        try {
            await fetch('/api/alert-rules-v2', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...rule, enabled: !rule.enabled })
            });
            fetchRules();
        } catch (error) {
            console.error('Failed to toggle rule:', error);
        }
    }

    function addCondition() {
        setNewRule({
            ...newRule,
            conditions: {
                ...newRule.conditions,
                rules: [
                    ...newRule.conditions.rules,
                    { metric: 'sleep_duration', operator: 'less_than', value_type: 'absolute', value: 5 }
                ]
            }
        });
    }

    function updateCondition(index: number, field: keyof RuleCondition, value: any) {
        const updated = [...newRule.conditions.rules];
        updated[index] = { ...updated[index], [field]: value };
        setNewRule({ ...newRule, conditions: { ...newRule.conditions, rules: updated } });
    }

    function removeCondition(index: number) {
        if (newRule.conditions.rules.length <= 1) return;
        const updated = newRule.conditions.rules.filter((_, i) => i !== index);
        setNewRule({ ...newRule, conditions: { ...newRule.conditions, rules: updated } });
    }

    if (!mounted) return null;

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
            <Sidebar />
            <main className="p-4 sm:ml-64">
                <div className="p-4 mt-14">
                    {/* Header */}
                    <div className="mb-8 flex items-center justify-between">
                        <div>
                            <div className="flex items-center gap-3">
                                <Settings className="h-8 w-8 text-purple-600" />
                                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Disease-Driven Alert Rules</h1>
                            </div>
                            <p className="mt-2 text-gray-600 dark:text-gray-400">
                                Configure flexible alert rules with disease conditions and activity metrics
                            </p>
                        </div>
                        <button
                            onClick={() => setShowNewRuleForm(true)}
                            className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                        >
                            <Plus className="h-4 w-4" />
                            Create Rule
                        </button>
                    </div>

                    {/* New Rule Form */}
                    {showNewRuleForm && (
                        <div className="mb-6 rounded-lg border-2 border-purple-300 bg-white dark:bg-gray-800 dark:border-purple-700 p-6">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="font-semibold text-gray-900 dark:text-white">Create New Alert Rule</h3>
                                <button onClick={() => setShowNewRuleForm(false)} className="text-gray-400 hover:text-gray-600">
                                    <X className="h-5 w-5" />
                                </button>
                            </div>

                            {/* Rule Name */}
                            <div className="mb-4">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Rule Name</label>
                                <input
                                    type="text"
                                    value={newRule.rule_name}
                                    onChange={(e) => setNewRule({ ...newRule, rule_name: e.target.value })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                    placeholder="e.g., High Risk Sleep Alert"
                                />
                            </div>

                            {/* Patient Condition */}
                            <div className="mb-4">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Patient Condition</label>
                                <select
                                    value={newRule.required_condition || ''}
                                    onChange={(e) => setNewRule({ ...newRule, required_condition: e.target.value || null })}
                                    className="w-full px-3 py-2 border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                >
                                    {CONDITIONS.map(c => (
                                        <option key={c.value || 'any'} value={c.value || ''}>{c.label}</option>
                                    ))}
                                </select>
                            </div>

                            {/* Conditions */}
                            <div className="mb-4">
                                <div className="flex items-center justify-between mb-2">
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">Trigger When</label>
                                    <select
                                        value={newRule.conditions.logic}
                                        onChange={(e) => setNewRule({ ...newRule, conditions: { ...newRule.conditions, logic: e.target.value } })}
                                        className="text-xs px-2 py-1 border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                    >
                                        <option value="AND">ALL conditions</option>
                                        <option value="OR">ANY condition</option>
                                    </select>
                                </div>

                                <div className="space-y-3">
                                    {newRule.conditions.rules.map((rule, idx) => (
                                        <div key={idx} className="flex items-center gap-2 p-3 bg-gray-50 dark:bg-gray-900 rounded">
                                            <select
                                                value={rule.metric}
                                                onChange={(e) => updateCondition(idx, 'metric', e.target.value)}
                                                className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                            >
                                                {METRICS.map(m => (
                                                    <option key={m.value} value={m.value}>{m.label}</option>
                                                ))}
                                            </select>
                                            <select
                                                value={rule.operator}
                                                onChange={(e) => updateCondition(idx, 'operator', e.target.value)}
                                                className="w-16 px-2 py-1 text-sm border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                            >
                                                {OPERATORS.map(o => (
                                                    <option key={o.value} value={o.value}>{o.label}</option>
                                                ))}
                                            </select>
                                            <input
                                                type="number"
                                                value={rule.value}
                                                onChange={(e) => updateCondition(idx, 'value', Number(e.target.value))}
                                                className="w-20 px-2 py-1 text-sm border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                            />
                                            <select
                                                value={rule.value_type}
                                                onChange={(e) => updateCondition(idx, 'value_type', e.target.value)}
                                                className="w-48 px-2 py-1 text-sm border border-gray-300 rounded dark:border-gray-600 dark:bg-gray-700 dark:text-white"
                                            >
                                                {VALUE_TYPES.map(v => (
                                                    <option key={v.value} value={v.value}>{v.label}</option>
                                                ))}
                                            </select>
                                            {newRule.conditions.rules.length > 1 && (
                                                <button
                                                    onClick={() => removeCondition(idx)}
                                                    className="p-1 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </button>
                                            )}
                                        </div>
                                    ))}
                                </div>

                                <button
                                    onClick={addCondition}
                                    className="mt-2 text-sm text-purple-600 hover:text-purple-700 dark:text-purple-400"
                                >
                                    + Add Condition
                                </button>
                            </div>

                            {/* Severity */}
                            <div className="mb-4">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Alert Severity</label>
                                <div className="flex gap-4">
                                    {['low', 'medium', 'high'].map(sev => (
                                        <label key={sev} className="flex items-center gap-2 cursor-pointer">
                                            <input
                                                type="radio"
                                                value={sev}
                                                checked={newRule.alert_severity === sev}
                                                onChange={(e) => setNewRule({ ...newRule, alert_severity: e.target.value })}
                                                className="text-purple-600"
                                            />
                                            <span className="text-sm capitalize text-gray-700 dark:text-gray-300">{sev}</span>
                                        </label>
                                    ))}
                                </div>
                            </div>

                            {/* Create Button */}
                            <div className="flex justify-end">
                                <button
                                    onClick={createRule}
                                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
                                >
                                    <Save className="h-4 w-4" />
                                    Create Rule
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Rules List */}
                    <div className="space-y-4">
                        {rules.map(rule => (
                            <div key={rule.id} className="rounded-lg border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800 p-4">
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <h3 className="font-semibold text-gray-900 dark:text-white">{rule.rule_name}</h3>
                                            <span className={`px-2 py-0.5 text-xs rounded ${rule.alert_severity === 'high' ? 'bg-red-100 text-red-700' :
                                                rule.alert_severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                                    'bg-blue-100 text-blue-700'
                                                }`}>
                                                {rule.alert_severity.toUpperCase()}
                                            </span>
                                            {rule.required_condition && (
                                                <span className="px-2 py-0.5 text-xs bg-purple-100 text-purple-700 rounded">
                                                    {rule.required_condition}
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-sm text-gray-600 dark:text-gray-400">
                                            <strong>{rule.conditions.logic}:</strong>{' '}
                                            {rule.conditions.rules.map((r, i) => (
                                                <span key={i}>
                                                    {METRICS.find(m => m.value === r.metric)?.label} {r.operator.replace('_', ' ')} {r.value} ({r.value_type.replace('_', '-')})
                                                    {i < rule.conditions.rules.length - 1 && ` ${rule.conditions.logic} `}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => toggleRule(rule)}
                                            className={`px-3 py-1 text-xs rounded ${rule.enabled ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                                                }`}
                                        >
                                            {rule.enabled ? 'Enabled' : 'Disabled'}
                                        </button>
                                        <button
                                            onClick={() => deleteRule(rule.id!)}
                                            className="p-2 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                                        >
                                            <Trash2 className="h-4 w-4" />
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
}
