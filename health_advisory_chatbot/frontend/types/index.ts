/**
 * TypeScript types for Health Advisory Chatbot
 */

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  evidenceSummary?: string;
  riskAlerts?: string[];
}

export interface Citation {
  sourceType: string;
  title: string;
  authors: string[];
  journal?: string;
  year?: number;
  doi?: string;
  url?: string;
  pmid?: string;
  evidenceLevel: 'systematic_review' | 'rct' | 'cohort_study' | 'case_control' | 'expert_opinion' | 'clinical_guideline' | 'manufacturer_data';
  confidenceScore: number;
}

export interface ApiCitation {
  source_type: string;
  title: string;
  authors: string[];
  journal?: string;
  year?: number;
  doi?: string;
  url?: string;
  pmid?: string;
  evidence_level: 'systematic_review' | 'rct' | 'cohort_study' | 'case_control' | 'expert_opinion' | 'clinical_guideline' | 'manufacturer_data';
  confidence_score: number;
}

export interface Recommendation {
  id: string;
  title: string;
  category: string;
  priority: number;
  description: string;
  evidenceLevel: string;
  confidence: number;
  actionSteps?: string[];
  citations: {
    title: string;
    authors: string[];
    year?: number;
  }[];
}

export interface RiskInfo {
  overallScore: number;
  overallLevel: 'critical' | 'high' | 'moderate' | 'low' | 'minimal';
  fallRisk?: number;
  cognitiveRisk?: number;
  sleepRisk?: number;
  criticalAlerts: string[];
}

export interface ChatResponse {
  success: boolean;
  sessionId: string;
  message: Message;
  risks?: RiskInfo;
  recommendations?: Recommendation[];
  actionPlan?: {
    actions: {
      id: string;
      title: string;
      description: string;
      priority: number;
      requiresClinician: boolean;
      policyRefs: string[];
    }[];
    contraindications: string[];
    confidence: number;
    policyVersion?: string;
    policyChangelogRefs?: string[];
  };
  newRiskAlerts?: string[];
  responseTimeMs?: number;
  error?: string;
}

export interface ApiMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: ApiCitation[];
  evidence_summary?: string;
  risk_alerts?: string[];
}

export interface ApiChatResponse {
  success: boolean;
  session_id: string;
  message: ApiMessage;
  action_plan?: {
    actions: {
      id: string;
      title: string;
      description: string;
      priority: number;
      requires_clinician: boolean;
      policy_refs: string[];
    }[];
    contraindications: string[];
    confidence: number;
    policy_version?: string;
    policy_changelog_refs?: string[];
  };
  new_risk_alerts?: string[];
  response_time_ms?: number;
  error?: string;
}

export interface SuggestedQuestion {
  question: string;
  category: string;
}

export interface ChatState {
  messages: Message[];
  sessionId: string | null;
  isLoading: boolean;
  error: string | null;
}
