"""
Comprehensive Pydantic schemas for Health Advisory Chatbot.

All models include proper validation, documentation, and type hints.
Designed for production use with external review.
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict


class EvidenceLevel(str, Enum):
    """
    Evidence quality hierarchy based on Oxford CEBM Levels.
    https://www.cebm.ox.ac.uk/resources/levels-of-evidence
    """
    SYSTEMATIC_REVIEW = "systematic_review"  # Level 1
    RCT = "rct"  # Randomized Controlled Trial - Level 2
    COHORT_STUDY = "cohort_study"  # Level 3
    CASE_CONTROL = "case_control"  # Level 4
    EXPERT_OPINION = "expert_opinion"  # Level 5
    CLINICAL_GUIDELINE = "clinical_guideline"  # Professional consensus
    MANUFACTURER_DATA = "manufacturer_data"  # Drug/device info


class SeverityLevel(str, Enum):
    """Risk severity classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    MINIMAL = "minimal"


class MessageRole(str, Enum):
    """Chat message roles."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Citation(BaseModel):
    """
    Reference citation for evidence-based recommendations.
    
    Attributes:
        source_type: Type of source (paper, guideline, etc.)
        title: Publication or document title
        authors: List of authors or organization
        journal: Journal or publisher name
        year: Publication year
        doi: Digital Object Identifier
        url: Direct link to source
        pmid: PubMed ID if applicable
        evidence_level: Quality of evidence per Oxford CEBM
        confidence_score: 0-100 confidence in relevance
    """
    model_config = ConfigDict(frozen=True)
    
    source_type: str = Field(..., description="Type of source document")
    title: str = Field(..., description="Publication title")
    authors: List[str] = Field(default=[], description="Authors or organization")
    journal: Optional[str] = Field(None, description="Journal or publisher")
    year: Optional[int] = Field(None, description="Publication year")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    url: Optional[str] = Field(None, description="Direct source URL")
    pmid: Optional[str] = Field(None, description="PubMed identifier")
    evidence_level: EvidenceLevel = Field(..., description="Quality of evidence")
    confidence_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Relevance confidence (0-100)"
    )
    
    @field_validator('confidence_score')
    @classmethod
    def round_confidence(cls, v: float) -> float:
        """Round confidence to 1 decimal place."""
        return round(v, 1)


class Medication(BaseModel):
    """
    Medication record from medical history.
    
    Mirrors structure from EnhancedProfile for compatibility.
    """
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="Medication name")
    dosage: str = Field(..., description="Dosage amount")
    frequency: str = Field(..., description="Administration frequency")
    purpose: Optional[str] = Field(None, description="Therapeutic indication")
    prescribing_doctor: Optional[str] = Field(None, description="Prescriber")
    last_refill: Optional[str] = Field(None, description="Last refill date")
    notes: Optional[str] = Field(None, description="Additional notes")


class Allergy(BaseModel):
    """Allergy record from medical history."""
    model_config = ConfigDict(extra="allow")
    
    allergen: str = Field(..., description="Allergen substance")
    reaction: str = Field(..., description="Observed reaction")
    severity: str = Field(default="mild", description="Severity: mild/moderate/severe")
    notes: Optional[str] = Field(None, description="Additional notes")


class MedicalProfile(BaseModel):
    """
    Comprehensive medical profile for an elder.
    
    Aggregated from EnhancedProfile and other health data sources.
    """
    model_config = ConfigDict(extra="allow")
    
    elder_id: str = Field(..., description="Unique elder identifier")
    full_name: Optional[str] = Field(None, description="Elder full name")
    age: Optional[int] = Field(None, ge=60, le=120, description="Current age")
    gender: Optional[str] = Field(None, description="Gender")
    
    # Medical conditions
    chronic_conditions: List[str] = Field(
        default=[], 
        description="Diagnosed chronic conditions"
    )
    acute_conditions: List[str] = Field(
        default=[], 
        description="Current acute conditions"
    )
    surgical_history: List[str] = Field(
        default=[], 
        description="Previous surgeries"
    )
    family_history: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Family medical history by condition type"
    )
    
    # Medications and allergies
    medications: List[Medication] = Field(default=[], description="Current medications")
    allergies: List[Allergy] = Field(default=[], description="Known allergies")
    
    # Baseline vitals
    baseline_blood_pressure: Optional[Dict[str, int]] = Field(
        None, 
        description="Typical BP: {'systolic': X, 'diastolic': Y}"
    )
    baseline_heart_rate: Optional[int] = Field(None, description="Typical resting HR")
    baseline_weight_kg: Optional[float] = Field(None, description="Typical weight")
    
    # Functional status
    mobility_status: Optional[str] = Field(
        None, 
        description="independent/assisted/wheelchair/bedridden"
    )
    cognitive_status: Optional[str] = Field(
        None,
        description="normal/mild_impairment/moderate_impairment/severe"
    )
    
    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    data_source: str = Field(default="enhanced_profile")
    
    def has_condition(self, condition: str) -> bool:
        """Check if the elder has a specific chronic condition (case-insensitive)."""
        condition_lower = condition.lower()
        return any(
            condition_lower in c.lower() or c.lower() in condition_lower
            for c in self.chronic_conditions
        )
    
    @field_validator('age')
    @classmethod
    def validate_elder_age(cls, v: Optional[int]) -> Optional[int]:
        """Ensure age is appropriate for elderly care context."""
        if v is not None and v < 60:
            raise ValueError("Age must be >= 60 for elderly care context")
        return v


class ADLSummary(BaseModel):
    """
    Summary of Activities of Daily Living over a time period.
    
    Aggregated from ADL history service.
    """
    model_config = ConfigDict(extra="allow")
    
    # Time range
    period_start: datetime = Field(..., description="Analysis period start")
    period_end: datetime = Field(..., description="Analysis period end")
    days_analyzed: int = Field(..., ge=1, description="Number of days in analysis")
    
    # Activity patterns
    total_activities: int = Field(0, description="Total activity count")
    activity_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by activity type"
    )
    
    # Temporal patterns
    nighttime_activity_count: int = Field(
        0, 
        description="Activities between 10PM-6AM"
    )
    nighttime_bathroom_visits: int = Field(
        0,
        description="Nighttime bathroom/toilet visits"
    )
    
    # Anomalies
    anomaly_count: int = Field(0, description="Detected anomalies")
    anomaly_rate: float = Field(
        0.0, 
        ge=0, 
        le=1, 
        description="Anomaly rate (0-1)"
    )
    
    # Trends (compared to previous period)
    activity_trend: Optional[float] = Field(
        None,
        description="Activity count change % vs previous period"
    )
    nighttime_activity_trend: Optional[float] = Field(
        None,
        description="Nighttime activity change %"
    )
    
    # Room usage
    room_usage_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Time spent per room (percentage)"
    )
    
    # Gait/Mobility indicators (if available)
    average_transition_time: Optional[float] = Field(
        None,
        description="Average room transition time (seconds)"
    )
    gait_speed_indicator: Optional[str] = Field(
        None,
        description="gait assessment: normal/slow/very_slow"
    )


class ICOPEAssessment(BaseModel):
    """
    ICOPE (Integrated Care for Older People) assessment summary.
    
    Covers the 6 intrinsic capacity domains.
    """
    model_config = ConfigDict(extra="allow")
    
    assessment_date: date = Field(..., description="Date of assessment")
    assessed_by: Optional[str] = Field(None, description="Assessor name/role")
    
    # ICOPE Domain Scores (0-100)
    cognitive_capacity: Optional[float] = Field(
        None, ge=0, le=100,
        description="Cognitive function score"
    )
    locomotor_capacity: Optional[float] = Field(
        None, ge=0, le=100,
        description="Mobility/movement score"
    )
    psychological_capacity: Optional[float] = Field(
        None, ge=0, le=100,
        description="Mental health/mood score"
    )
    sensory_capacity: Optional[float] = Field(
        None, ge=0, le=100,
        description="Vision and hearing score"
    )
    vitality_nutrition: Optional[float] = Field(
        None, ge=0, le=100,
        description="Nutrition and vitality score"
    )
    
    # Overall
    overall_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Trends
    cognitive_trend: Optional[float] = Field(None, description="Change from last")
    locomotor_trend: Optional[float] = Field(None, description="Change from last")
    
    # Flags
    domains_at_risk: List[str] = Field(
        default=[],
        description="Domains with score < 60"
    )


class SleepSummary(BaseModel):
    """
    Sleep analysis summary for a specific night or averaged over period.
    """
    model_config = ConfigDict(extra="allow")
    
    # Time
    analysis_date: date = Field(..., description="Date of sleep analysis")
    
    # Duration
    total_duration_hours: float = Field(..., ge=0, le=24)
    time_in_bed_hours: float = Field(..., ge=0, le=24)
    
    # Efficiency
    sleep_efficiency: float = Field(..., ge=0, le=100)
    
    # Sleep stages (minutes)
    light_sleep_minutes: Optional[float] = None
    deep_sleep_minutes: Optional[float] = None
    rem_sleep_minutes: Optional[float] = None
    awake_minutes: Optional[float] = None
    
    # Quality
    quality_score: Optional[float] = Field(None, ge=0, le=100)
    
    # Patterns
    sleep_onset_minutes: Optional[float] = Field(
        None,
        description="Time to fall asleep"
    )
    awakenings_count: Optional[int] = Field(
        None,
        description="Number of nighttime awakenings"
    )
    
    # Insights from analysis
    insights: List[str] = Field(default=[], description="Generated insights")
    sleep_apnea_risk: Optional[str] = Field(
        None,
        description="Risk assessment: low/moderate/high"
    )
    insomnia_indicators: List[str] = Field(default=[], description="Insomnia signs")
    
    # Trends
    efficiency_trend: Optional[float] = Field(None, description="Change % vs avg")
    duration_trend: Optional[float] = Field(None, description="Change % vs avg")


class RiskFactor(BaseModel):
    """
    Individual risk factor contributing to overall risk assessment.
    """
    model_config = ConfigDict(extra="allow")
    
    factor_name: str = Field(..., description="Name of risk factor")
    category: str = Field(..., description="Category: fall/cognitive/sleep/medication/frailty")
    severity: SeverityLevel = Field(..., description="Risk severity")
    
    # Scoring
    risk_score: float = Field(..., ge=0, le=100, description="Individual risk score")
    weight: float = Field(default=1.0, ge=0, le=5, description="Factor weight")
    weighted_score: float = Field(..., ge=0, le=500)
    
    # Evidence
    contributing_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source data contributing to this risk"
    )
    evidence_description: str = Field(..., description="Human-readable explanation")
    
    # Prediction
    trend_direction: Optional[str] = Field(
        None,
        description="improving/stable/worsening"
    )
    projected_timeframe: Optional[str] = Field(
        None,
        description="e.g., '30 days' for prediction horizon"
    )


class RiskAssessment(BaseModel):
    """
    Comprehensive multi-domain risk assessment.
    """
    model_config = ConfigDict(extra="allow")
    
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    assessment_period_days: int = Field(default=7, description="Data period analyzed")
    
    # Domain-specific risks
    fall_risk: Optional[float] = Field(None, ge=0, le=100)
    cognitive_decline_risk: Optional[float] = Field(None, ge=0, le=100)
    sleep_disorder_risk: Optional[float] = Field(None, ge=0, le=100)
    medication_risk: Optional[float] = Field(None, ge=0, le=100)
    frailty_risk: Optional[float] = Field(None, ge=0, le=100)
    
    # Overall
    overall_risk_score: float = Field(..., ge=0, le=100)
    overall_risk_level: SeverityLevel = Field(...)
    
    # Breakdown
    risk_factors: List[RiskFactor] = Field(default=[], description="All factors")
    top_risk_factors: List[RiskFactor] = Field(
        default=[],
        description="Top 5 weighted factors"
    )
    
    # Alerts
    critical_alerts: List[str] = Field(default=[], description="Immediate concerns")
    recommendations: List[str] = Field(default=[], description="High-level actions")


class TrajectoryPrediction(BaseModel):
    """
    Predictive trajectory for a specific health domain.
    """
    model_config = ConfigDict(extra="allow")
    
    domain: str = Field(..., description="Health domain being predicted")
    current_status: str = Field(..., description="Current classification")
    
    # Prediction
    prediction_horizon: str = Field(..., description="e.g., '30 days', '6 months'")
    predicted_status: str = Field(..., description="Predicted classification")
    confidence: float = Field(..., ge=0, le=100, description="Prediction confidence")
    
    # Probability distribution
    probability_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Probability per outcome"
    )
    
    # Contributing factors
    key_drivers: List[str] = Field(default=[], description="Main predictive factors")
    protective_factors: List[str] = Field(default=[], description="Protective elements")
    
    # Model info
    model_version: str = Field(..., description="Prediction model version")
    features_used: List[str] = Field(default=[], description="Input features")


class HealthContext(BaseModel):
    """
    Complete health context synthesized from all data sources.
    
    This is the unified view used by the advisory engine.
    """
    model_config = ConfigDict(extra="allow")
    
    elder_id: str = Field(...)
    context_timestamp: datetime = Field(default_factory=datetime.now)
    
    # Source data
    medical_profile: Optional[MedicalProfile] = None
    adl_summary: Optional[ADLSummary] = None
    icope_assessment: Optional[ICOPEAssessment] = None
    sleep_summary: Optional[SleepSummary] = None
    
    # Derived analysis
    risk_assessment: Optional[RiskAssessment] = None
    trajectories: List[TrajectoryPrediction] = Field(default=[])
    
    # Metadata
    data_completeness: Dict[str, bool] = Field(
        default_factory=dict,
        description="Which sources are available"
    )
    context_summary: Optional[str] = Field(
        None,
        description="Human-readable context summary"
    )


class AdvisoryRecommendation(BaseModel):
    """
    Individual recommendation with full evidence chain.
    """
    model_config = ConfigDict(extra="allow")
    
    recommendation_id: str = Field(..., description="Unique recommendation ID")
    category: str = Field(..., description="Category: fall/sleep/cognitive/etc")
    priority: int = Field(..., ge=1, le=10, description="Priority 1-10")
    
    # Content
    title: str = Field(..., description="Brief recommendation title")
    description: str = Field(..., description="Detailed explanation")
    action_items: List[str] = Field(default=[], description="Specific actions")
    
    # Evidence
    evidence_level: EvidenceLevel = Field(...)
    confidence_score: float = Field(..., ge=0, le=100)
    citations: List[Citation] = Field(default=[], description="Supporting sources")
    
    # Clinical context
    contraindications: List[str] = Field(
        default=[],
        description="Situations where NOT applicable"
    )
    related_medications: List[str] = Field(
        default=[],
        description="Relevant medications"
    )
    
    # Predictive context
    expected_outcome: Optional[str] = Field(
        None,
        description="Expected benefit"
    )
    timeline: Optional[str] = Field(
        None,
        description="When to expect results"
    )


class ActionPlanAction(BaseModel):
    """
    Deterministic policy action produced by the clinical policy engine.
    """
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique action identifier")
    title: str = Field(..., description="Action title")
    description: str = Field(..., description="Action details")
    priority: int = Field(..., ge=1, le=10, description="Priority 1-10")
    requires_clinician: bool = Field(
        default=False,
        description="Whether clinician involvement is required"
    )
    policy_refs: List[str] = Field(
        default=[],
        description="Policy references supporting this action"
    )


class ActionPlan(BaseModel):
    """
    Deterministic action plan generated from policy rules.
    """
    model_config = ConfigDict(extra="allow")

    actions: List[ActionPlanAction] = Field(default=[])
    contraindications: List[str] = Field(default=[])
    confidence: float = Field(..., ge=0, le=100)
    policy_version: str = Field(default="unknown")
    policy_changelog_refs: List[str] = Field(default=[])


class Message(BaseModel):
    """Individual chat message."""
    model_config = ConfigDict(extra="allow")
    
    role: MessageRole = Field(...)
    content: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # For assistant messages
    recommendations: List[AdvisoryRecommendation] = Field(default=[])
    risk_alerts: List[str] = Field(default=[])
    
    # Citations for this message
    citations: List[Citation] = Field(default=[])
    evidence_summary: Optional[str] = Field(
        None,
        description="Summary of evidence quality"
    )


class ConversationContext(BaseModel):
    """Context maintained throughout a conversation session."""
    model_config = ConfigDict(extra="allow")
    
    session_id: str = Field(...)
    elder_id: str = Field(...)
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    
    # Health snapshot at start
    initial_health_context: Optional[HealthContext] = None
    
    # Conversation state
    messages: List[Message] = Field(default=[])
    topics_discussed: List[str] = Field(default=[])
    pending_recommendations: List[str] = Field(default=[])
    
    # User preferences
    language: str = Field(default="en")
    detail_level: str = Field(default="standard")  # brief/standard/detailed


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    model_config = ConfigDict(extra="allow")
    
    elder_id: str = Field(..., description="Target elder ID")
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    
    # Context flags
    include_medical_history: bool = Field(
        default=True,
        description="Include medical profile in context"
    )
    include_adl_data: bool = Field(default=True)
    include_icope_data: bool = Field(default=True)
    include_sleep_data: bool = Field(default=True)
    
    # Response preferences
    language: str = Field(default="en")
    detail_level: str = Field(default="standard")
    require_citations: bool = Field(default=True)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    model_config = ConfigDict(extra="allow")
    
    session_id: str = Field(...)
    message: Message = Field(...)
    
    # Health context snapshot
    health_context: Optional[HealthContext] = None
    
    # Risk summary
    current_risks: Optional[RiskAssessment] = None
    new_risk_alerts: List[str] = Field(default=[])
    
    # Recommendations
    recommendations: List[AdvisoryRecommendation] = Field(default=[])
    action_plan: Optional[ActionPlan] = None
    
    # Metadata
    response_time_ms: int = Field(..., description="Processing time")
    model_version: str = Field(..., description="Advisory engine version")
