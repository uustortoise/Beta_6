# Health Advisory Chatbot - Architecture Documentation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ChatbotWidget (Floating Button)                                    │   │
│  │  └── ChatWindow (Expandable Panel)                                  │   │
│  │      ├── MessageBubble (User/Assistant)                            │   │
│  │      ├── SuggestedQuestions (Quick Actions)                        │   │
│  │      ├── EvidenceBadge (Quality Indicator)                         │   │
│  │      └── CitationPanel (Source Viewer)                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (Next.js)                                │
│  POST /api/chat              │ GET /api/chat/history/{id}                   │
│  GET /api/chat/suggestions   │ GET /api/health                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ADVISORY ENGINE (Python)                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ORCHESTRATION (AdvisoryEngine)                    │   │
│  │  1. Receive request → 2. Build context → 3. Calculate risks         │   │
│  │  4. Retrieve evidence → 5. Generate LLM response → 6. Validate      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   Context Fusion    │  │   Predictive Risk   │  │   LLM Service       │ │
│  │   Engine            │  │   Engine            │  │   (RAG-Enhanced)    │ │
│  │                     │  │                     │  │                     │ │
│  │  • ADL History      │  │  • Fall Risk Calc   │  │  • Prompt Engineer  │ │
│  │  • Medical Profile  │  │  • Cognitive Traj   │  │  • Streaming Gen    │ │
│  │  • ICOPE Scores     │  │  • Frailty Index    │  │  • Response Parse   │ │
│  │  • Sleep Analysis   │  │  • Sleep Prediction │  │  • Claim Extract    │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Citation Validator                                │   │
│  │  • Verify claims against knowledge base                             │   │
│  │  • Enrich citations with metadata                                   │   │
│  │  • Calculate evidence quality score                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE LAYER                                      │
│                                                                              │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ │
│  │  Clinical Guidelines │ │  Drug Interactions   │ │  Research Corpus     │ │
│  │                      │ │                      │ │                      │ │
│  │  • AGS Beers 2023    │ │  • Drug-drug checks  │ │  • PubMed abstracts  │ │
│  │  • AGS Falls 2023    │ │  • Contraindications │ │  • Key papers        │ │
│  │  • WHO ICOPE 2017    │ │  • ACB calculator    │ │  • Embeddings index  │ │
│  │  • AASM Sleep 2023   │ │  • Medication risk   │ │  • Citation tracking │ │
│  │  • ADA Diabetes 2023 │ │                      │ │                      │ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────────┘ │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ICOPE Standards (WHO Implementation)                                │  │
│  │  • 6 domain assessments • Care pathways • Intervention protocols     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ (Read-only integration)
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BETA 5.5 DATA SOURCES                                   │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Enhanced    │  │  ADL Service │  │  ICOPE       │  │  Sleep       │    │
│  │  Profile     │  │              │  │  Service     │  │  Service     │    │
│  │              │  │              │  │              │  │              │    │
│  │  • Conditions│  │  • Activities│  │  • 6 domains │  │  • Stages    │    │
│  │  • Medications│ │  • Anomalies │  │  • Scores    │  │  • Efficiency│    │
│  │  • Allergies │  │  • Patterns  │  │  • Trends    │  │  • Insights  │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                              │
│  NOTE: Chatbot module is READ-ONLY. No modifications to Beta 5.5 data.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CONTEXT BUILDING                                         │
│     - Fetch medical profile (EnhancedProfile)               │
│     - Get ADL summary (last 7-30 days)                      │
│     - Retrieve ICOPE assessment                             │
│     - Get sleep analysis                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  2. RISK ANALYSIS                                            │
│     - Calculate fall risk (Tinetti-inspired)                │
│     - Assess cognitive decline trajectory                   │
│     - Evaluate medication risks                             │
│     - Compute frailty index (Fried Criteria)                │
│     - Generate 30-day predictions                           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  3. EVIDENCE RETRIEVAL (RAG)                                 │
│     - Extract topics from query + context                   │
│     - Search clinical guidelines                            │
│     - Query research corpus                                 │
│     - Rank by relevance and evidence level                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  4. LLM GENERATION                                           │
│     - Build structured prompt with context                  │
│     - Inject retrieved evidence                             │
│     - Generate response with [Source: ID] citations         │
│     - Stream response (if supported)                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  5. VALIDATION & ENRICHMENT                                  │
│     - Verify citations against knowledge base               │
│     - Validate medical claims                               │
│     - Calculate evidence quality score                      │
│     - Enrich citations with URLs, full text                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  6. RESPONSE ASSEMBLY                                        │
│     - Format message with citations                         │
│     - Add risk alerts (if critical)                         │
│     - Include evidence summary                              │
│     - Add medical disclaimer                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
User Response
```

## Component Interactions

### AdvisoryEngine Orchestration

```python
# Simplified workflow
class AdvisoryEngine:
    def process_chat_request(request):
        # 1. Build context
        context = context_fusion.build_health_context(...)
        
        # 2. Calculate risks (parallel)
        risk_assessment = risk_stratifier.assess_comprehensive_risk(...)
        trajectories = trajectory_predictor.predict_all(...)
        
        # 3. Retrieve evidence
        evidence = retrieve_evidence(context, request.message)
        
        # 4. Generate response
        llm_response = llm_service.generate_advisory(
            context=context,
            user_question=request.message,
            retrieved_evidence=evidence,
        )
        
        # 5. Validate
        validation = citation_validator.validate_response(llm_response)
        
        # 6. Assemble final response
        return ChatResponse(
            message=assistant_message,
            citations=validation.corrected_citations,
            risks=risk_assessment,
        )
```

### Knowledge Base Query Flow

```
Query: "How can I prevent falls?"
    │
    ▼
┌──────────────────────────────────────────────┐
│ Topic Extraction                              │
│ → "fall_prevention", "safety"                 │
└──────────────────────────────────────────────┘
    │
    ├──► Clinical Guidelines DB
    │      └── Search by category "fall_prevention"
    │      └── Return: AGS Falls 2023 guidelines
    │
    ├──► Research Corpus
    │      └── Search by keywords ["falls", "prevention", "elderly"]
    │      └── Return: Cameron 2018 Cochrane review
    │
    └──► ICOPE Standards
           └── Get domain interventions for locomotor
           └── Return: Exercise recommendations
    │
    ▼
┌──────────────────────────────────────────────┐
│ Rank & Deduplicate                            │
│ → Sort by evidence level (A > B > C)          │
│ → Remove duplicates                           │
│ → Limit to top 10                             │
└──────────────────────────────────────────────┘
```

## Safety Architecture

### Claim Validation Pipeline

```
LLM Generated Claim
    │
    ▼
┌──────────────────────────────────────────────┐
│ 1. Citation Extraction                        │
│    Extract [Source: ID] references            │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ 2. Citation Verification                      │
│    Check against:                             │
│    • Clinical guidelines DB                   │
│    • Research corpus                          │
│    • Known citation patterns                  │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ 3. Medical Claim Validation                   │
│    Check claim against:                       │
│    • Guideline recommendations                │
│    • Evidence-based facts                     │
│    Flag questionable claims                   │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ 4. Evidence Quality Scoring                   │
│    Calculate overall evidence score           │
│    Grade: High / Moderate / Low               │
└──────────────────────────────────────────────┘
    │
    ▼
Validation Result (Pass / Review / Fail)
```

## Scalability Considerations

### Current Design (Single Instance)
- In-memory session storage
- Synchronous LLM calls
- File-based knowledge base

### Production Scaling
- Redis for session storage
- Async LLM with request queue
- Vector database (ChromaDB/Pinecone)
- Caching layer for common queries
- Horizontal scaling of API layer

---

## Security Model

### Data Access
- **Read-only** access to Beta 5.5 data
- No PHI sent to external LLMs (only anonymized summaries)
- All citations verified before display

### API Security
- Session-based authentication
- Rate limiting per elder_id
- Input validation/sanitization

---

## Module Boundaries

```
health_advisory_chatbot/     ← This module (NEW)
├── backend/                 ← Python backend
│   ├── chatbot/            ← Core logic
│   ├── models/             ← Pydantic schemas
│   └── api/                ← Route handlers
│
Beta_5.5/                    ← Existing codebase (UNTOUCHED)
├── backend/                 ← Beta 5.5 backend
│   ├── elderlycare_v1_16/  ← Existing services
│   └── ...                 ← Other modules
│
Integration: READ-ONLY via ContextFusionEngine
```

---

## External Dependencies

### Required
- Python 3.9+
- Pydantic 2.0+

### Optional (LLM)
- OpenAI API key (for GPT-4)
- Anthropic API key (for Claude)

### Future (RAG Enhancement)
- sentence-transformers
- ChromaDB or Pinecone
- Redis (for session storage)

---

*Document Version: 1.0.0*  
*Last Updated: 2026-01-31*
