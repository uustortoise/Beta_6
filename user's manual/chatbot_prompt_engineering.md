# Health Advisory Chatbot - Prompt Engineering Guide

> **Document Purpose:** This guide documents the prompt structure and logic used by the Health Advisory Chatbot when interfacing with LLM providers (DeepSeek, OpenAI, Anthropic). Use this for future modifications, A/B testing, and discussion of prompt engineering strategies.

---

## Overview

The chatbot uses a **structured prompt composition** approach that combines:
- System instructions (behavior, safety rules, formatting)
- Patient health context (risk scores, conditions, medications)
- Evidence-based citations (clinical guidelines, research)
- User query

---

## Prompt Structure

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SYSTEM PROMPT                                            │
│    - Role definition                                        │
│    - Core principles                                        │
│    - Response structure                                     │
│    - Safety rules                                           │
│    - Citation format                                        │
├─────────────────────────────────────────────────────────────┤
│ 2. PATIENT HEALTH CONTEXT                                   │
│    - Demographics (age, gender)                             │
│    - Medical conditions                                     │
│    - Medications                                            │
│    - Mobility status                                        │
│    - Cognitive status                                       │
├─────────────────────────────────────────────────────────────┤
│ 3. RISK ASSESSMENT                                          │
│    - Overall risk score                                     │
│    - Domain-specific risks (fall, cognitive, sleep)         │
│    - Top risk factors with weighted scores                  │
├─────────────────────────────────────────────────────────────┤
│ 4. PREDICTED TRAJECTORIES                                   │
│    - Domain → Current → Predicted (confidence %)            │
├─────────────────────────────────────────────────────────────┤
│ 5. RETRIEVED EVIDENCE                                       │
│    - Clinical guidelines                                    │
│    - Research papers                                        │
│    - Evidence levels                                        │
├─────────────────────────────────────────────────────────────┤
│ 6. CONVERSATION HISTORY (last 3 messages)                   │
├─────────────────────────────────────────────────────────────┤
│ 7. USER QUESTION                                            │
├─────────────────────────────────────────────────────────────┤
│ 8. RESPONSE INSTRUCTIONS                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. System Prompt

```markdown
You are a health advisory assistant for elderly care, grounded in evidence-based medicine.

CORE PRINCIPLES:
1. All recommendations must be supported by clinical guidelines or peer-reviewed research
2. Cite sources for all medical claims using [Source: ID] format
3. Include confidence levels (High/Medium/Low) for each recommendation
4. Flag contraindications and safety concerns prominently
5. Use clear, elder-friendly language
6. Always include a medical disclaimer

RESPONSE STRUCTURE:
1. Brief summary of key findings
2. Risk assessment with specific metrics
3. Evidence-based recommendations with citations
4. Actionable next steps
5. When to seek immediate medical attention

SAFETY RULES:
- Never recommend stopping prescribed medications
- Flag potential drug interactions clearly
- Highlight fall risks and safety concerns
- Recommend professional consultation for significant concerns
- Do not provide specific dosing adjustments

CITATION FORMAT:
Use [Source: ID] after each medical claim. IDs will be resolved to full citations.
```

### Discussion Points:
- **Temperature:** Currently set to 0.3 for medical accuracy vs. creativity
- **Max tokens:** 2000 (may need adjustment for complex cases)
- **Language:** Currently English only - i18n support needed?

---

## 2. Patient Health Context

### Data Fields Included:

| Field | Example | Source |
|-------|---------|--------|
| `full_name` | Margaret Chen | EnhancedProfile |
| `age` | 82 | EnhancedProfile |
| `gender` | Female | EnhancedProfile |
| `chronic_conditions` | Type 2 Diabetes, Hypertension, Osteoarthritis | EnhancedProfile |
| `medications` | Metformin, Lisinopril, Zolpidem | EnhancedProfile |
| `mobility_status` | Uses walker, assisted locomotion | EnhancedProfile |
| `cognitive_status` | Normal | EnhancedProfile |

### Example Output:
```markdown
--- PATIENT HEALTH CONTEXT ---
Margaret Chen (Age 82, Female)
Conditions: Type 2 Diabetes, Hypertension, Osteoarthritis
Medications: Metformin, Lisinopril, Zolpidem (sedative)
Mobility: Uses walker, assisted locomotion
Cognitive: Normal
```

---

## 3. Risk Assessment

### Calculated Metrics:

| Metric | Range | Description |
|--------|-------|-------------|
| `overall_risk_score` | 0-100 | Composite risk across all domains |
| `overall_risk_level` | low/moderate/high/critical | Categorical classification |
| `fall_risk` | 0-100 | Fall probability based on gait, medications, history |
| `cognitive_decline_risk` | 0-100 | Cognitive trajectory prediction |
| `sleep_disorder_risk` | 0-100 | Sleep quality and apnea risk |
| `medication_risk` | 0-100 | Polypharmacy and interaction risk |

### Top Risk Factors (Top 3):

```markdown
TOP RISK FACTORS:
- Sedative/hypnotic use (weighted score: 30.0)
- Mobility impairment (weighted score: 23.0)
- Low sleep efficiency (weighted score: 15.0)
```

### Risk Stratification Algorithm:
- **Low:** 0-30
- **Moderate:** 31-60
- **High:** 61-80
- **Critical:** 81-100

---

## 4. Predicted Trajectories

### Trajectory Domains:

| Domain | Current Status | Predicted Status | Confidence |
|--------|---------------|------------------|------------|
| mobility | assisted | assisted | 85% |
| sleep | poor | poor | 75% |
| cognitive | normal | normal | 90% |

### Format:
```markdown
PREDICTED TRAJECTORIES:
- {domain}: {current_status} -> {predicted_status} (confidence: {confidence}%)
```

---

## 5. Retrieved Evidence

### Evidence Sources:

1. **Clinical Guidelines Database**
   - AGS/BGS Falls Prevention Guidelines
   - WHO ICOPE Guidelines
   - ADA Diabetes Management

2. **Research Corpus**
   - PubMed-indexed papers
   - Cochrane Reviews
   - JAGS publications

### Citation Format:
```markdown
--- RETRIEVED EVIDENCE ---

[1] {title}
    Authors: {authors}
    Journal: {journal}, {year}
    Evidence Level: {high/medium/low}
```

### Evidence Levels:
- **High:** Systematic reviews, RCTs, major guidelines
- **Medium:** Cohort studies, case-control
- **Low:** Case reports, expert opinion

---

## 6. Conversation History

### Inclusion Logic:
- Last 3 messages only (to manage context window)
- Truncated to 200 characters per message
- Format: `USER:` / `ASSISTANT:`

### Example:
```markdown
--- CONVERSATION HISTORY ---
USER: How did I sleep last night?...
ASSISTANT: Based on your sleep data, you had 5.9 hours of sleep...
```

---

## 7. User Question

### Input Processing:
- Raw user query (no preprocessing)
- Natural language questions
- Can be short ("status") or detailed

### Example Questions:
- "get the status of the user"
- "How did I sleep last night?"
- "What are my fall risks?"
- "Should I be worried about my medications?"

---

## 8. Response Instructions

### Output Requirements:
```markdown
--- RESPONSE INSTRUCTIONS ---
1. Answer the user's specific question
2. Cite sources using [Source: ID] format
3. Include confidence level for each recommendation
4. Highlight any urgent concerns
5. End with medical disclaimer
```

---

## Complete Example Prompt

```markdown
SYSTEM: You are a health advisory assistant for elderly care, grounded in evidence-based medicine.
[... system prompt ...]

--- PATIENT HEALTH CONTEXT ---
Margaret Chen (Age 82, Female)
Conditions: Type 2 Diabetes, Hypertension, Osteoarthritis
Medications: Metformin, Lisinopril, Zolpidem (sedative)
Mobility: Uses walker, assisted locomotion
Cognitive: Normal

RISK ASSESSMENT:
- Overall Risk: low (28/100)
- Fall Risk: 45/100
- Cognitive Risk: 10/100
- Sleep Risk: 30/100

TOP RISK FACTORS:
- Sedative/hypnotic use (weighted score: 30.0)
- Mobility impairment (weighted score: 23.0)
- Low sleep efficiency (weighted score: 15.0)

PREDICTED TRAJECTORIES:
- mobility: assisted -> assisted (confidence: 85%)
- sleep: poor -> poor (confidence: 75%)
- cognitive: normal -> normal (confidence: 90%)

--- RETRIEVED EVIDENCE ---

[1] Falls Prevention in Older Adults
    Authors: AGS, BGS
    Journal: Journal of the American Geriatrics Society, 2023
    Evidence Level: high

[2] Exercise for Fall Prevention
    Authors: Sherrington C, et al.
    Journal: Cochrane Database, 2019
    Evidence Level: high

--- USER QUESTION ---
get the status of the user

--- RESPONSE INSTRUCTIONS ---
1. Answer the user's specific question
2. Cite sources using [Source: ID] format
3. Include confidence level for each recommendation
4. Highlight any urgent concerns
5. End with medical disclaimer
```

---

## Prompt Engineering Considerations

### Current Token Count:
- System prompt: ~300 tokens
- Patient context: ~100 tokens
- Risk assessment: ~50 tokens
- Trajectories: ~30 tokens
- Evidence (5 items): ~150 tokens
- Conversation history: ~100 tokens
- User question: ~10 tokens
- **Total: ~740 tokens** (leaves ~1,260 for response)

### Optimization Opportunities:

1. **Dynamic Context Truncation**
   - Prioritize high-risk factors when token limit approached
   - Summarize older conversation history

2. **Evidence Selection**
   - Currently top 5 citations
   - Could use semantic similarity to query
   - Weight by recency and evidence level

3. **Personalization**
   - Adjust language complexity based on user profile
   - Cultural considerations for diverse populations

4. **Multi-language Support**
   - System prompt translation
   - Evidence corpus in multiple languages

---

## A/B Testing Ideas

| Test | Hypothesis | Metric |
|------|-----------|--------|
| Shorter system prompt | Faster response time | Latency |
| More evidence items | Higher citation accuracy | Citation validator score |
| Trajectory confidence thresholds | Better prediction calibration | User trust score |
| Risk factor ordering | Improved user attention | Click-through on recommendations |

---

## Related Documents

- [ICope Framework](./icope_framework.md)
- [Data Architecture](./data_architecture.md)
- [Algorithmic Logic](./algorithmic_logic.md)
- [Operation Manual](./operation_manual.md)

---

## Change Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-01 | AI Assistant | Initial documentation of prompt structure |

---

## Discussion Notes

<!-- Use this section for team discussions, decisions, and open questions -->

### Open Questions:
1. Should we include family caregiver context in prompts?
2. How to handle conflicting evidence sources?
3. What is the optimal number of risk factors to display?
4. Should we include medication timing (AM/PM) in context?

### Decisions:
- [2026-02-01] Keep temperature at 0.3 for medical safety
- [2026-02-01] Limit to top 3 risk factors to avoid overwhelming users
