# Demo Dashboard - Usage Guide

## 🚀 Quick Start

### 1. Start the Demo Server

```bash
cd health_advisory_chatbot/demo
./start_demo.sh
```

Or manually:

```bash
cd health_advisory_chatbot/demo/backend
PYTHONPATH="../../backend:$PYTHONPATH" python3 demo_server.py
```

### 2. Open Dashboard

Open your browser to: **http://localhost:8000**

---

## 🎭 Demo Scenarios

### Margaret Chen - High Fall Risk
- **Age:** 82
- **Conditions:** Type 2 Diabetes, Hypertension, Osteoarthritis
- **Medications:** Metformin, Amlodipine, **Lorazepam (sedative)**
- **Risk Factors:**
  - 3+ nighttime bathroom visits
  - Benzodiazepine use
  - Reduced mobility
- **Expected Chatbot Focus:** Fall prevention, medication safety, nighttime safety

**Sample Questions:**
- "What are my fall risks?"
- "Should I be worried about my medications?"
- "How can I prevent falls at night?"

---

### Robert Williams - Cognitive Concerns
- **Age:** 78
- **Conditions:** Hypertension, Mild Cognitive Impairment, **Sleep Apnea**
- **Medications:** Lisinopril, Atorvastatin, Donepezil
- **Risk Factors:**
  - Memory concerns
  - Sleep fragmentation
  - High sleep apnea risk
- **Expected Chatbot Focus:** Cognitive health, sleep improvement, vascular risk

**Sample Questions:**
- "How is my memory?"
- "Why do I wake up tired?"
- "How can I improve my sleep?"

---

### Helen Thompson - Generally Healthy
- **Age:** 75
- **Conditions:** Osteoarthritis (mild)
- **Medications:** Acetaminophen (PRN), Vitamin D
- **Status:** Active, independent, good sleep
- **Expected Chatbot Focus:** Maintenance, prevention, healthy aging

**Sample Questions:**
- "How did I sleep last night?"
- "What can I do to stay healthy?"
- "Am I taking my medications correctly?"

---

## 📊 Dashboard Features

### Left Panel: Elder Selection
- Click on any elder card to select
- See scenario type and description
- Highlights active selection

### Center Panel: Chat Interface
- **Messages:** Full conversation history
- **Suggestions:** Quick-question buttons
- **Input:** Type custom questions
- **Evidence:** Citations and confidence scores

### Right Panel: Health Assessment
- **Overall Risk Score:** 0-100 with color coding
  - 🔴 Red: High/Critical (60+)
  - 🟡 Yellow: Moderate (40-59)
  - 🟢 Green: Low (<40)
- **Domain Risks:** Fall, Cognitive, Sleep, Medication
- **Critical Alerts:** Immediate concerns
- **Top Risk Factors:** Weighted risk contributors

---

## 🔍 What to Observe

### 1. Risk Assessment Accuracy
- Margaret should show **high fall risk** (sedative + nocturia)
- Robert should show **cognitive + sleep concerns**
- Helen should show **low overall risk**

### 2. Evidence-Based Responses
All responses include:
- Clinical guideline citations
- Evidence levels (A/B/C)
- Confidence scores
- Source links

### 3. Context Awareness
The chatbot knows:
- Medical history (conditions, meds)
- ADL patterns (nighttime visits)
- ICOPE scores (6 domains)
- Sleep quality metrics

### 4. Safety Features
- Drug interaction warnings
- Fall risk alerts
- Contraindication flags
- Medical disclaimers

---

## 🧪 Testing Checklist

### Core Functionality
- [ ] Select each elder profile
- [ ] Send sample questions
- [ ] View risk assessments
- [ ] Check citations expand/collapse
- [ ] Verify suggestion buttons work

### Risk Assessment
- [ ] Margaret shows high fall risk (60+)
- [ ] Robert shows cognitive concerns
- [ ] Helen shows low risk (<40)
- [ ] Risk factors match profile

### Evidence Quality
- [ ] Citations appear in responses
- [ ] Evidence badges show quality
- [ ] Sources are from AGS/WHO/AASM
- [ ] Confidence scores displayed

### Edge Cases
- [ ] Empty message handling
- [ ] Rapid question sending
- [ ] Switching elders mid-chat
- [ ] Long messages

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python3 demo_server.py --port 8001
```

### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="../../backend:$PYTHONPATH"
```

### Module Not Found
```bash
# Install pydantic
pip install pydantic
```

---

## 📈 Performance

Expected metrics on a modern laptop:
- **Cold Start:** < 2 seconds
- **Risk Calculation:** < 100ms
- **Response Generation:** < 50ms (mock mode)
- **Total Request Time:** < 200ms

---

## 🎯 Demo Goals

By the end of this demo, you should understand:

1. **How the chatbot integrates** medical history, ADL, ICOPE, and sleep data
2. **How risk assessment** works with multiple domains
3. **How evidence-based** recommendations are generated
4. **How the UI presents** complex health information clearly
5. **How the system** maintains safety and accuracy

---

## 📝 Notes

- This demo uses **mock data** (no real Beta 5.5 connection)
- LLM responses are **mocked** (no API key required)
- Data **resets** when server restarts
- Perfect for **demonstrations** and **development**

---

## 🚢 Next Steps

After demo approval:
1. Integrate with Beta 5.5 services
2. Add real LLM (OpenAI/Claude)
3. Deploy to staging environment
4. User acceptance testing

---

**Demo Version:** 1.0.0  
**Last Updated:** 2026-01-31
