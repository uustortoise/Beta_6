# Health Advisory Chatbot - Demo Dashboard

A standalone demo to showcase the chatbot functionality without requiring Beta 5.5 integration.

## Features

- 🎭 Mock Elder Profiles (3 pre-configured scenarios)
- 💬 Interactive Chat Interface
- 📊 Health Context Visualization
- 🎯 Risk Assessment Display
- 📚 Evidence Citations
- ⚡ Real-time Advisory Generation (mock mode)

## Quick Start

### Option 1: Python Demo (Backend Only)

```bash
cd health_advisory_chatbot/demo/backend
python demo_server.py
```

Then open `http://localhost:8000` in your browser.

### Option 2: Full Stack Demo (Backend + Frontend)

```bash
# Terminal 1 - Start backend
cd health_advisory_chatbot/demo/backend
python demo_server.py

# Terminal 2 - Start frontend
cd health_advisory_chatbot/demo/frontend
npm install
npm run dev
```

Then open `http://localhost:3000` in your browser.

## Mock Elder Scenarios

### 1. Margaret (High Fall Risk)
- Age: 82, Diabetes, Hypertension
- 3+ nighttime bathroom visits
- Sedative use
- **Expected Chatbot Focus:** Fall prevention, medication review

### 2. Robert (Cognitive Concerns)
- Age: 78, Early memory issues
- Sleep apnea risk
- Social isolation
- **Expected Chatbot Focus:** Cognitive health, sleep improvement

### 3. Helen (Generally Healthy)
- Age: 75, Active lifestyle
- Minor arthritis
- Good sleep
- **Expected Chatbot Focus:** Maintenance, preventive care

## Demo Capabilities

✅ **Chat Interface**
- Ask health questions
- Get evidence-based responses
- View citations
- See risk assessments

✅ **Health Dashboard**
- View elder profile
- See ADL summary
- Check ICOPE scores
- Review sleep data

✅ **Risk Visualization**
- Overall risk score
- Domain-specific risks (fall, cognitive, sleep)
- Trend indicators

✅ **Evidence Explorer**
- Browse clinical guidelines
- View research citations
- Check evidence quality

## API Endpoints (Demo)

```
GET  /                    → Demo dashboard HTML
GET  /api/demo/elders     → List mock elders
GET  /api/demo/elders/:id → Get elder profile
POST /api/chat            → Send message (mock LLM)
GET  /api/demo/context/:id → Get health context
```

## Architecture

```
┌─────────────────────────────────────────────┐
│           Demo Dashboard (Browser)          │
├─────────────────────────────────────────────┤
│  Elder Selector │ Chat Window │ Risk Panel │
└─────────────────────────────────────────────┘
                   │
                   ▼ HTTP
┌─────────────────────────────────────────────┐
│           Demo Server (Python)              │
│  ┌─────────────────────────────────────┐   │
│  │  Mock Data Generator                │   │
│  │  • Elder profiles                   │   │
│  │  • ADL data                         │   │
│  │  • ICOPE scores                     │   │
│  │  • Sleep analysis                   │   │
│  └─────────────────────────────────────┘   │
│  ┌─────────────────────────────────────┐   │
│  │  Advisory Engine (Real)             │   │
│  │  • Context fusion                   │   │
│  │  • Risk stratification              │   │
│  │  • Evidence retrieval               │   │
│  │  • Mock LLM generation              │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Notes

- This demo uses **mock LLM responses** (no API key required)
- All Beta 5.5 services are mocked
- Data resets on server restart
- Perfect for demonstrations and testing

## Screenshots

[Dashboard](./screenshots/dashboard.png)  
[Chat Interface](./screenshots/chat.png)  
[Risk Assessment](./screenshots/risk.png)
