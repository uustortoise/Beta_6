# Health Advisory Chatbot Module

## Overview
An evidence-based health advisory chatbot for elderly care, integrating ADL history, medical records, ICOPE assessments, and sleep analysis to provide predictive, research-grounded health advice.

## Architecture
```
health_advisory_chatbot/
├── backend/           # Python backend services
│   ├── chatbot/       # Core chatbot logic
│   ├── api/           # REST API endpoints
│   ├── models/        # Pydantic schemas
│   └── tests/         # Unit & integration tests
├── frontend/          # React/TypeScript components
├── database/          # Schema migrations
├── scripts/           # Utility scripts
└── docs/              # Architecture & API documentation
```

## Key Features
- **Evidence-Based**: All advice grounded in peer-reviewed medical research
- **Predictive Risk Modeling**: Fall risk, cognitive decline, frailty trajectory
- **Medical History Integration**: Conditions, medications, allergies from EnhancedProfile
- **Citation Transparency**: Every recommendation includes source and confidence level
- **Elder-Friendly UI**: Accessibility-first design

## Development Status
See `docs/PROGRESS.md` for detailed implementation status.

## Installation
```bash
# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
cd frontend && npm install
```

## Testing
```bash
# Backend tests
pytest backend/tests/

# Frontend tests
npm test
```

## License
Internal module for Beta 5.5 Elderly Care Platform
