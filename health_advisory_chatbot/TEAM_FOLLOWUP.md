# Health Advisory Chatbot - Development Follow-up

This document outlines the required actions for the development team based on the technical review conducted on 2026-02-01.

---

## 🔴 CRITICAL / HIGH PRIORITY

### 🏥 Clinical Safety
- [ ] **Emergency Symptom Detection**: Implement a protocol to detect "red flag" symptoms (chest pain, unilateral weakness, severe shortness of breath) and immediately stop the chat to provide emergency contact instructions (e.g., "Call 911 immediately").
- [ ] **Delirium Screening**: Add a screen for acute cognitive changes using CAM-ICU or similar brief assessment tools.

### ⚙️ Core Engineering
- [ ] **Fix Flask Async Mismatch**: In `backend/api/routes.py` (lines 273-278), the Flask routes call `api.chat()` which is an `async` function. This needs to be wrapped in `asyncio.run()` or `async_to_sync()` to prevent runtime failures in Flask.
- [ ] **Medication Complexity Thresholds**: Update the risk stratifier to account for medication complexity/risk level, not just raw pill count.

---

## 🟡 MEDIUM PRIORITY

### 🛡️ Infrastructure & Security
- [ ] **Production Session Storage**: Replace the current in-memory `self._sessions` dictionary with a persistent store like Redis or a database to support scalability and survival across restarts.
- [ ] **API Rate Limiting**: Implement basic rate limiting on the `/api/chat` endpoint to prevent resource exhaustion.
- [ ] **Input Sanitization**: Ensure all user inputs are sanitized against XSS before being used in LLM prompts or rendered in the history.

### 🧪 Quality Assurance
- [ ] **Expand Test Coverage**: Aim for >80% coverage. Specifically add tests for:
    - LLM service mocking and fallback logic.
    - Citation validator edge cases.
    - Concurrent session handling.

---

## 🔵 LOW PRIORITY / PHASE 2

- [ ] **Dependency Injection Refactor**: Move away from many singletons in `chatbot/core/` and `chatbot/predictive/` toward a more robust dependency injection pattern.
- [ ] **Accessibility (UI)**: Add a font-size toggle and consider voice-to-text input options for elders with mobility or vision impairments.
- [ ] **Caregiver Notifications**: Implement a system to notify caregivers/nurses when a "High" or "Critical" risk score is generated.

---

**Link to Full Review:** [health_advisory_chatbot_review.md](file:///Users/dicksonng/.gemini/antigravity/brain/46517405-81e4-48f9-a717-456194459f7b/health_advisory_chatbot_review.md)
