# Health Advisory Chatbot - Mobile Deployment Planning Document

**Version:** 2.0  
**Date:** 2026-02-10  
**Status:** Updated with Current System State - Backend Production-Ready

---

## 1. Executive Summary

### Current State (February 2026)
The Health Advisory Chatbot (Beta 5.5) is now **production-ready** for mobile deployment:

| Component | Status | Details |
|-----------|--------|---------|
| **Backend API** | ✅ Complete | All CRUD endpoints operational, JSON-based |
| **Knowledge Base** | ✅ Complete | 65+ entries (12 guidelines, 10 drugs, 23 research, 8 FAQs) |
| **Admin UI** | ✅ Complete | Web-based CRUD editor with activity logging |
| **Topic Extraction** | ✅ Enhanced | 9-topic intelligent matching with variations |
| **Citation System** | ✅ Complete | Fail-closed validation with evidence tracking |
| **LLM Integration** | ✅ Active | DeepSeek API integrated and tested |
| **Demo Server** | ✅ Running | http://localhost:8000 with 3 mock elders |

### Mobile Deployment Feasibility: ✅ **READY FOR DEVELOPMENT**

The backend is **fully mobile-ready**. Required changes for production:
- Cloud deployment
- JWT authentication
- HTTPS/SSL

**No changes needed to core chatbot logic or API structure.**

---

## 2. Current Architecture Overview

### 2.1 Data Flow (Current Web)

```
┌─────────────────┐      HTTP/REST      ┌─────────────────────────────────────┐
│   Web Browser   │ ◄──────────────────► │  Next.js API Layer (localhost:3001) │
│  (React/Next)   │                      └─────────────────────────────────────┘
└─────────────────┘                                           │
                                                              ▼
                        ┌─────────────────────────────────────────────────────┐
                        │              Python Backend API                      │
                        │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
                        │  │   Context   │  │  Predictive │  │    LLM      │ │
                        │  │   Fusion    │  │    Risk     │  │   Service   │ │
                        │  │   Engine    │  │   Engine    │  │   (RAG)     │ │
                        │  └─────────────┘  └─────────────┘  └─────────────┘ │
                        └─────────────────────────────────────────────────────┘
                                               │
            ┌──────────────────────────────────┼──────────────────────────────────┐
            ▼                                  ▼                                  ▼
   ┌─────────────────┐              ┌─────────────────┐               ┌─────────────────┐
   │  Beta 5.5 Data  │              │  Knowledge Base │               │  External LLM   │
   │  (SQLite/Local) │              │  (Clinical DB)  │               │  (OpenAI/Claude)│
   └─────────────────┘              └─────────────────┘               └─────────────────┘
```

### 2.2 API Endpoints (Current)

| Endpoint | Method | Purpose | Mobile Ready? |
|----------|--------|---------|---------------|
| `/api/chat` | POST | Send message, get advisory | ✅ Yes |
| `/api/chat/history/{session_id}` | GET | Conversation history | ✅ Yes |
| `/api/chat/suggestions` | GET | Suggested questions | ✅ Yes |
| `/api/health` | GET | Health check | ✅ Yes |

### 2.3 Key Data Models

**Request:** `ChatRequest`
```typescript
{
  elder_id: string;
  message: string;
  session_id?: string;
  include_medical_history: boolean;
  include_adl_data: boolean;
  include_icope_data: boolean;
  include_sleep_data: boolean;
  require_citations: boolean;
}
```

**Response:** `ChatResponse`
```typescript
{
  session_id: string;
  message: Message;
  health_context: HealthContext;
  current_risks: RiskAssessment;
  recommendations: AdvisoryRecommendation[];
  response_time_ms: number;
}
```

---

## 3. Mobile Deployment Options

### Summary Comparison

| Option | Effort | Pros | Cons | Best For |
|--------|--------|------|------|----------|
| **PWA** | 1-2 days | Fastest, no app store | Limited native features | MVP validation |
| **React Native** | 4-6 weeks | Native features, best UX | Requires mobile expertise | Production app |
| **WebView** | 1 week | Fast, reuse web UI | Poor UX | Prototype only |

### 3.1 Option A: React Native (Recommended)

**Approach:** Build native iOS/Android apps using React Native

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MOBILE APP (React Native)                          │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │   iOS Native    │    │  Android Native │    │    Shared JavaScript    │ │
│  │      UI         │    │       UI        │    │  (Hooks, State, API)    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
│           │                      │                         │               │
│           └──────────────────────┴─────────────────────────┘               │
│                                      │                                      │
│                              React Native Bridge                           │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │
                                       ▼ HTTPS/WebSocket
                            ┌─────────────────────┐
                            │   Backend API       │
                            │   (Cloud-hosted)    │
                            └─────────────────────┘
```

**Pros:**
- 70-80% code reuse from existing React frontend
- Native performance and UI feel
- Access to native features (push notifications, camera, biometrics)
- Single codebase for iOS + Android

**Cons:**
- Requires mobile development expertise
- Native module integration complexity
- App store approval process

**Estimated Effort:** 4-6 weeks

---

### 3.2 Option B: Progressive Web App (PWA)

**Approach:** Convert existing Next.js app to PWA

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PWA (Browser + Native Feel)                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Existing Next.js Web App                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Service     │  │   Web App   │  │   Push      │  │   Offline  │  │   │
│  │  │   Worker    │  │   Manifest  │  │   Notif     │  │   Cache    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                              Install Prompt → Home Screen                    │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │ HTTPS
                                       ▼
                            ┌─────────────────────┐
                            │   Backend API       │
                            └─────────────────────┘
```

**Pros:**
- Minimal code changes (existing web app)
- No app store required
- Automatic updates
- Works offline with service workers

**Cons:**
- Limited native feature access
- iOS restrictions on push notifications
- Not "true" native app experience

**Estimated Effort:** 1-2 days

---

### 3.1.5 Option P: PWA (Recommended for MVP)

**Approach:** Convert existing Next.js web app to Progressive Web App

**Implementation:**
```javascript
// Add to next.config.js
const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
});
```

**Pros:**
- Fastest deployment (1-2 days)
- No app store approval process
- Automatic updates
- Works offline with service workers
- Same codebase as web

**Cons:**
- Limited native feature access on iOS
- No push notifications on iOS (Android OK)
- Not "true" native app experience

**Use Case:** Quick validation of mobile demand before investing in native development.

### 3.3 Option C: Hybrid (WebView Wrapper)

**Approach:** Wrap existing web app in native WebView container

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HYBRID APP (WebView)                                 │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Native Container (iOS/Android)                   │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              WebView (renders existing web app)             │   │   │
│  │  │                                                             │   │   │
│  │  │   ┌─────────────────────────────────────────────────────┐   │   │   │
│  │  │   │         Existing React/Next.js Chatbot UI           │   │   │   │
│  │  │   └─────────────────────────────────────────────────────┘   │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                              │                                      │   │
│  │                    JavaScript Bridge (JS → Native)                 │   │
│  └──────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │ HTTPS
                                  ▼
                       ┌─────────────────────┐
                       │   Backend API       │
                       └─────────────────────┘
```

**Pros:**
- Fastest deployment (reuse 100% of web UI)
- Access to some native features via bridge
- Single codebase

**Cons:**
- WebView performance limitations
- Limited offline capability
- Not truly native UX

**Estimated Effort:** 1 week

---

## 4. Proposed Mobile Data Flow

### 4.1 Recommended Architecture (React Native)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MOBILE CLIENT                                   │
│                         (React Native + TypeScript)                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PRESENTATION LAYER                           │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │   │
│  │  │  Chat Screen   │  │  Risk Alerts   │  │   Health Dashboard     │  │   │
│  │  │  (ChatWindow)  │  │   (Native)     │  │   (Timeline/Charts)    │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         BUSINESS LOGIC LAYER                         │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │   │
│  │  │  useChatbot()  │  │  Risk Monitor  │  │   Notification Mgr     │  │   │
│  │  │    (Hook)      │  │    (Service)   │  │   (Push/Local)         │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA LAYER                                  │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │   │
│  │  │   API Client   │  │ Secure Storage │  │    Offline Cache       │  │   │
│  │  │  (Axios/Fetch) │  │  (Keychain/    │  │   (AsyncStorage/       │  │   │
│  │  │                │  │   Keystore)    │  │     WatermelonDB)      │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       │ HTTPS + Auth Token
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BACKEND SERVICES                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      API GATEWAY (Cloud)                             │   │
│  │         Auth (JWT) │ Rate Limiting │ SSL │ Request Routing           │   │
│  └────────────────────────────────────┬────────────────────────────────┘   │
│                                       │                                     │
│  ┌────────────────────────────────────┼────────────────────────────────┐   │
│  │                         CORE SERVICES                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Chatbot   │  │  Context    │  │  Predictive │  │    LLM     │  │   │
│  │  │     API     │  │   Fusion    │  │    Risk     │  │  Service   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                       │                                     │
│  ┌────────────────────────────────────┼────────────────────────────────┐   │
│  │                           DATA SOURCES                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Beta 5.5  │  │   Session   │  │  Knowledge  │  │  External  │  │   │
│  │  │    Data     │  │    Store    │  │     Base    │  │    LLM     │  │   │
│  │  │  (SQLite)   │  │  (Redis)    │  │  (VectorDB) │  │  (OpenAI)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Authentication Flow

```
┌──────────┐                    ┌──────────┐                    ┌──────────┐
│   User   │                    │  Mobile  │                    │ Backend  │
└────┬─────┘                    └────┬─────┘                    └────┬─────┘
     │                               │                               │
     │ 1. Open App                   │                               │
     │──────────────────────────────>│                               │
     │                               │                               │
     │                               │ 2. Check Secure Storage        │
     │                               │    (Existing Token?)           │
     │                               │                               │
     │                               │                               │
     │ 3. No Token / Expired         │                               │
     │<──────────────────────────────│                               │
     │                               │                               │
     │ 4. Login (Biometric/PIN)      │                               │
     │──────────────────────────────>│                               │
     │                               │                               │
     │                               │ 5. POST /auth/login           │
     │                               │──────────────────────────────>│
     │                               │                               │
     │                               │ 6. Return JWT + Refresh Token │
     │                               │<──────────────────────────────│
     │                               │                               │
     │                               │ 7. Store Securely             │
     │                               │    (Keychain/Keystore)        │
     │                               │                               │
     │ 8. Show Chat UI               │                               │
     │<──────────────────────────────│                               │
     │                               │                               │
     │ 9. Send Message               │                               │
     │──────────────────────────────>│                               │
     │                               │                               │
     │                               │ 10. POST /api/chat            │
     │                               │     Authorization: Bearer JWT │
     │                               │──────────────────────────────>│
     │                               │                               │
     │                               │ 11. Response                  │
     │                               │<──────────────────────────────│
     │                               │                               │
     │ 12. Display Response          │                               │
     │<──────────────────────────────│                               │
```

### 4.3 Push Notification Flow (Critical Risk Alerts)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RISK ALERT NOTIFICATION FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Backend   │      │   FCM/APNs  │      │  User's     │      │   Mobile    │
│   Engine    │      │  (Push Svc) │      │   Device    │      │    App      │
└──────┬──────┘      └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
       │                    │                    │                    │
       │ 1. Detect High     │                    │                    │
       │    Fall Risk       │                    │                    │
       │                    │                    │                    │
       │ 2. Check User      │                    │                    │
       │    Preferences     │                    │                    │
       │                    │                    │                    │
       │ 3. Send Push       │                    │                    │
       │───────────────────>│                    │                    │
       │                    │                    │                    │
       │                    │ 4. Route to Device │                    │
       │                    │───────────────────>│                    │
       │                    │                    │                    │
       │                    │                    │ 5. Display         │
       │                    │                    │    Notification    │
       │                    │                    │───────────────────>│
       │                    │                    │                    │
       │                    │                    │ 6. User Taps       │
       │                    │                    │<───────────────────│
       │                    │                    │                    │
       │                    │                    │ 7. Open App        │
       │                    │                    │    to Risk View    │
       │                    │                    │                    │
```

---

## 5. Mobile-Specific Features

### 5.1 Core Features (Phase 1)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Chat Interface** | Text-based health advisor chat | P0 |
| **Risk Dashboard** | Visual risk score display | P0 |
| **Message History** | Persistent conversation history | P0 |
| **Secure Login** | Biometric + PIN authentication | P0 |

### 5.2 Enhanced Features (Phase 2)

| Feature | Description | Value |
|---------|-------------|-------|
| **Push Notifications** | Critical risk alerts (fall, medication) | High |
| **Voice Input** | Speech-to-text for elderly users | High |
| **Offline Mode** | Cached responses when no internet | Medium |
| **Dark Mode** | Accessibility for low vision | Medium |

### 5.3 Advanced Features (Phase 3)

| Feature | Description | Value |
|---------|-------------|-------|
| **Document Scanner** | OCR for medical documents | High |
| **Medication Reminders** | Push notification alerts | High |
| **Emergency SOS** | One-tap emergency contact (requires clinical governance: consent, escalation/911 policy, false-positive handling) | Critical |
| **Health Trends** | Charts and visualizations | Medium |

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
```
□ Setup React Native project with TypeScript
□ Configure API client with authentication
□ Port shared types from frontend/types/
□ Implement Secure Storage for tokens
□ Build basic Chat UI screen
□ Integrate with existing /api/chat endpoint
```

### Phase 2: Core Features (Weeks 3-4)
```
□ Implement useChatbot() hook (port from web)
□ Add MessageBubble component (adapt for mobile)
□ Build Risk Alert cards
□ Add conversation history persistence
□ Implement biometric authentication
□ Error handling and retry logic
```

### Phase 3: Backend Infrastructure (Weeks 5-6)
```
□ Deploy backend to cloud (AWS/GCP/Azure)
□ Setup Redis for session management
□ Configure SSL/HTTPS
□ Implement JWT authentication
□ Setup push notification service (FCM/APNs)
□ Add rate limiting and security headers
```

### Phase 4: Polish & Release (Weeks 7-8)
```
□ Add push notifications for critical alerts
□ Implement voice input (STT)
□ Offline mode with caching
□ App store assets and descriptions
□ Beta testing (TestFlight/Internal)
□ App Store / Play Store submission
```

---

## 7. Technical Specifications

### 7.1 Recommended Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **Framework** | React Native (Expo) | Code reuse, fast development |
| **Language** | TypeScript | Type safety, shared with web |
| **State Mgmt** | Zustand | Lightweight, similar to web |
| **Navigation** | Expo Router | File-based like Next.js |
| **Styling** | NativeWind | Tailwind for React Native |
| **Storage** | expo-secure-store | Encrypted local storage |
| **HTTP Client** | Axios | Familiar, interceptors |
| **Push Notif** | expo-notifications | Unified FCM/APNs |

### 7.2 Backend Changes Required

#### Already Complete ✅

| Change | Status | Notes |
|--------|--------|-------|
| **API Design** | ✅ Complete | JSON-based, stateless endpoints |
| **CRUD Operations** | ✅ Complete | Full KB editor with create/update/delete |
| **Topic Extraction** | ✅ Complete | 9-topic intelligent matching |
| **Citation Validation** | ✅ Complete | Fail-closed with evidence tracking |
| **Knowledge Base** | ✅ Complete | 65+ entries, searchable |

#### Required for Mobile Production

| Change | Description | Effort | Priority |
|--------|-------------|--------|----------|
| **Cloud Deployment** | Move from localhost to AWS/GCP | 2-3 days | 🔴 Critical |
| **JWT Authentication** | Add token-based auth for mobile | 2 days | 🔴 Critical |
| **HTTPS/SSL** | SSL certificate for secure transport | 2 hours | 🔴 Critical |
| **CORS** | Allow mobile app origins | 1 hour | 🟡 High |
| **Session Store** | Redis for session management | 1 day | 🟡 High |
| **Push Service** | FCM/APNs integration | 2 days | 🟡 High |
| **Rate Limiting** | Per-user request limits | 1 day | 🟡 High |

### 7.3 Security Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SECURITY LAYERS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TRANSPORT SECURITY                                                       │
│     ✅ HTTPS only (TLS 1.3)                                                  │
│     ✅ Certificate pinning (optional)                                        │
│     ✅ No HTTP fallback                                                      │
│                                                                              │
│  2. AUTHENTICATION                                                           │
│     ✅ JWT with short expiry (15 min)                                        │
│     ✅ Refresh tokens (7 days)                                               │
│     ✅ Biometric + PIN fallback                                              │
│     ✅ Secure storage (Keychain/Keystore)                                    │
│                                                                              │
│  3. DATA PROTECTION                                                          │
│     ✅ No PHI in local storage (only session ID)                             │
│     ✅ Memory-only sensitive data                                            │
│     ✅ Screenshot prevention (flag secure)                                   │
│     ✅ Auto-logout on background (configurable)                              │
│                                                                              │
│  4. API SECURITY                                                             │
│     ✅ Rate limiting (100 req/min)                                           │
│     ✅ Input validation (Pydantic schemas)                                   │
│     ✅ SQL injection prevention (parameterized queries)                      │
│     ✅ Audit logging                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Cost Estimates

### 8.1 Development Costs

| Resource | Duration | Cost (USD) |
|----------|----------|------------|
| React Native Developer | 8 weeks | $16,000 |
| Backend Developer | 2 weeks | $4,000 |
| UI/UX Designer | 2 weeks | $3,000 |
| QA Testing | 2 weeks | $2,000 |
| **Total** | | **$25,000** |

### 8.2 Infrastructure Costs (Monthly)

| Service | Provider | Cost (USD) |
|---------|----------|------------|
| Cloud Hosting (API) | AWS/GCP | $200-500 |
| Redis Cache | Redis Cloud | $50 |
| Push Notifications | Firebase | Free tier |
| SSL Certificate | Let's Encrypt | Free |
| Monitoring | DataDog/NewRelic | $100 |
| **Total/Month** | | **$350-650** |

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API latency on mobile | Medium | High | Implement caching, optimistic UI |
| Battery drain (GPS/BG) | Medium | Medium | Optimize background tasks |
| App store rejection | Low | High | Follow guidelines, medical disclaimer |
| PHI data leakage | Low | Critical | Encryption, secure storage, audit |
| Elderly UX challenges | High | Medium | Large fonts, voice input, simple flow |

---

## 10. Decision Matrix

| Criteria | React Native | PWA | WebView |
|----------|--------------|-----|---------|
| Development Speed | ⭐⭐⭐ (4-6 wks) | ⭐⭐⭐⭐⭐ (1-2 days) | ⭐⭐⭐⭐ (1 week) |
| Native Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Native Features (Push, Camera) | ⭐⭐⭐⭐⭐ | ⭐⭐ (Android: ⭐⭐⭐⭐) | ⭐⭐⭐ |
| Maintenance Cost | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| User Experience | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Offline Capability | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| App Store Presence | ✅ Yes | ❌ No | ✅ Yes |
| Code Reuse from Web | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Recommendation** | **✅ BEST for Production** | **✅ BEST for MVP** | Not recommended |

### Recommended Approach: **PWA First, Then Native**

```
Week 1: Deploy PWA MVP
├── Cloud backend deployment
├── Add PWA support to existing web app
├── Test with 5-10 users
└── Gather feedback

Weeks 2-7: Build React Native (if PWA validates demand)
├── Full native development
├── Push notifications, biometrics
├── App store submission
└── Production release
```

---

## 11. Questions for Discussion

### 11.1 Product Strategy

| # | Question | Context | Options |
|---|----------|---------|---------|
| 1 | **Who is the primary user?** | Determines UX complexity and feature priorities | A) Elderly residents directly<br>B) Family caregivers<br>C) Both with different interfaces<br>D) Healthcare professionals |
| 2 | **What is the primary use case?** | Guides feature prioritization | A) Emergency alerts only<br>B) Daily health check-ins<br>C) Full health advisory (current web parity)<br>D) Medication management focus |
| 3 | **Platform priority?** | Affects development timeline | A) iOS first (typical for healthcare)<br>B) Android first (broader market)<br>C) Both simultaneously<br>D) PWA as MVP, then native |
| 4 | **Offline capability requirement?** | Impacts architecture complexity | A) Full offline chat (cached responses)<br>B) Read-only offline (view history)<br>C) No offline (online only)<br>D) Smart sync (queue messages) |

### 11.2 Technical Architecture

| # | Question | Context | Options |
|---|----------|---------|---------|
| 5 | **Backend hosting strategy?** | Affects cost and maintenance | A) AWS (current expertise)<br>B) Google Cloud (Firebase integration)<br>C) Azure (enterprise compliance)<br>D) On-premise (data sovereignty) |
| 6 | **Session management approach?** | Security vs convenience tradeoff | A) Long-lived tokens (30 days)<br>B) Short tokens + refresh (15 min + 7 days)<br>C) Biometric-only (no passwords)<br>D) Magic links (email-based) |
| 7 | **Push notification strategy?** | Critical for health alerts | A) All risk alerts push immediately<br>B) Daily digest only<br>C) Critical only (emergency)<br>D) User-configurable per alert type |
| 8 | **Data synchronization model?** | For multi-device users | A) Single device only<br>B) Real-time sync across devices<br>C) Manual refresh only<br>D) Last-write-wins with conflict resolution |

### 11.3 Compliance & Security

| # | Question | Context | Options |
|---|----------|---------|---------|
| 9 | **Regulatory compliance scope?** | Legal requirements vary | A) HIPAA (US healthcare)<br>B) GDPR (EU privacy)<br>C) PDPO (Hong Kong)<br>D) Multiple jurisdictions |
| 10 | **Data residency requirements?** | Where is data stored/processed? | A) Same country as user<br>B) Specific region (e.g., EU, APAC)<br>C) Cloud provider's choice<br>D) Hybrid (metadata cloud, PHI on-prem) |
| 11 | **Audit and logging requirements?** | For compliance and debugging | A) Full request/response logging<br>B) Metadata only (no PHI)<br>C) Error logs only<br>D) HIPAA-compliant audit trail |
| 12 | **Medical disclaimer requirements?** | Legal protection | A) Static disclaimer on first launch<br>B) Per-message disclaimer<br>C) No advice without confirmation<br>D) Require caregiver approval for actions |

### 11.4 User Experience

| # | Question | Context | Options |
|---|----------|---------|---------|
| 13 | **Accessibility level target?** | Elderly users have specific needs | A) WCAG 2.1 AA (standard)<br>B) WCAG 2.1 AAA (enhanced)<br>C) Senior-specific guidelines<br>D) Voice-first interface |
| 14 | **Voice interaction requirement?** | Many elderly prefer voice | A) Required (primary input method)<br>B) Optional enhancement<br>C) Not needed (text only)<br>D) Voice output only (read responses) |
| 15 | **Emergency/SOS feature?** | Critical safety consideration | A) One-tap emergency button<br>B) Automatic emergency detection<br>C) Integration with 999/911<br>D) No emergency features |
| 16 | **Multi-language support?** | User demographics | A) English only<br>B) English + Chinese<br>C) English + Chinese + Other<br>D) Dynamic based on user location |

### 11.5 Business & Operations

| # | Question | Context | Options |
|---|----------|---------|---------|
| 17 | **App distribution method?** | Affects user acquisition | A) Public App Store / Play Store<br>B) Enterprise distribution only<br>C) TestFlight / Internal Testing<br>D) Sideload (enterprise devices) |
| 18 | **Monetization model?** | Sustainability of the app | A) Free (included in care package)<br>B) Subscription model<br>C) Freemium (basic free, premium paid)<br>D) One-time purchase |
| 19 | **Support and maintenance plan?** | Post-launch operations | A) 24/7 support for critical issues<br>B) Business hours only<br>C) Self-service with documentation<br>D) Dedicated account manager |
| 20 | **Success metrics?** | How do we measure success? | A) Download numbers<br>B) Daily active users (DAU)<br>C) Health outcomes (fall prevention)<br>D) User satisfaction scores |

### 11.6 Integration & Data

| # | Question | Context | Options |
|---|----------|---------|---------|
| 21 | **Integration with existing Beta 5.5?** | Data flow architecture | A) Direct database access (same DB)<br>B) API-only integration<br>C) Separate database with sync<br>D) Read-only replica for mobile |
| 22 | **Third-party health device integration?** | Apple Watch, Fitbit, etc. | A) Apple HealthKit (iOS)<br>B) Google Fit (Android)<br>C) Both platforms<br>D) No device integration |
| 23 | **LLM provider for mobile?** | Cost and performance tradeoff | A) Same as web (OpenAI/Claude)<br>B) Mobile-optimized model (faster)<br>C) Edge/on-device model (privacy)<br>D) Hybrid (cloud + edge) |
| 24 | **Data retention policy?** | Privacy and storage costs | A) 30 days conversation history<br>B) 90 days<br>C) 1 year<br>D) Indefinite (until user deletes) |

### 11.7 Rollout & Testing

| # | Question | Context | Options |
|---|----------|---------|---------|
| 25 | **Pilot/soft launch strategy?** | Risk mitigation | A) Internal team only<br>B) Single facility (10-20 users)<br>C) Beta program (100 users)<br>D) Full launch with monitoring |
| 26 | **Rollback plan?** | If issues arise | A) Force update mechanism<br>B) Feature flags (disable remotely)<br>C) Manual intervention required<br>D) No rollback (fix forward only) |
| 27 | **Training requirements?** | User onboarding | A) In-person training required<br>B) Video tutorials in-app<br>C) Contextual help/tooltips<br>D) Self-explanatory (no training) |

---

## 12. Next Steps

1. **Stakeholder Review:** Discuss this plan with product and clinical teams
2. **Technical Spike:** 2-day prototype with React Native + existing API
3. **Backend Preparation:** Deploy chatbot API to staging environment
4. **UX Design:** Mobile wireframes and elderly-friendly UI patterns
5. **Sprint Planning:** Break down Phase 1 into 2-week sprints

---

## Appendix A: API Contract (Mobile)

```typescript
// Base URL: https://api.careplatform.com/v1

// POST /chat
interface ChatRequest {
  elder_id: string;
  message: string;
  session_id?: string;
  device_id: string;  // NEW: For push notifications
  platform: 'ios' | 'android';  // NEW: Platform-specific handling
}

// Response remains same as web

// POST /auth/register-device
interface RegisterDeviceRequest {
  elder_id: string;
  device_token: string;
  platform: 'ios' | 'android';
}
```

## Appendix B: Push Notification Payload

**Policy:** No PHI in notification title/body/data. Show a generic alert and fetch details only after app unlock/auth.

```json
{
  "to": "device_fcm_token",
  "notification": {
    "title": "Health Alert",
    "body": "A new health alert is available. Open the app to review."
  },
  "data": {
    "type": "risk_alert",
    "alert_id": "alert_abc123",
    "risk_type": "fall",
    "severity": "high",
    "action": "open_risk_dashboard"
  }
}
```

## Appendix C: iPhone App Size Estimation

### Current Codebase Summary

| Component | Size | Lines of Code |
|-----------|------|---------------|
| **Backend (Python)** | ~269 KB | ~6,200 lines |
| **Frontend (TS/TSX)** | ~24 KB | ~700 lines |
| **Documentation** | ~165 KB | - |
| **Total Source** | ~458 KB | ~6,900 lines |
| **Full Directory** | **840 KB** | - |

### Deployment Scenarios

#### Option A: React Native App (Recommended)

This is a **client-server architecture** where the Python backend runs on a server, and the iPhone app is a thin client.

| Component | Estimated Size | Notes |
|-----------|----------------|-------|
| **React Native Runtime** | ~15-20 MB | Core framework + Hermes engine |
| **App JavaScript Bundle** | ~3-5 MB | Business logic, UI components, types |
| **Native Dependencies** | ~5-10 MB | Push notifications, secure storage, biometric auth |
| **Assets (icons, images)** | ~2-3 MB | App icons, UI graphics |
| **Offline Cache (optional)** | ~5-10 MB | Cached responses, knowledge base snippets |
| **Total Download Size** | **~25-40 MB** | From App Store |
| **Installed Size** | **~50-80 MB** | With caches and temp files |

**Breakdown of App Code:**
```
src/
├── Shared Types (from web)        ~5 KB
├── useChatbot Hook (adapted)      ~8 KB
├── UI Components (native)         ~50 KB
├── API Client + Auth              ~15 KB
├── Secure Storage Service         ~10 KB
├── Push Notification Handler      ~12 KB
└── Navigation + Screens           ~30 KB
                                 -----
Subtotal: ~130 KB source → ~3-5 MB compiled/bundled
```

#### Option B: Full Offline App (NOT RECOMMENDED)

If you tried to put the entire Python backend + LLM on-device:

| Component | Size | Feasibility |
|-----------|------|-------------|
| **Python Runtime (iOS)** | ~50-100 MB | PyMobil / BeeWare / Kivy |
| **Backend Code + Data** | ~5 MB | Knowledge base, guidelines |
| **LLM Model (GPT-2 size)** | ~500 MB - 2 GB | Minimum viable quality |
| **Sentence Embeddings** | ~100-500 MB | For RAG search |
| **Total** | **>1 GB** | ❌ Not practical |

**Verdict:** Not feasible for iPhone deployment. LLMs require cloud API.

#### Option C: Hybrid WebView Wrapper

Wrap existing web app in native container:

| Component | Size |
|-----------|------|
| **WebView Container** | ~2-3 MB |
| **Bundled Web Assets** | ~500 KB - 1 MB |
| **Native Bridge (minimal)** | ~1 MB |
| **Total** | **~5-8 MB** |

**Trade-off:** Smaller size but worse UX than React Native.

### Size Comparison by Approach

| Approach | Download Size | Installed Size | Pros/Cons |
|----------|---------------|----------------|-----------|
| **React Native (Recommended)** | 25-40 MB | 50-80 MB | Best UX, native features |
| **PWA (Add to Home Screen)** | ~1 MB | ~5 MB | No App Store, limited features |
| **WebView Wrapper** | 5-8 MB | 10-15 MB | Fastest dev, mediocre UX |
| **Full Offline** | >1 GB | >2 GB | ❌ Not viable |

### Comparable Healthcare Apps

| App | Size |
|-----|------|
| MyChart | ~45 MB |
| Apple Health | ~15 MB (system app) |
| Babylon Health | ~35 MB |
| Teladoc | ~40 MB |

**Target:** ~35 MB (well within acceptable range)

---

## Appendix D: Client-Server Architecture Clarification

### Data Flow (iPhone ↔ Cloud)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              IPHONE APP                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  User types │───►│  sendMessage │───►│  POST /api/chat (HTTPS)     │ │
│  │  "How do I  │    │  (useChatbot)│    │  {message, elder_id, ...}   │ │
│  │  sleep better│    └─────────────┘    └─────────────────────────────┘ │
│  └─────────────┘                          │                             │
└───────────────────────────────────────────┼─────────────────────────────┘
                                            ▼
                              ┌─────────────────────────────┐
                              │      CLOUD BACKEND          │
                              │  ┌─────────────────────┐    │
                              │  │  Python FastAPI     │    │
                              │  │  - Context Fusion   │    │
                              │  │  - Risk Calculation │    │
                              │  └─────────────────────┘    │
                              │              │              │
                              │              ▼              │
                              │  ┌─────────────────────┐    │
                              │  │  LLM API Call       │    │
                              │  │  (OpenAI/Claude)    │    │
                              │  │  - GPT-4 / Claude 3 │    │
                              │  └─────────────────────┘    │
                              │              │              │
                              │              ▼              │
                              │  ┌─────────────────────┐    │
                              │  │  Response Assembly  │    │
                              │  │  + Citations        │    │
                              │  └─────────────────────┘    │
                              └─────────────────────────────┘
                                            │
                                            ▼ HTTPS
┌───────────────────────────────────────────┼─────────────────────────────┐
│                              IPHONE APP   │                             │
│  ┌─────────────┐    ┌─────────────┐      │                             │
│  │  Display    │◄───│  Receive    │◄─────┘                             │
│  │  response   │    │  JSON resp  │                                    │
│  │  + risks    │    │             │                                    │
│  └─────────────┘    └─────────────┘                                    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                        │
│  │ Show alerts │  "High fall risk detected!"                            │
│  │  (if any)   │                                                        │
│  └─────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### Processing Location Summary

| Component | Location | Why |
|-----------|----------|-----|
| **User Interface** | iPhone (React Native) | Native feel, instant feedback |
| **State Management** | iPhone (Zustand/Context) | Fast UI updates |
| **Chat History** | iPhone (AsyncStorage) | Offline viewing |
| **Secure Auth Token** | iPhone (Keychain) | Biometric-protected |
| **Context Fusion** | ☁️ Cloud (Python) | Needs Beta 5.5 data access |
| **Risk Calculation** | ☁️ Cloud (Python) | CPU-intensive algorithms |
| **LLM (GPT-4/Claude)** | ☁️ Cloud (OpenAI API) | ~500MB-2GB model, won't fit on phone |
| **Knowledge Base** | ☁️ Cloud (clinical DB) | 200+ drug interactions, guidelines |
| **Citation Validator** | ☁️ Cloud (Python) | Needs full research corpus |

### API Call Example

```typescript
// iPhone sends this to cloud
const response = await fetch('https://api.yourdomain.com/api/chat', {
  method: 'POST',
  headers: { 
    'Content-Type': 'application/json',
    'Authorization': 'Bearer JWT_TOKEN_FROM_KEYCHAIN'
  },
  body: JSON.stringify({
    elder_id: "HK001_jessica",
    message: "How did I sleep last night?",
    session_id: "sess_abc123",
    include_medical_history: true,
    include_adl_data: true,
    include_icope_data: true,
    include_sleep_data: true,
    require_citations: true,
    device_id: "iphone_abc123",
    platform: "ios"
  }),
});

// Response from cloud
const data = await response.json();
// { message, risks, recommendations, citations, ... }
```

### Expected Latency

| Step | Typical Time | Notes |
|------|--------------|-------|
| iPhone → Cloud API | 50-200ms | Depends on network (5G/WiFi) |
| Context Fusion | 100-300ms | Database queries |
| LLM Generation (GPT-4) | 2-5 seconds | Slowest part |
| Response Assembly | 50ms | Citation validation |
| Cloud → iPhone | 50-200ms | |
| **Total** | **3-6 seconds** | Acceptable for health advice |

### Mobile Optimization Strategies

| Strategy | Implementation |
|----------|----------------|
| **Streaming Response** | Show LLM output word-by-word (like ChatGPT) |
| **Optimistic UI** | Show user message instantly, don't wait |
| **Loading Indicator** | Skeleton screens during LLM call |
| **Retry Logic** | Auto-retry on network failure |
| **Offline Queue** | Queue messages if no connection |

### Key Clarifications for Team Discussion

| Question | Answer |
|----------|--------|
| Is LLM on iPhone? | ❌ No - in cloud (OpenAI/Claude API) |
| Is processing on iPhone? | ❌ No - Python backend in cloud |
| What does iPhone app do? | UI, state management, secure storage, API calls |
| Data sent to cloud? | Text input + elder_id + preferences |
| Data received back? | AI response + risk alerts + citations |
| App size? | ~35 MB (thin client) |
| Comparable architecture? | ChatGPT, Claude, Babylon Health apps |

---

## Appendix E: Backend Infrastructure Requirements (Updated)

### Pre-Mobile Checklist

Before starting mobile development, the following backend infrastructure must be in place:

| Requirement | Status | Effort | Priority |
|-------------|--------|--------|----------|
| **Cloud Deployment** | ❌ Not started | 2-3 days | 🔴 Critical |
| **JWT Authentication** | ❌ Not started | 2 days | 🔴 Critical |
| **HTTPS/SSL** | ❌ Not started | 1 day | 🔴 Critical |
| **Redis Session Store** | ❌ Not started | 1-2 days | 🟡 High |
| **Push Notification Service** | ❌ Not started | 2 days | 🟡 High |
| **Rate Limiting** | ❌ Not started | 1 day | 🟡 High |
| **Beta 5.5 Integration** | ⚠️ Partial (mock data) | 3-5 days | 🔴 Critical |
| **Production LLM Keys** | ❌ Not started | 1 hour | 🟡 High |

### Risk Assessment Update

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Backend infrastructure delays mobile timeline | High | High | Start backend work NOW, parallel tracks |
| LLM latency unacceptable on mobile | Medium | Medium | Implement streaming, set expectations |
| Network connectivity issues for elderly | High | Medium | Offline queue, retry logic, clear error messages |
| Data usage concerns (elderly users) | Medium | Low | ~1-2 KB per message, minimal impact |

---

---

## Appendix F: Current System Quick Reference

### Running Services

| Service | URL | Description |
|---------|-----|-------------|
| Chatbot Demo | http://localhost:8000 | Interactive demo with 3 elders |
| Admin UI | http://localhost:8080 | Knowledge base editor |
| API Base | http://localhost:8000/api | All backend endpoints |

### Knowledge Base Status

| Category | Count | Admin Editable |
|----------|-------|----------------|
| Guidelines | 12 | ✅ Yes |
| Drugs | 10 | ✅ Yes |
| Research Papers | 23 | ✅ Yes |
| FAQs | 8 | ✅ Yes |
| **Total** | **53** | |

### API Endpoints (All Mobile-Ready)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | Send message, get advisory |
| `/api/chat/history/{session}` | GET | Conversation history |
| `/api/admin/guidelines` | GET/POST | CRUD guidelines |
| `/api/admin/drugs` | GET/POST | CRUD drugs |
| `/api/admin/research` | GET/POST | CRUD research papers |
| `/api/admin/faq` | GET/POST | CRUD FAQs |

### LLM Configuration

| Setting | Value |
|---------|-------|
| Provider | DeepSeek |
| Model | deepseek-chat |
| API Key | Configured |
| Status | ✅ Active |

---

*Document prepared for technical and product team discussion.*
*Last updated: 2026-02-10*
