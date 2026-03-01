# Plan: Android Standalone Demo App (Health Advisory Chatbot)

## 1. Objective
Build a fully self-contained Android APK for demonstration purposes. The app will showcase the **full health advisory lifecycle** (Chat + Risk Assessment + Clinical Evidence).
*   **Serverless**: No backend (Python/Node) server required.
*   **Smart**: Uses **Direct LLM Integration** (OpenAI API) from the phone for real, dynamic conversations.
*   **Requirement**: The phone MUST have an active internet connection.

## 2. Technical Stack
*   **Framework**: React Native (0.73+) with **Expo SDK 50** (Managed Workflow).
*   **AI Engine**: **OpenAI API** (Client-side integration).
*   **Language**: TypeScript (Strict Mode).
*   **Styling**: **NativeWind** (Tailwind CSS for React Native).
*   **Navigation**: **Expo Router**.

## 3. Implementation Blueprint (Step-by-Step)

### Phase 1: Project Initialization
**Goal**: Setup the foundation with correct dependencies.

1.  **Initialize**:
    ```bash
    npx create-expo-app@latest health-chat-mobile --template blank-typescript
    cd health-chat-mobile
    ```
2.  **Install Deps**:
    ```bash
    npx expo install expo-router react-native-safe-area-context react-native-screens expo-haptics
    npm install nativewind tailwindcss lucide-react-native clsx tailwind-merge date-fns
    ```
3.  **Config**: Setup NativeWind (babel/tailwind config).
4.  **App Config** (`app.json`):
    *   Set package name: `com.beta55.healthchat`
    *   Configure app icon and splash screen assets
    *   Verify permissions (INTERNET only)
    *   Set `expo-build-properties` for target SDK versions

### Phase 2: The "Smart Brain" (Direct LLM)
**Goal**: Replace the "Mock Brain" with a real LLM connection that "roleplays" based on our mock data.

#### Step 2.1: Mock Data (`src/services/mockData.ts`)
*   **Port Data**: Copy the `Margaret`, `Robert`, `Helen` profiles from Python to TypeScript.
*   **Purpose**: This data will be injected into the **System Prompt** so the LLM knows who it is talking about.

#### Step 2.2: Risk Engine (`src/services/RiskStratifier.ts`)
*   **Logic**: Keep the local TypeScript implementation of the Risk Engine (Fall Risk, Frailty).
*   **Why?**: We still want to show the "Risk Dashboard" calculated instantly on-device, independent of the LLM.
*   **Parity Check**: Verify TypeScript calculations match Python `risk_stratifier.py` output:
    ```bash
    # Generate test cases from Python backend
    python -c "from chatbot.predictive.risk_stratifier import ...; print(json.dumps(calc_risk(test_data)))"
    # Compare with TS implementation output
    ```

#### Step 2.3: LLM Client (`src/services/OpenAIClient.ts`)
*   **Action**: Create a service to call `https://api.openai.com/v1/chat/completions`.
*   **Prompt Engineering**:
    ```typescript
    const generateSystemPrompt = (elder: ElderProfile, risks: RiskAssessment) => `
      You are an expert geriatric health advisor.
      
      CURRENT PATIENT CONTEXT:
      Name: ${elder.name}
      Conditions: ${elder.conditions.join(", ")}
      Medications: ${elder.medications.map(m => m.name).join(", ")}
      Calculated Fall Risk: ${risks.overallScore} (Level: ${risks.level})
      
      INSTRUCTIONS:
      1. Answer questions based ONLY on the evidence.
      2. If high risk is detected, warn the user.
      3. Cite guidelines (AGS Beers 2023, WHO ICOPE) where relevant.
      4. Format response with Markdown.
    `;
    ```
*   **Security Note**: For this demo, we will hardcode a **Restricted API Key** (capped usage) into the app build config (`.env`).

#### Step 2.4: Offline Fallback (`src/services/OfflineFallback.ts`)
*   **Purpose**: Graceful degradation when network is unavailable.
*   **Implementation**:
    ```typescript
    // If LLM call fails or no network, return cached/scripted responses
    if (!navigator.onLine) {
      return getCannedResponse(query, elder);
    }
    ```
*   **Canned Responses**: Pre-defined answers for common questions per elder profile.

### Phase 3: Android UI Implementation
**Goal**: 1:1 Visual Parity with the Web Dashboard.

#### Step 3.1: Elder Selection Screen (`app/index.tsx`)
*   **UI**: 3 Cards. Selecting an elder updates the `SystemPrompt` context.

#### Step 3.2: Chat Interface (`app/chat.tsx`)
*   **Logic**:
    1.  User types message.
    2.  App appends functionality context (System Prompt + History + User Message).
    3.  `POST` to OpenAI.
    4.  Displays streaming text response.
*   **Visuals**: Large fonts, Sage Green bubbles.
*   **UX Enhancements**:
    *   Typing indicator / skeleton while LLM is thinking
    *   Haptic feedback on message send (`expo-haptics`)

#### Step 3.3: Risk Dashboard (`app/risks.tsx`)
*   **Visualization**: Native progress bars showing the locally-calculated risk scores.

#### Step 3.4: Error Handling UI
*   **Loading States**: Skeleton/shimmer during LLM response generation.
*   **Error Toasts**: User-friendly messages for rate limits, timeouts, network errors.
*   **Retry Mechanism**: "Try Again" button on failure with exponential backoff.
*   **Offline Banner**: Visual indicator when operating in offline fallback mode.

### Phase 4: Build & Deliver
1.  **Configure**: Finalize `app.json` with icons, splash, and metadata.
2.  **Build**: `eas build -p android --profile preview`.
3.  **Output**: `application-debug.apk`.

## 4. Execution Checklist
*   [ ] **Day 1**: Init Project, NativeWind, `app.json` config (icons/splash).
*   [ ] **Day 1**: Port Mock Data & Risk Engine (with parity verification).
*   [ ] **Day 2**: Implement `OpenAIClient.ts` with Prompt Engineering.
*   [ ] **Day 2**: Implement offline fallback & canned responses.
*   [ ] **Day 3**: Build UI screens (Elder Select, Chat, Risk Dashboard).
*   [ ] **Day 3**: Add error handling UI, loading states, haptic feedback.
*   [ ] **Day 3**: Test "Margaret" end-to-end scenarios.
*   [ ] **Day 4**: Cross-device testing (API 28, 31, 34+).
*   [ ] **Day 4**: Test edge cases (unknown conditions, empty profiles, network loss).
*   [ ] **Day 4**: Final polish, APK generation, and distribution.
