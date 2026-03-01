"""
Demo Server for Health Advisory Chatbot

A simple HTTP server that demonstrates the chatbot functionality
with mock data. No Beta 5.5 integration required.
"""

import json
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import uuid

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key, value)

# Set DeepSeek configuration if not already set
if not os.getenv('LLM_PROVIDER'):
    os.environ['LLM_PROVIDER'] = 'deepseek'
    os.environ['DEEPSEEK_API_KEY'] = 'sk-b79bc5b917e147eca31e92bdcc456955'
    os.environ['DEEPSEEK_API_BASE'] = 'https://api.deepseek.com'
    os.environ['LLM_MODEL'] = 'deepseek-chat'
    os.environ['LLM_TEMPERATURE'] = '0.3'
    os.environ['LLM_MAX_TOKENS'] = '2000'

# Add backend to path
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from mock_data import get_mock_elder, list_mock_elders, MOCK_ELDERS

# Import chatbot components
from chatbot.core import get_advisory_engine
from models.schemas import ChatRequest


class DemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for demo dashboard."""
    
    advisory_engine = get_advisory_engine()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/":
            self._serve_dashboard()
        elif path == "/api/demo/elders":
            self._serve_elders_list()
        elif path.startswith("/api/demo/elders/"):
            elder_id = path.split("/")[-1]
            self._serve_elder_detail(elder_id)
        elif path.startswith("/api/demo/context/"):
            elder_id = path.split("/")[-1]
            self._serve_health_context(elder_id)
        else:
            self._serve_404()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == "/api/chat":
            self._handle_chat()
        else:
            self._serve_404()
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html = self._get_dashboard_html()
        self._send_response(200, "text/html", html)
    
    def _serve_elders_list(self):
        """Serve list of mock elders."""
        elders = list_mock_elders()
        self._send_json({"elders": elders})
    
    def _serve_elder_detail(self, elder_id: str):
        """Serve detailed elder data."""
        elder = get_mock_elder(elder_id)
        if not elder:
            self._serve_404()
            return
        
        self._send_json({
            "elder_id": elder_id,
            "profile": {
                "elder_id": elder["profile"].elder_id,
                "full_name": elder["profile"].full_name,
                "age": elder["profile"].age,
                "gender": elder["profile"].gender,
                "chronic_conditions": elder["profile"].chronic_conditions,
                "medications": [
                    {"name": m.name, "dosage": m.dosage, "frequency": m.frequency}
                    for m in elder["profile"].medications
                ],
                "mobility_status": elder["profile"].mobility_status,
                "cognitive_status": elder["profile"].cognitive_status,
            },
            "adl_summary": {
                "days_analyzed": elder["adl"].days_analyzed,
                "nighttime_bathroom_visits": elder["adl"].nighttime_bathroom_visits,
                "anomaly_count": elder["adl"].anomaly_count,
                "average_transition_time": elder["adl"].average_transition_time,
            },
            "icope": {
                "overall_score": elder["icope"].overall_score,
                "cognitive_capacity": elder["icope"].cognitive_capacity,
                "locomotor_capacity": elder["icope"].locomotor_capacity,
                "psychological_capacity": elder["icope"].psychological_capacity,
                "sensory_capacity": elder["icope"].sensory_capacity,
                "vitality_nutrition": elder["icope"].vitality_nutrition,
                "domains_at_risk": elder["icope"].domains_at_risk,
            },
            "sleep": {
                "total_duration_hours": elder["sleep"].total_duration_hours,
                "sleep_efficiency": elder["sleep"].sleep_efficiency,
                "quality_score": elder["sleep"].quality_score,
                "awakenings_count": elder["sleep"].awakenings_count,
                "sleep_apnea_risk": elder["sleep"].sleep_apnea_risk,
            },
        })
    
    def _serve_health_context(self, elder_id: str):
        """Serve computed health context with risks."""
        elder = get_mock_elder(elder_id)
        if not elder:
            self._serve_404()
            return
        
        # Build context using real engine
        from chatbot.core.context_fusion import get_context_fusion_engine
        
        engine = get_context_fusion_engine()
        context = engine.build_health_context(
            elder_id=elder_id,
            medical_profile=elder["profile"],
            adl_summary=elder["adl"],
            icope_assessment=elder["icope"],
            sleep_summary=elder["sleep"],
        )
        
        # Format response
        response = {
            "elder_id": context.elder_id,
            "data_completeness": context.data_completeness,
            "context_summary": context.context_summary,
            "risk_assessment": {
                "overall_score": context.risk_assessment.overall_risk_score if context.risk_assessment else None,
                "overall_level": context.risk_assessment.overall_risk_level.value if context.risk_assessment else None,
                "fall_risk": context.risk_assessment.fall_risk if context.risk_assessment else None,
                "cognitive_risk": context.risk_assessment.cognitive_decline_risk if context.risk_assessment else None,
                "sleep_risk": context.risk_assessment.sleep_disorder_risk if context.risk_assessment else None,
                "medication_risk": context.risk_assessment.medication_risk if context.risk_assessment else None,
                "critical_alerts": context.risk_assessment.critical_alerts if context.risk_assessment else [],
                "top_risk_factors": [
                    {
                        "name": f.factor_name,
                        "category": f.category,
                        "severity": f.severity.value,
                        "score": f.weighted_score,
                    }
                    for f in (context.risk_assessment.top_risk_factors if context.risk_assessment else [])
                ],
            },
            "trajectories": [
                {
                    "domain": t.domain,
                    "current_status": t.current_status,
                    "predicted_status": t.predicted_status,
                    "confidence": t.confidence,
                }
                for t in context.trajectories
            ] if context.trajectories else [],
        }
        
        self._send_json(response)
    
    def _handle_chat(self):
        """Handle chat message."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body.decode('utf-8'))
        
        elder_id = data.get('elder_id')
        message = data.get('message')
        session_id = data.get('session_id')
        
        if not elder_id or not message:
            self._send_json({"error": "Missing elder_id or message"}, 400)
            return
        
        elder = get_mock_elder(elder_id)
        if not elder:
            self._send_json({"error": "Elder not found"}, 404)
            return
        
        # Create request
        request = ChatRequest(
            elder_id=elder_id,
            message=message,
            session_id=session_id,
        )
        
        # Process through advisory engine
        try:
            response = self.advisory_engine.process_chat_request(
                request=request,
                medical_profile=elder["profile"],
                adl_summary=elder["adl"],
                icope_assessment=elder["icope"],
                sleep_summary=elder["sleep"],
            )
            
            # Format response
            result = {
                "success": True,
                "session_id": response.session_id,
                "message": {
                    "role": response.message.role.value,
                    "content": response.message.content,
                    "timestamp": response.message.timestamp.isoformat(),
                    "evidence_summary": response.message.evidence_summary,
                    "risk_alerts": response.message.risk_alerts,
                },
                "risks": {
                    "overall_score": response.current_risks.overall_risk_score if response.current_risks else None,
                    "overall_level": response.current_risks.overall_risk_level.value if response.current_risks else None,
                    "fall_risk": response.current_risks.fall_risk if response.current_risks else None,
                    "cognitive_risk": response.current_risks.cognitive_decline_risk if response.current_risks else None,
                    "sleep_risk": response.current_risks.sleep_disorder_risk if response.current_risks else None,
                    "medication_risk": response.current_risks.medication_risk if response.current_risks else None,
                    "critical_alerts": response.current_risks.critical_alerts if response.current_risks else [],
                    "top_risk_factors": [
                        {
                            "name": f.factor_name,
                            "category": f.category,
                            "severity": f.severity.value,
                            "score": f.weighted_score,
                        }
                        for f in (response.current_risks.top_risk_factors if response.current_risks else [])
                    ],
                } if response.current_risks else None,
                "recommendations": [
                    {
                        "title": r.title,
                        "category": r.category,
                        "priority": r.priority,
                        "description": r.description[:100] + "..." if len(r.description) > 100 else r.description,
                        "evidence_level": r.evidence_level.value,
                    }
                    for r in response.recommendations
                ],
                "response_time_ms": response.response_time_ms,
            }
            
            self._send_json(result)
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Error in chat handler: {error_msg}")
            self._send_json({"success": False, "error": str(e)}, 500)
    
    def _serve_404(self):
        """Serve 404 error."""
        self._send_json({"error": "Not found"}, 404)
    
    def _send_response(self, status: int, content_type: str, content: str):
        """Send HTTP response."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self._send_response(status, "application/json", json.dumps(data, indent=2))
    
    def _get_dashboard_html(self) -> str:
        """Generate dashboard HTML."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Health Advisory Chatbot - Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            text-align: center;
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .header p { font-size: 1.2rem; opacity: 0.9; }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .grid {
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 2rem;
            margin-top: 2rem;
        }
        @media (max-width: 1200px) {
            .grid { grid-template-columns: 1fr; }
        }
        .panel {
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .panel-header {
            background: #f8f9fa;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #e9ecef;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .panel-body { padding: 1.5rem; }
        .elder-card {
            padding: 1rem;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .elder-card:hover { border-color: #667eea; background: #f8f9ff; }
        .elder-card.active { border-color: #667eea; background: #f0f2ff; }
        .elder-name { font-weight: 600; font-size: 1.1rem; }
        .elder-info { color: #6c757d; font-size: 0.9rem; margin-top: 0.25rem; }
        .elder-scenario {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        .scenario-high { background: #fee; color: #c33; }
        .scenario-moderate { background: #ffeaa7; color: #856404; }
        .scenario-low { background: #d4edda; color: #155724; }
        .chat-window {
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 1rem;
            max-width: 80%;
        }
        .message.user { margin-left: auto; }
        .message-bubble {
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.95rem;
        }
        .message.user .message-bubble {
            background: #667eea;
            color: white;
        }
        .message.assistant .message-bubble {
            background: white;
            border: 1px solid #e9ecef;
        }
        .chat-input {
            display: flex;
            gap: 0.5rem;
            padding: 1rem;
            border-top: 1px solid #e9ecef;
            background: white;
        }
        .chat-input input {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
        }
        .chat-input button {
            padding: 0.75rem 1.5rem;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
        }
        .chat-input button:hover { background: #5568d3; }
        .chat-input button:disabled { opacity: 0.5; cursor: not-allowed; }
        .risk-panel { }
        .risk-score {
            text-align: center;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .risk-score-value {
            font-size: 3rem;
            font-weight: 700;
        }
        .risk-score-label {
            font-size: 0.9rem;
            color: #6c757d;
            margin-top: 0.25rem;
        }
        .risk-high { color: #dc3545; }
        .risk-moderate { color: #ffc107; }
        .risk-low { color: #28a745; }
        .risk-factors { margin-top: 1rem; }
        .risk-factor {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid #e9ecef;
            font-size: 0.9rem;
        }
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            color: #6c757d;
        }
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .health-metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        .metric {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .metric-value { font-size: 1.5rem; font-weight: 700; }
        .metric-label { font-size: 0.8rem; color: #6c757d; }
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: #6c757d;
        }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .suggestion-chip {
            padding: 0.5rem 1rem;
            background: #e9ecef;
            border-radius: 16px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .suggestion-chip:hover { background: #667eea; color: white; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Health Advisory Chatbot</h1>
        <p>Demo Dashboard - Evidence-Based Health Guidance for Elderly Care</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <!-- Elder Selection Panel -->
            <div class="panel">
                <div class="panel-header">👥 Select Elder</div>
                <div class="panel-body" id="elderList">
                    <div class="loading"><div class="spinner"></div>Loading...</div>
                </div>
            </div>
            
            <!-- Chat Panel -->
            <div class="panel">
                <div class="panel-header">💬 Health Advisor Chat</div>
                <div class="chat-window" id="chatWindow">
                    <div class="chat-messages" id="chatMessages">
                        <div class="empty-state">
                            <p>Select an elder to start the conversation</p>
                        </div>
                    </div>
                    <div class="chat-input">
                        <input type="text" id="messageInput" placeholder="Ask a health question..." disabled>
                        <button id="sendButton" disabled>Send</button>
                    </div>
                </div>
            </div>
            
            <!-- Risk Panel -->
            <div class="panel">
                <div class="panel-header">📊 Health Assessment</div>
                <div class="panel-body" id="riskPanel">
                    <div class="empty-state">
                        <p>Select an elder to view health assessment</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // State
        let currentElder = null;
        let sessionId = null;
        let messages = [];
        
        // Load elders on startup
        async function loadElders() {
            try {
                const response = await fetch('/api/demo/elders');
                const data = await response.json();
                renderElderList(data.elders);
            } catch (error) {
                console.error('Failed to load elders:', error);
            }
        }
        
        // Render elder list
        function renderElderList(elders) {
            const container = document.getElementById('elderList');
            container.innerHTML = elders.map(elder => `
                <div class="elder-card ${currentElder === elder.id ? 'active' : ''}" 
                     onclick="selectElder('${elder.id}')">
                    <div class="elder-name">${elder.name}</div>
                    <div class="elder-info">Age ${elder.age} • ${elder.scenario.replace(/_/g, ' ')}</div>
                    <div class="elder-scenario scenario-${getScenarioClass(elder.scenario)}">
                        ${elder.scenario.replace(/_/g, ' ')}
                    </div>
                </div>
            `).join('');
        }
        
        function getScenarioClass(scenario) {
            if (scenario.includes('high')) return 'high';
            if (scenario.includes('cognitive')) return 'moderate';
            return 'low';
        }
        
        // Select elder
        async function selectElder(elderId) {
            currentElder = elderId;
            messages = [];
            sessionId = null;
            
            // Update UI
            await loadElders();
            
            // Load health context
            await loadHealthContext(elderId);
            
            // Enable chat
            document.getElementById('messageInput').disabled = false;
            document.getElementById('sendButton').disabled = false;
            
            // Add welcome message
            addMessage('assistant', `Hello! I'm your health advisor. I can see you're working with ${elderId}. How can I help today?`);
            
            // Show suggestions
            showSuggestions();
        }
        
        // Load health context and risks
        async function loadHealthContext(elderId) {
            try {
                const response = await fetch(`/api/demo/context/${elderId}`);
                const data = await response.json();
                renderRiskPanel(data);
            } catch (error) {
                console.error('Failed to load health context:', error);
            }
        }
        
        // Render risk panel
        function renderRiskPanel(data) {
            const panel = document.getElementById('riskPanel');
            const risk = data.risk_assessment;
            
            if (!risk || !risk.overall_score) {
                panel.innerHTML = '<div class="empty-state"><p>No risk data available</p></div>';
                return;
            }
            
            const riskClass = risk.overall_level === 'high' || risk.overall_level === 'critical' ? 'risk-high' :
                            risk.overall_level === 'moderate' ? 'risk-moderate' : 'risk-low';
            
            panel.innerHTML = `
                <div class="risk-score">
                    <div class="risk-score-value ${riskClass}">${Math.round(risk.overall_score)}</div>
                    <div class="risk-score-label">Overall Risk Score</div>
                    <div style="margin-top: 0.5rem; text-transform: uppercase; font-weight: 600;" class="${riskClass}">
                        ${risk.overall_level}
                    </div>
                </div>
                
                <div class="health-metrics">
                    <div class="metric">
                        <div class="metric-value">${Math.round(risk.fall_risk || 0)}</div>
                        <div class="metric-label">Fall Risk</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${Math.round(risk.cognitive_risk || 0)}</div>
                        <div class="metric-label">Cognitive</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${Math.round(risk.sleep_risk || 0)}</div>
                        <div class="metric-label">Sleep</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${Math.round(risk.medication_risk || 0)}</div>
                        <div class="metric-label">Medication</div>
                    </div>
                </div>
                
                ${risk.critical_alerts && risk.critical_alerts.length > 0 ? `
                    <div style="margin-top: 1rem; padding: 1rem; background: #fee; border-radius: 8px;">
                        <div style="font-weight: 600; color: #c33; margin-bottom: 0.5rem;">⚠️ Critical Alerts</div>
                        ${risk.critical_alerts.map(alert => `<div style="color: #666; font-size: 0.9rem;">• ${alert}</div>`).join('')}
                    </div>
                ` : ''}
                
                ${risk.top_risk_factors && risk.top_risk_factors.length > 0 ? `
                    <div class="risk-factors">
                        <div style="font-weight: 600; margin-bottom: 0.5rem;">Top Risk Factors</div>
                        ${risk.top_risk_factors.map(factor => `
                            <div class="risk-factor">
                                <span>${factor.name}</span>
                                <span style="font-weight: 600; ${factor.severity === 'high' ? 'color: #dc3545;' : ''}">${Math.round(factor.score)}</span>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
            `;
        }
        
        // Show suggestion chips
        function showSuggestions() {
            const suggestions = [
                'How did I sleep last night?',
                'What are my fall risks?',
                'Should I be worried about my medications?',
                'How can I improve my mobility?',
                'What does my ICOPE score mean?'
            ];
            
            const messagesDiv = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.innerHTML = `
                <div class="message-bubble">
                    <div style="margin-bottom: 0.5rem;">Here are some questions you can ask:</div>
                    <div class="suggestions">
                        ${suggestions.map(s => `<span class="suggestion-chip" onclick="sendSuggestion('${s}')">${s}</span>`).join('')}
                    </div>
                </div>
            `;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Send suggestion
        function sendSuggestion(text) {
            document.getElementById('messageInput').value = text;
            sendMessage();
        }
        
        // Add message to chat
        function addMessage(role, content) {
            messages.push({ role, content });
            const messagesDiv = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<div class="message-bubble">${content.replace(/\\n/g, '<br>')}</div>`;
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const button = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message || !currentElder) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Show loading
            button.disabled = true;
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message assistant';
            loadingDiv.innerHTML = '<div class="message-bubble"><div class="loading"><div class="spinner"></div>Analyzing...</div></div>';
            document.getElementById('chatMessages').appendChild(loadingDiv);
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        elder_id: currentElder,
                        message: message,
                        session_id: sessionId
                    })
                });
                
                const data = await response.json();
                
                // Remove loading
                loadingDiv.remove();
                
                // Check if response is OK and has content
                if (response.ok && data.message && data.message.content) {
                    sessionId = data.session_id;
                    addMessage('assistant', data.message.content);
                    
                    // Update risks if available
                    if (data.risks) {
                        renderRiskPanel({ risk_assessment: data.risks });
                    }
                } else if (data.error) {
                    console.error('Server error:', data.error);
                    addMessage('assistant', 'Sorry, I encountered an error: ' + data.error);
                } else {
                    addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
                }
            } catch (error) {
                loadingDiv.remove();
                console.error('Network error:', error);
                addMessage('assistant', 'Sorry, I encountered a network error. Please try again.');
            } finally {
                button.disabled = false;
            }
        }
        
        // Event listeners
        document.getElementById('sendButton').addEventListener('click', sendMessage);
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Initialize
        loadElders();
    </script>
</body>
</html>
'''


def run_server(port=8000):
    """Run the demo server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, DemoHandler)
    print(f"\n{'='*60}")
    print(f"🚀 Health Advisory Chatbot Demo Server")
    print(f"{'='*60}")
    print(f"\nServer running at: http://localhost:{port}")
    print(f"\nMock Elders:")
    for elder in list_mock_elders():
        print(f"  • {elder['name']} ({elder['id']}) - {elder['description']}")
    print(f"\nPress Ctrl+C to stop")
    print(f"{'='*60}\n")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Health Advisory Chatbot Demo Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run server on')
    args = parser.parse_args()
    
    run_server(args.port)
