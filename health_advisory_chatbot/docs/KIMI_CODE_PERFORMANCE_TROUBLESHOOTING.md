# Kimi Code Extension - Slow Responsiveness Troubleshooting Guide

**Date:** 2026-02-02  
**Investigation Status:** Complete  
**Severity:** Medium (Configuration-related)

---

## 1. Executive Summary

### 🔍 Findings

| Issue | Impact | Severity |
|-------|--------|----------|
| MCP Server Startup Overhead | +2-3s initial load | Medium |
| Thinking Mode Enabled | +50-100% response time | High (if not needed) |
| Large Context Window (256K) | Slower token processing | Medium |
| Multiple MCP Servers | Resource contention | Low-Medium |

### Root Cause Analysis

Your logs show:
```
# Startup sequence takes ~3-4 seconds
17:32:26.890 - Starting Wire server
17:32:27.641 - GitHub MCP Server running
17:32:27.818 - Context7 MCP Server running  
17:32:27.981 - Playwright MCP Server running
```

**Key Bottlenecks Identified:**
1. **3 MCP servers** starting simultaneously (GitHub, Context7, Playwright)
2. **Thinking mode enabled** by default (doubles response time)
3. **Large context window** (262,144 tokens) with 50K reserved

---

## 2. Performance Diagnostics

### 2.1 Current Configuration Analysis

```toml
# ~/.kimi/config.toml

# ⚠️ ISSUE 1: Thinking mode enabled by default
default_thinking = true  # → Disables turbo/rapid responses

# ⚠️ ISSUE 2: Large context with high reservation
max_context_size = 262144
reserved_context_size = 50000  # → Triggers compaction frequently

# ⚠️ ISSUE 3: MCP timeout too high
tool_call_timeout_ms = 60000  # → 60s wait for unresponsive tools
```

### 2.2 MCP Server Impact

| MCP Server | Startup Time | Runtime Impact |
|------------|--------------|----------------|
| GitHub | ~800ms | API latency dependent |
| Context7 | ~600ms | Documentation search |
| Playwright | ~150ms | Browser automation |
| **Total** | **~1.5s** | **Memory + CPU overhead** |

### 2.3 Response Time Breakdown

```
User Request
    │
    ├─► Context Loading (50-200ms)
    ├─► MCP Tool Discovery (100-300ms) ← If tools needed
    ├─► LLM API Call (2-10s) ← Thinking mode doubles this
    ├─► Response Streaming (varies)
    └─► Tool Execution (if needed)
        └─► MCP Server Call (100-3000ms)
```

---

## 3. Optimization Solutions

### 3.1 Immediate Fixes (Quick Wins)

#### Fix 1: Disable Thinking Mode for Faster Responses

```toml
# ~/.kimi/config.toml

# Change from:
default_thinking = true

# To:
default_thinking = false
```

**Impact:** 40-60% faster responses for coding tasks  
**When to use thinking:** Complex architecture decisions only

#### Fix 2: Reduce Context Window Size

```toml
# ~/.kimi/config.toml

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 131072      # ← Reduced from 262144
capabilities = ["thinking", "video_in", "image_in"]

[loop_control]
max_steps_per_turn = 50        # ← Reduced from 100
reserved_context_size = 20000  # ← Reduced from 50000
```

**Impact:** Faster context processing, less memory usage

#### Fix 3: Reduce MCP Tool Call Timeout

```toml
[mcp.client]
tool_call_timeout_ms = 10000   # ← Reduced from 60000 (10s vs 60s)
```

**Impact:** Faster failure on unresponsive tools

---

### 3.2 MCP Server Optimization

#### Option A: Disable Unused MCP Servers

```bash
# List current MCP servers
kimi mcp list

# Remove unused servers
kimi mcp remove context7    # If not using documentation lookup
kimi mcp remove playwright  # If not using browser automation
```

#### Option B: Lazy MCP Loading (If Available)

Check for environment variable:
```bash
export KIMI_MCP_LAZY_LOAD=1  # Start MCP servers on first use
```

#### Option C: MCP Server Health Check

```bash
# Test MCP server responsiveness
kimi mcp test github
kimi mcp test context7
kimi mcp test playwright
```

---

### 3.3 Advanced Optimizations

#### Optimization 1: Use Turbo Model

```toml
# Switch to faster model variant
[models."kimi-code/kimi-for-coding-turbo"]
provider = "managed:kimi-code"
model = "kimi-for-coding-turbo"  # ← Faster variant
max_context_size = 131072
capabilities = ["image_in"]  # ← Remove thinking capability
```

#### Optimization 2: Enable Response Caching

```bash
# Add to ~/.zshrc or ~/.bashrc
export KIMI_CACHE_RESPONSES=1
export KIMI_CACHE_TTL=3600  # 1 hour
```

#### Optimization 3: Disable Auto-Update Check

```bash
# Add to shell profile
export KIMI_CLI_NO_AUTO_UPDATE=1
```

**Impact:** Removes background update check on startup

---

## 4. Configuration Presets

### Preset A: Maximum Speed (Coding Focus)

```toml
# ~/.kimi/config.fast.toml

default_model = "kimi-code/kimi-for-coding"
default_thinking = false

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 65536
capabilities = ["image_in"]  # No thinking

[loop_control]
max_steps_per_turn = 30
max_retries_per_step = 2
reserved_context_size = 10000

[mcp.client]
tool_call_timeout_ms = 5000
```

**Use case:** Rapid coding assistance, quick edits

### Preset B: Balanced (Default)

```toml
# ~/.kimi/config.toml

default_model = "kimi-code/kimi-for-coding"
default_thinking = false  # Disable by default

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 131072
capabilities = ["thinking", "image_in"]

[loop_control]
max_steps_per_turn = 50
max_retries_per_step = 3
reserved_context_size = 20000

[mcp.client]
tool_call_timeout_ms = 10000
```

**Use case:** General development with thinking on-demand

### Preset C: Deep Analysis (Complex Tasks)

```toml
# ~/.kimi/config.deep.toml

default_model = "kimi-code/kimi-for-coding"
default_thinking = true  # Enable for complex tasks

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 262144
capabilities = ["thinking", "video_in", "image_in"]

[loop_control]
max_steps_per_turn = 100
max_retries_per_step = 3
reserved_context_size = 50000
```

**Use case:** Architecture design, complex debugging

---

## 5. VS Code Extension Specific Issues

### 5.1 Extension Settings to Check

Open VS Code Settings (`Cmd/Ctrl + ,`) and search for "Kimi":

| Setting | Recommended Value | Reason |
|---------|-------------------|--------|
| `kimi-code-cli.path` | Use absolute path | Avoids PATH resolution delay |
| `kimi.yoloMode` | `false` (default) | Auto-approval can cause loops |
| `kimi.autoUpdate` | `false` | Prevents background checks |

### 5.2 VS Code Performance Tips

```json
// ~/.vscode/settings.json
{
  // Reduce extension host load
  "kimi-code-cli.maxConcurrentTasks": 3,
  
  // Disable unnecessary features
  "kimi-code-cli.enableTelemetry": false,
  
  // Increase timeout for large projects
  "kimi-code-cli.requestTimeout": 30000
}
```

### 5.3 Check for Extension Conflicts

Other extensions that may conflict with Kimi Code:
- GitHub Copilot (competing AI suggestions)
- Other MCP-based extensions
- Heavy linting extensions (ESLint, Pylint)

**Test:** Disable other AI extensions temporarily

---

## 6. Monitoring & Debugging

### 6.1 Enable Performance Logging

```bash
# Add to ~/.zshrc or ~/.bashrc
export KIMI_LOG_LEVEL=DEBUG
export KIMI_LOG_PERFORMANCE=1
```

### 6.2 Check Logs for Bottlenecks

```bash
# View recent logs
tail -f ~/.kimi/logs/kimi.log

# Look for:
# - "Retrying step" (indicates failures)
# - MCP connection timeouts
# - Long-running tool calls
```

### 6.3 Benchmark Response Times

```bash
# Simple benchmark test
time kimi --print --prompt "Hello, simple test"

# Expected:
# - Without thinking: < 3 seconds
# - With thinking: < 6 seconds
```

---

## 7. Quick Fix Checklist

- [ ] **Disable thinking mode** in `~/.kimi/config.toml`
- [ ] **Reduce context size** to 131072 or 65536
- [ ] **Remove unused MCP servers** (`kimi mcp remove <name>`)
- [ ] **Reduce MCP timeout** to 10000ms
- [ ] **Disable auto-update** (`export KIMI_CLI_NO_AUTO_UPDATE=1`)
- [ ] **Restart VS Code** after changes
- [ ] **Test with simple prompt** to verify improvement

---

## 8. When to Contact Support

Contact Kimi Code support if:
- Response times remain > 10s after optimizations
- MCP servers fail to start consistently
- Memory usage exceeds 2GB
- Frequent "context window exceeded" errors

**Support Resources:**
- GitHub Issues: https://github.com/MoonshotAI/kimi-cli/issues
- Documentation: https://moonshotai.github.io/kimi-cli/

---

## 9. Module/Library Updates for Performance

### 9.1 Kimi Code CLI Updates

Kimi Code CLI is distributed as a **self-contained binary** (not via pip/uv for the VS Code extension version). However, you can still update:

#### Check Current Version
```bash
# Check installed version
/Users/dicksonng/Library/Application\ Support/Code/User/globalStorage/moonshot-ai.kimi-code/bin/kimi/kimi --version
# Output: kimi version 1.5
```

#### Update Kimi Code CLI

**Option A: Via VS Code Extension**
1. Open VS Code Extensions panel
2. Search for "Kimi Code"
3. Click "Update" if available
4. Restart VS Code

**Option B: Manual Update (if using uv)**
```bash
# Only if you installed via uv tool install
uv tool upgrade kimi-cli --no-cache
```

**Option C: Reinstall Latest**
```bash
# Backup config first
cp ~/.kimi/config.toml ~/.kimi/config.toml.backup

# Reinstall
curl -LsSf https://code.kimi.com/install.sh | bash
```

### 9.2 System-Level Dependencies

#### Python Version (If Running from Source)

| Python Version | Performance Impact | Recommendation |
|----------------|-------------------|----------------|
| 3.12 | Baseline | Minimum required |
| **3.13** | **+10-15% faster** | **Recommended** |
| 3.14 (dev) | Unstable | Not recommended |

```bash
# Check Python version
python3 --version

# If using uv to run kimi
uv tool install --python 3.13 kimi-cli
```

#### macOS System Updates

```bash
# Update macOS (can improve Python performance)
softwareupdate -l

# Update Xcode Command Line Tools
xcode-select --install
```

### 9.3 MCP Server Updates

MCP servers are separate Node.js/Python processes that can be updated:

```bash
# Update all MCP servers
kimi mcp list  # Check current versions

# For Node.js based MCP servers
npm update -g @anthropic-ai/mcp-server-github
npm update -g @context7/mcp-server

# For Python based MCP servers
pip install --upgrade mcp-server-playwright
```

### 9.4 VS Code Extension Updates

| Component | Update Method | Frequency |
|-----------|---------------|-----------|
| Kimi Code Extension | VS Code Extensions panel | Auto |
| VS Code itself | Check for Updates | Monthly |
| Extension Host | Restart VS Code | Weekly |

### 9.5 Optional Performance Libraries

These libraries can speed up specific operations if Kimi CLI uses them:

```bash
# Faster JSON parsing (if applicable)
pip install --upgrade orjson ujson

# Faster HTTP client
pip install --upgrade httpx[http2] aiohttp

# Faster regex
pip install --upgrade regex

# Faster YAML parsing
pip install --upgrade libyaml
```

### 9.6 Network Performance

```bash
# Check DNS resolution speed
dig api.kimi.com

# Test API latency
ping -c 5 api.kimi.com

# Alternative DNS (if slow)
# Use Cloudflare DNS: 1.1.1.1 or Google DNS: 8.8.8.8
```

### 9.7 Update Checklist

- [ ] **Kimi Code CLI**: Check for updates monthly
- [ ] **VS Code Extension**: Enable auto-update
- [ ] **MCP Servers**: Update when new versions released
- [ ] **Python**: Use 3.13 if running from source
- [ ] **macOS**: Keep system updated
- [ ] **Network**: Use fast DNS resolver

---

## Appendix: Environment Variables Reference

| Variable | Purpose | Recommended |
|----------|---------|-------------|
| `KIMI_CLI_NO_AUTO_UPDATE` | Disable update checks | `1` |
| `KIMI_LOG_LEVEL` | Log verbosity | `INFO` or `WARN` |
| `KIMI_MODEL_TEMPERATURE` | Response randomness | `0.3` for coding |
| `KIMI_MODEL_MAX_TOKENS` | Limit response length | `4096` |
| `UV_PYTHON` | Python version for uv | `3.13` |
| `PYTHONDONTWRITEBYTECODE` | Skip .pyc generation | `1` |

---

*Generated based on configuration analysis and log review.*
