"""Sawyer Consumer Client — the user-facing inference gateway.

This is what the person with the 8GB laptop runs. It provides:

1. An OpenAI-compatible API at /v1/chat/completions so any existing tool
   (curl, OpenAI SDK, Ollama clients) can use it without modification.

2. A web chat UI at / so users can just open localhost:8000 and start chatting.

3. A local inference fallback — if no Sawyer network is available, it can
   fall back to a local model (llama.cpp, Ollama, etc.) for basic inference.

The whole point: cheaper inference than OpenAI/Anthropic, using distributed
MoE when available, local fallback when not.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from sawyer.config import SawyerConfig

logger = logging.getLogger(__name__)


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class InferenceResult:
    """Result from an inference request."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str = "stop"
    cost_tokens: int = 0  # tokens deducted from account


# ── Local Inference Fallback ───────────────────────────────────────


class LocalInference:
    """Fallback to local inference when no Sawyer network is available.

    Tries, in order:
    1. llama.cpp server (if running locally)
    2. Ollama (if installed and serving)
    3. Returns an error suggesting the user start a model
    """

    def __init__(self, config: SawyerConfig | None = None) -> None:
        self.config = config or SawyerConfig()
        self._ollama_url = "http://localhost:11434"
        self._llama_url = "http://localhost:8444"

    def infer(
        self,
        prompt: str,
        model: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> InferenceResult:
        """Try local inference backends in order."""
        # Try llama.cpp server first
        result = self._try_llama(prompt, model, max_tokens, temperature, top_p)
        if result:
            return result

        # Try Ollama
        result = self._try_ollama(prompt, model, max_tokens, temperature)
        if result:
            return result

        # No local backend available
        raise RuntimeError(
            "No inference backend available. Start a model with:\n"
            "  sawyer serve --offline --model mixtral-8x7b\n"
            "  or install Ollama: https://ollama.com\n"
            "  or connect to the Sawyer network: sawyer serve"
        )

    def _try_llama(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> InferenceResult | None:
        """Try llama.cpp server at localhost:8444."""
        try:
            import httpx

            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{self._llama_url}/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data.get("choices", [{}])[0]
                    usage = data.get("usage", {})
                    return InferenceResult(
                        text=choice.get("message", {}).get("content", ""),
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        latency_ms=data.get("latency_ms", 0),
                        model=data.get("model", model),
                        finish_reason=choice.get("finish_reason", "stop"),
                        cost_tokens=usage.get("completion_tokens", 0),
                    )
        except Exception:
            pass
        return None

    def _try_ollama(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> InferenceResult | None:
        """Try Ollama at localhost:11434."""
        try:
            import httpx

            # Map Sawyer model names to Ollama model names
            ollama_model = model or "llama3"
            model_map = {
                "mixtral-8x7b": "mixtral",
                "deepseek-v2-lite": "deepseek-v2",
            }
            ollama_model = model_map.get(ollama_model, ollama_model)

            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{self._ollama_url}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature,
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return InferenceResult(
                        text=data.get("message", {}).get("content", ""),
                        input_tokens=data.get("prompt_eval_count", 0),
                        output_tokens=data.get("eval_count", 0),
                        latency_ms=data.get("total_duration", 0) / 1_000_000,
                        model=data.get("model", model),
                        finish_reason="stop" if data.get("done") else "length",
                        cost_tokens=data.get("eval_count", 0),
                    )
        except Exception:
            pass
        return None

    def is_available(self) -> dict[str, bool]:
        """Check which local backends are available."""
        available = {}
        import httpx

        with httpx.Client(timeout=2) as client:
            try:
                resp = client.get(f"{self._llama_url}/health")
                available["llama_cpp"] = resp.status_code == 200
            except Exception:
                available["llama_cpp"] = False

            try:
                resp = client.get(self._ollama_url)
                available["ollama"] = resp.status_code == 200
            except Exception:
                available["ollama"] = False

        return available


# ── Chat UI ─────────────────────────────────────────────────────────


CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sawyer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --text: #e4e4ef;
    --text-dim: #8888a0;
    --accent: #12c7ef;
    --accent-dim: #0e9fc3;
    --user-bg: #1a2a35;
    --assistant-bg: #1a1a25;
    --error: #ef4444;
  }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  header h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.5px;
  }
  .header-info {
    font-size: 12px;
    color: var(--text-dim);
    display: flex;
    gap: 16px;
    align-items: center;
  }
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
  }
  .status-dot.online { background: #22c55e; }
  .status-dot.offline { background: var(--error); }
  .status-dot.local { background: #f59e0b; }
  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .message {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    font-size: 14px;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .message.user {
    align-self: flex-end;
    background: var(--user-bg);
    border: 1px solid var(--accent-dim);
  }
  .message.assistant {
    align-self: flex-start;
    background: var(--assistant-bg);
    border: 1px solid var(--border);
  }
  .message.error {
    background: #2a1515;
    border: 1px solid var(--error);
    color: #fca5a5;
  }
  .message.system {
    align-self: center;
    background: transparent;
    color: var(--text-dim);
    font-size: 12px;
    border: none;
    padding: 4px;
  }
  .meta {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 6px;
  }
  #input-area {
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 16px 24px;
    flex-shrink: 0;
  }
  .input-row {
    display: flex;
    gap: 8px;
    align-items: flex-end;
  }
  textarea {
    flex: 1;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 14px;
    padding: 12px;
    resize: none;
    font-family: inherit;
    min-height: 44px;
    max-height: 200px;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus {
    border-color: var(--accent);
  }
  button {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s;
    white-space: nowrap;
  }
  button:hover { opacity: 0.9; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  .model-select {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    padding: 10px 12px;
    font-size: 13px;
    outline: none;
  }
  .model-select:focus {
    border-color: var(--accent);
  }
  .welcome {
    text-align: center;
    padding: 60px 24px;
    color: var(--text-dim);
  }
  .welcome h2 {
    color: var(--text);
    font-size: 24px;
    margin-bottom: 8px;
    font-weight: 600;
  }
  .welcome p {
    font-size: 14px;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
  }
  .token-info {
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 8px;
  }
</style>
</head>
<body>
<header>
  <h1>Sawyer</h1>
  <div class="header-info">
    <span>
      <span class="status-dot local" id="status-dot"></span>
      <span id="status-text">Checking...</span>
    </span>
    <span id="model-display">local</span>
  </div>
</header>

<div id="messages">
  <div class="welcome">
    <h2>Distributed MoE Inference</h2>
    <p>Send a prompt and get cheaper inference than your provider. The Sawyer network splits the load across nodes so you only pay for what you use.</p>
  </div>
</div>

<div id="input-area">
  <div class="token-info" id="token-info"></div>
  <div class="input-row">
    <select class="model-select" id="model-select">
      <option value="">auto</option>
      <option value="mixtral-8x7b">Mixtral 8x7B</option>
      <option value="deepseek-v2-lite">DeepSeek V2 Lite</option>
      <option value="llama3">Llama 3</option>
    </select>
    <textarea id="prompt" rows="1" placeholder="Type your prompt..." autofocus></textarea>
    <button id="send" onclick="sendMessage()">Send</button>
  </div>
</div>

<script>
const messagesDiv = document.getElementById('messages');
const promptInput = document.getElementById('prompt');
const sendBtn = document.getElementById('send');
const modelSelect = document.getElementById('model-select');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const modelDisplay = document.getElementById('model-display');
const tokenInfo = document.getElementById('token-info');

let conversationHistory = [];
let isLoading = false;

// Auto-resize textarea
promptInput.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// Enter to send (Shift+Enter for newline)
promptInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Check backend status
async function checkStatus() {
  try {
    const resp = await fetch('/v1/models');
    if (resp.ok) {
      const data = await resp.json();
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Connected';
      if (data.data && data.data.length > 0) {
        modelDisplay.textContent = data.data.map(m => m.id).join(', ');
      }
    }
  } catch (e) {
    try {
      const resp = await fetch('/health');
      if (resp.ok) {
        statusDot.className = 'status-dot local';
        statusText.textContent = 'Local';
        modelDisplay.textContent = 'local';
      }
    } catch (e2) {
      statusDot.className = 'status-dot offline';
      statusText.textContent = 'Offline';
    }
  }
}

checkStatus();
setInterval(checkStatus, 30000);

// Fetch token balance
async function updateTokenInfo() {
  try {
    const resp = await fetch('/v1/balance');
    if (resp.ok) {
      const data = await resp.json();
      tokenInfo.textContent = 'Balance: ' + (data.balance || 0) + ' tokens';
    }
  } catch (e) {
    // Token accounting not available
  }
}
updateTokenInfo();

function addMessage(role, content, meta) {
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.textContent = content;
  if (meta) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'meta';
    metaDiv.textContent = meta;
    div.appendChild(metaDiv);
  }
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return div;
}

async function sendMessage() {
  const prompt = promptInput.value.trim();
  if (!prompt || isLoading) return;

  isLoading = true;
  sendBtn.disabled = true;
  promptInput.value = '';
  promptInput.style.height = 'auto';

  addMessage('user', prompt);

  const assistantDiv = document.createElement('div');
  assistantDiv.className = 'message assistant';
  assistantDiv.textContent = '';
  messagesDiv.appendChild(assistantDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;

  conversationHistory.push({role: 'user', content: prompt});

  const startTime = performance.now();

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: modelSelect.value || 'sawyer',
        messages: conversationHistory,
        max_tokens: 1024,
        temperature: 0.7,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error?.message || data.detail || 'Request failed');
    }

    const choice = data.choices?.[0];
    const content = choice?.message?.content || '';
    const usage = data.usage || {};

    conversationHistory.push({role: 'assistant', content: content});

    assistantDiv.textContent = content;

    const elapsed = Math.round(performance.now() - startTime);
    const meta = [
      usage.prompt_tokens ? 'in:' + usage.prompt_tokens : '',
      usage.completion_tokens ? 'out:' + usage.completion_tokens : '',
      elapsed + 'ms',
    ].filter(Boolean).join(' | ');
    if (meta) {
      const metaDiv = document.createElement('div');
      metaDiv.className = 'meta';
      metaDiv.textContent = meta;
      assistantDiv.appendChild(metaDiv);
    }

    updateTokenInfo();

  } catch (err) {
    assistantDiv.className = 'message error';
    assistantDiv.textContent = 'Error: ' + err.message;
  }

  isLoading = false;
  sendBtn.disabled = false;
  promptInput.focus();
}
</script>
</body>
</html>"""


# ── API Server ──────────────────────────────────────────────────────


def create_client_app(config: SawyerConfig | None = None) -> FastAPI:
    """Create the FastAPI app for the consumer client."""
    config = config or SawyerConfig()
    local_inference = LocalInference(config)

    app = FastAPI(
        title="Sawyer Client",
        description="Distributed MoE inference — cheaper than your provider",
        version="0.1.0",
    )

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui():
        """Serve the chat UI."""
        return HTMLResponse(content=CHAT_HTML)

    @app.get("/health")
    async def health():
        """Health check."""
        backends = local_inference.is_available()
        return {
            "status": "ok",
            "backends": backends,
            "mode": "local" if not any(backends.values()) else "available",
        }

    @app.get("/v1/models")
    async def list_models():
        """List available models (OpenAI-compatible)."""
        return {
            "object": "list",
            "data": [
                {"id": "sawyer", "object": "model", "owned_by": "sawyer"},
                {"id": "mixtral-8x7b", "object": "model", "owned_by": "sawyer"},
                {"id": "deepseek-v2-lite", "object": "model", "owned_by": "sawyer"},
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint.

        This is the main entry point for inference. It tries:
        1. Sawyer network (distributed MoE) if available
        2. Local fallback (llama.cpp, Ollama) if not
        """
        body = await request.json()
        messages = body.get("messages", [])
        model = body.get("model", "sawyer")
        max_tokens = body.get("max_tokens", 512)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)
        stream = body.get("stream", False)

        # Extract the last user message for local fallback
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if not last_user_msg:
            raise HTTPException(status_code=400, detail="No user message found")

        # Try local inference for now (network routing comes later)
        try:
            result = local_inference.infer(
                prompt=last_user_msg,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model or model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.text,
                        },
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": result.input_tokens,
                    "completion_tokens": result.output_tokens,
                    "total_tokens": result.input_tokens + result.output_tokens,
                },
            }
            return response

        except RuntimeError as e:
            # No backend available
            raise HTTPException(
                status_code=503,
                detail=str(e),
            ) from None
        except Exception as e:
            logger.exception("Inference error")
            raise HTTPException(
                status_code=500,
                detail=f"Inference error: {e}",
            ) from None

    @app.get("/v1/balance")
    async def get_balance():
        """Get token balance for the current account."""
        # Token accounting will be wired to real inference later
        return {
            "balance": 0,
            "mode": "local",
            "message": "Token accounting active when connected to Sawyer network",
        }

    return app


def serve_client(host: str = "localhost", port: int = 8000, config: SawyerConfig | None = None) -> None:
    """Start the consumer client server.

    This is what users run to get cheaper inference. Opens a web UI
    and an OpenAI-compatible API endpoint.
    """
    app = create_client_app(config)
    print(f"\n  Sawyer Client")
    print(f"  Chat UI:     http://{host}:{port}")
    print(f"  API:         http://{host}:{port}/v1/chat/completions")
    print(f"  Models:      http://{host}:{port}/v1/models")
    print()
    uvicorn.run(app, host=host, port=port)