<div align="center"><img src="Sawyer_Agent_Github_Readme_Logo.png" alt="Sawyer Agent on Bedrock" width="800"></div>

# Sawyer Agent

[![PyPI](https://img.shields.io/pypi/v/sawyer-core?color=%2312c7ef&label=pypi%3A%20sawyer-core)](https://pypi.org/project/sawyer-core/)

> **Status: Active prototype** — Provider onboarding and APIs are evolving. Sawyer is under active development toward an alpha milestone.

**The load is split. Friends help.**

Named for Tom Sawyer, who turned an impossible chore into a community effort by making participation irresistible. Sawyer Agent is your local AI that connects to a distributed MoE inference network. Run it on a laptop, a gaming rig, or a server farm. Use the agent. Join the network. Or both.

---

## Get Started

**One command:**

```bash
pip install sawyer-core
sawyer run
```

```powershell
pip install sawyer-core
sawyer run
```

That's it. `sawyer run` detects your setup, picks a model, starts the router, opens the chat UI, and launches the agent. No GPU required on your end — inference happens on the network.

Pick a model, skip the agent, or choose what to run:

```bash
sawyer run glm-5.1:cloud          # Specific model
sawyer run --no-agent             # Router only, no agent
sawyer run --no-browser           # Don't open browser
sawyer run --agent cursor         # Use Cursor instead of Hermes
```

**One-click install (Windows):**

```powershell
irm https://sawyer.infill.systems/install.ps1 | iex
```

**Install from source:**

```bash
git clone https://github.com/drc10101/sawyer-network.git
cd sawyer-network
pip install -e ".[dev]"
```

**GPU inference (hosting expert nodes):**

```bash
pip install sawyer-core[inference]       # bash
pip install "sawyer-core[inference]"     # PowerShell
```

Note: `vllm` and `llama-cpp-python` require CUDA and a C++ compiler. If installation fails, install them separately following their docs, then install sawyer-core without extras.

---

## Using the Agent

You just want to talk to an AI. You don't care about servers or GPUs. These commands are for you.

| Command | What it does |
|---------|-------------|
| `sawyer run` | Start everything at once. Picks a model, starts the router, opens chat. |
| `sawyer run glm-5.1:cloud` | Same as above, but use a specific model. |
| `sawyer run --no-agent` | Start the router only. No agent, just the API. |
| `sawyer run --no-browser` | Don't open the browser window. |
| `sawyer run --agent cursor` | Use Cursor instead of Hermes as your agent. |
| `sawyer chat` | Open the chat web page. Talk to the model in your browser. |
| `sawyer chat --ollama-bridge` | Chat, and also share your local Ollama models with the network. |
| `sawyer models` | Show every model you can use right now. |
| `sawyer models --use chat` | Show only chat models. |
| `sawyer models --use code` | Show only code models. |

---

## Joining the Network

You have a GPU and you want to earn money while it sits idle. These commands are for you.

| Command | What it does |
|---------|-------------|
| `sawyer serve` | Start hosting. Your GPU runs expert models for the network. You earn tokens. |
| `sawyer serve --model mixtral` | Host a specific model's experts. |
| `sawyer status` | See how many tokens you've served, your uptime, and your earnings. |
| `sawyer register` | Sign up this machine as a network node. |
| `sawyer register --gpu` | Sign up and automatically tell the network what GPU you have. |
| `sawyer download` | Download model files ahead of time so serving starts faster. |
| `sawyer bench` | Run a speed test. See how fast your GPU can process tokens. |
| `sawyer account create` | Create your token account. |
| `sawyer provider register` | Register as a provider so you can get paid. |
| `sawyer provider onboarding <id>` | Connect your bank via Stripe so you can receive payouts. |

Run `sawyer --help` for the full list.

---

## Connect Other Tools

Sawyer gives you an OpenAI-compatible API at `http://localhost:8000/v1`. Anything that talks to OpenAI can talk to Sawyer. Just set the base URL.

**Hermes:**
```bash
hermes config set model.base_url http://localhost:8000/v1
hermes config set model.provider openai_compatible
hermes config set model.default glm-5.1:cloud
```

**Claude Code:**
```bash
OPENAI_API_KEY=sawyer OPENAI_BASE_URL=http://localhost:8000/v1 claude
```

**Python:**
```python
from openai import OpenAI

client = OpenAI(api_key="sawyer", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="glm-5.1:cloud",
    messages=[{"role": "user", "content": "Hello"}],
)
```

Works with: Hermes, OpenClaw, Claude Code, Cursor, Continue, Aider, Cline, LangChain, LlamaIndex, CrewAI, AutoGPT, and any OpenAI-compatible client.

Full integration guides: [`docs/agent-integration.md`](docs/agent-integration.md)

---

## Earning on the Network

Your GPU sits idle most of the day. Sawyer puts it to work. A 4090 on Tier 4 earns 4x what a laptop on Tier 1 earns per token.

When serving, Sawyer shows a real-time dashboard at `http://localhost:8000/` -- tokens served, earnings, uptime, model breakdown, tier badge, payout info. API at `/api/stats`.

### Hardware Tiers

| Tier | VRAM | Multiplier | Can Host | Monthly Estimate* |
|------|------|-----------|----------|-------------------|
| Tier 1 | 4 GB | 1x | Qwen1.5-MoE experts (0.5 GB) | $15-50 |
| Tier 2 | 8 GB | 2x | + DeepSeek-V2 experts (0.8 GB) | $40-120 |
| Tier 3 | 12 GB | 3x | + Mixtral experts (1.5 GB) | $80-250 |
| Tier 4 | 24 GB+ | 4x | All experts, local models | $200-800+ |

*Estimates based on 100 Pro subscribers at $15/mo, varies with network size and utilization.

Same 100K tokens served:
- Tier 1 laptop: 100K x 1x = 100K weighted
- Tier 4 monster: 100K x 4x = 400K weighted

The monster PC earns 4x for the same token count because it invested in hardware that can do more for the network.

### How You Get Paid

Every quarter, 70% of all subscription revenue goes into the provider pool:

```
Provider Pool = Total Subscribers x Avg Subscription x 70%
```

The pool is split:
- **90%** by throughput (tokens served x tier multiplier)
- **10%** by uptime (just being available matters)

Payouts quarterly via Stripe Connect. Minimum $10, below that rolls over. Nobody loses money.

---

## Supported Models

| Model | Params | Experts | Active/Token | Q4 Size | Expert Size | Best For |
|-------|--------|---------|-------------|---------|-------------|----------|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~24 GB | ~1.5 GB | Chat, Code |
| DeepSeek-V2 Lite | 15.7B | 64 | 6 | ~9 GB | ~0.8 GB | Chat |
| Qwen1.5-MoE | 14.3B | 60 | 4 | ~7 GB | ~0.5 GB | Chat (lightweight) |
| DBRX Instruct | 132B | 16 | 4 | ~65 GB | ~2.5 GB | Code |

Use `sawyer models` to list all, or filter: `sawyer models --use chat`, `sawyer models --use code`.

---

## Pricing

| Tier | Price | Tokens | Per 1K Tokens | Best For |
|------|-------|--------|---------------|----------|
| Explorer | 14-day free trial | Unlimited | $0.00 | Try it out |
| Pro | $15/mo | 2M | $0.0075 | Development, production workloads |
| Pioneer | $40/mo | 5M | $0.008 | Scale, growing teams |
| Enterprise | $200/mo | 10M | $0.020 | Teams, custom deployment |

14-day free trial with unlimited tokens. Then pick your plan. 70% of subscription revenue goes to the hosts who serve inference — real money attracts real hardware.

~3-6x cheaper than GPT-4. No rate limits. No surprise bills. Token budget resets monthly, unused tokens roll over (max 1 month).

---

## Architecture

```
[User/Client]
     |
     v
[Sawyer Router]  <-- Bedrock identity, consent-gated routing
     |
     +--> [Node: Expert 0]  (RTX 4090, Dallas)     Tier 4 - 4x earnings
     +--> [Node: Expert 2]  (A100, Frankfurt)       Tier 4 - 4x earnings
     +--> [Node: Expert 5]  (RTX 3060, Tokyo)       Tier 3 - 3x earnings
     +--> [Node: Expert 7]  (GTX 1060, Sao Paulo)   Tier 1 - 1x earnings
     |
     v
[Aggregated Output] --> User
```

Sawyer does not require providers to host full models. Providers host isolated MoE expert workloads that the router activates only when needed. That is why Sawyer is not just another distributed inference project — it distributes only the sparse, independently activated sub-networks that MoE architectures make possible.

Built on [Bedrock](https://github.com/drc10101/bedrock) for node identity, consent-gated routing, and auditability. Sawyer runs on Bedrock. Sawyer does not own Bedrock.

### Core Modules

- **`sawyer/router/`** — Expert Router: receives embeddings, routes to correct experts, aggregates output, tracks latency, falls back on timeout
- **`sawyer/node/`** — Node Agent: registers via Bedrock identity, advertises GPU/VRAM/bandwidth, hosts expert weights, serves inference via gRPC/QUIC
- **`sawyer/token/`** — Token Economics: trial, budgets, debit per inference, monthly reset, 70% provider pool
- **`sawyer/provider/`** — Provider Economics & Dashboard: tier multipliers, quarterly payouts, real-time web dashboard
- **`sawyer/identity/`** — Bedrock Integration: cryptographic identity, consent tokens, audit chain
- **`sawyer/model/`** — Model Registry: supported MoE models, versioned expert weights, on-demand download

### Protocol

```
1. Node registers with Sawyer network
   --> Bedrock identity issued (certificate, scope, audit chain)
   --> Node advertises: GPU, VRAM, bandwidth, experts available
   --> Tier classification: Tier 1-4 based on VRAM

2. User sends inference request
   --> Sawyer router authenticates user (token balance check)
   --> Router runs gating network locally to select experts
   --> Router sends expert activation request to node(s)
   --> Node validates consent token, runs expert forward pass
   --> Node returns expert output, logs to audit chain
   --> Router aggregates, returns to user
   --> Token balance debited

3. Quarterly settlement
   --> Provider pool = 70% of all subscription revenue
   --> 90% distributed by weighted throughput (tokens x tier multiplier)
   --> 10% distributed by uptime
   --> Payouts >= $25 via Stripe Connect
   --> Below $25 rolls over to next quarter
```

---

## Dependencies

- **Bedrock** (infill-bedrock): Node identity, consent tokens, audit chain
- **vLLM / llama.cpp**: Expert inference backend
- **gRPC / QUIC**: Low-latency inter-node communication
- **Stripe**: Subscription and host payout management
- **HuggingFace Hub**: Model weight distribution

## License

BSL-1.1 — free for non-production use. Production use requires a paid license. Converts to Apache 2.0 after the change date.

---

**Alpha milestone:** Single-router, two-node demo with one toy MoE model — real node registration, real health checks, real routing logs, real tier-weighted earnings. Prove the network behavior first, then graduate to larger quantized MoE weights.