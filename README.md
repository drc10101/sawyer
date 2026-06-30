[![PyPI](https://img.shields.io/pypi/v/sawyer-core?color=%2312c7ef&label=pypi%3A%20sawyer-core)](https://pypi.org/project/sawyer-core/)

# Sawyer — Distributed MoE Inference Network

> **Status: Active prototype** — Provider onboarding and APIs are evolving. Sawyer is under active development toward an alpha milestone.

**"The load is split. Friends help."**

<div align="center"><img src="sawyer_logo.png" alt="Sawyer on Bedrock" width="600"></div>

Named for Tom Sawyer, who turned an impossible chore into a community effort by making participation irresistible. Sawyer turns GPU inference — a credit-draining trap — into a distributed network where each node carries a piece of the load, and everyone benefits.

**Sawyer does not require providers to host full models.** Providers host isolated MoE expert workloads that the router activates only when needed. That is why Sawyer is not just another distributed inference project — it distributes only the sparse, independently activated sub-networks that MoE architectures make possible.

Built on [Bedrock](https://github.com/drc10101/bedrock) for node identity, consent-gated routing, and auditability. Sawyer runs on Bedrock. Sawyer does not own Bedrock.

## The Problem

Cloud API credits run out. A single model call on GPT-4-class inference costs cents that compound into hundreds of dollars. Frontier quantized models (Mixtral 8x7B, DeepSeek-V2, Qwen MoE) can run locally but require 2-4 GPUs for full precision. Most developers have one GPU — or none.

## The Idea

A distributed network where:

1. **Volunteers host MoE expert weights** on their hardware (a single RTX 3090 can host one expert)
2. **A router activates only the relevant experts per token** (MoE sparsity — only 2 of 8 experts fire on Mixtral)
3. **Users pay $5/month** for a token budget — cheap enough to experiment, paid enough to sustain
4. **Hosts earn a share** proportional to compute contributed — the incentive altruism alone can't provide
5. **Bedrock provides the trust layer** — node identity, consent tokens, audit chain

## Why It Works

- **MoE is more distributable than dense inference.** Experts are independent sub-networks. Unlike tensor parallelism (which splits a single matrix across GPUs), each expert runs its own forward pass. MoE is more distributable than dense tensor-parallel inference because experts are independently activated, but Sawyer's core engineering challenge is keeping routing, expert execution, and aggregation fast enough to feel local.
- **Sparsity means efficiency.** Only ~25% of parameters activate per token on Mixtral. The network doesn't pay for dormant compute.
- **Quantized models fit on consumer hardware.** Q4_K_M Mixtral expert ≈ 1.5GB. A 3090 can host 2-3 experts comfortably alongside other workloads.
- **$5/mo is the sweet spot.** Below the psychological barrier of "another subscription." Enough tokens to prototype, test, and run real workloads. Revenue sustains the network without extracting from users.

## Architecture

```
[User/Client]
     │
     ▼
[Sawyer Router]  ←── Bedrock identity, consent-gated routing
     │
     ├──→ [Node: Expert 0]  (RTX 3090, Dallas)
     ├──→ [Node: Expert 2]  (A100, Frankfurt)
     ├──→ [Node: Expert 5]  (M2 Max, Tokyo)
     └──→ [Node: Expert 7]  (T4, São Paulo)
     │
     ▼
[Aggregated Output] → User
```

## Core Modules

### 1. `sawyer/router/` — Expert Router
- Receives token embeddings from the user's local dense layers
- Routes to the correct expert(s) based on the model's gating network
- Aggregates expert outputs, returns to user
- Tracks latency per node, falls back to redundant experts on timeout

### 2. `sawyer/node/` — Node Agent
- Registers with the network via Bedrock node identity
- Advertises capabilities: GPU model, VRAM, bandwidth, latency
- Hosts one or more expert weight files
- Serves inference requests via encrypted gRPC/QUIC
- Reports health and throughput to the router

### 3. `sawyer/token/` — Token Economics
- $5/mo subscription grants a token budget (e.g., 500K tokens)
- Tokens debit per inference request (input + output tokens)
- Token budget resets monthly, rolls over unused tokens (max 1 month)
- Hosts earn credits proportional to tokens served
- Credits convert to USD payout at thresholds ($10 minimum)

### 4. `sawyer/identity/` — Bedrock Integration
- Every node holds a Bedrock cryptographic identity
- Router verifies node certificates before routing
- Consent tokens gate which models a node will serve
- Audit chain logs every inference request for compliance

### 5. `sawyer/model/` — Model Registry
- Catalog of supported MoE models and their expert layouts
- Expert weight files versioned and checksummed
- Nodes download experts on registration or on-demand
- Supports Mixtral 8x7B, DeepSeek-V2, Qwen MoE, and extensible for new models

## Protocol

```
1. Node registers with Sawyer network
   → Bedrock identity issued (certificate, scope, audit chain)
   → Node advertises: GPU, VRAM, bandwidth, experts available

2. User sends inference request
   → Sawyer router authenticates user (token balance check)
   → Router runs gating network locally to select experts
   → Router sends expert activation request to node(s)
   → Node validates consent token, runs expert forward pass
   → Node returns expert output, logs to audit chain
   → Router aggregates, returns to user
   → Token balance debited

3. Monthly settlement
   → Host credits calculated from tokens served
   → Payouts processed at $10 threshold
```

## Pricing

| Tier | Price | Token Budget | Use Case |
|------|-------|-------------|----------|
| Explorer | $5/mo | 500K tokens | Prototyping, experimentation |
| Builder | $20/mo | 2M tokens | Development, testing |
| Operator | $50/mo | 5M tokens | Production workloads |

Token costs vary by model (frontier models cost more tokens per request). Quantized models get a token discount (lower quality, lower cost).

## Host Economics

- Earn credits per token of expert inference served
- Credits proportional to: tokens served × model complexity × response time SLA
- Payout at $10 threshold via Stripe
- A single RTX 3090 hosting 2 Mixtral experts at ~30% utilization: estimated $8-15/mo

## Supported Models (Initial)

| Model | Params | Experts | Active/Token | Q4_K_M Size | Expert Size |
|-------|--------|---------|-------------|-------------|-------------|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~24GB | ~1.5GB |
| DeepSeek-V2 Lite | 15.7B | 64 (shared) | 6 | ~9GB | varies |
| Qwen1.5-MoE-A2.7B | 14.3B | 60 | 4 | ~7GB | varies |
| DBRX | 132B | 16 | 4 | ~65GB | ~2.5GB |

## Repository Structure

```
sawyer/
├── README.md
├── LICENSE                  # BSL-1.1 (same as Bedrock)
├── pyproject.toml
├── sawyer/
│   ├── __init__.py
│   ├── cli.py               # sawyer register, sawyer serve, sawyer status
│   ├── router/
│   │   ├── __init__.py
│   │   ├── gateway.py       # Main router server (gRPC/QUIC)
│   │   ├── scheduler.py     # Expert selection, load balancing
│   │   ├── gating.py        # Model-specific gating network runner
│   │   └── aggregator.py    # Combine expert outputs
│   ├── node/
│   │   ├── __init__.py
│   │   ├── agent.py         # Node agent — hosts experts, serves inference
│   │   ├── registry.py      # Register capabilities, download experts
│   │   ├── inference.py     # Expert forward pass (vLLM / llama.cpp)
│   │   └── health.py        # Heartbeat, throughput reporting
│   ├── token/
│   │   ├── __init__.py
│   │   ├── budget.py        # Token budget management
│   │   ├── accounting.py    # Debit/credit per request
│   │   └── settlement.py    # Host payouts, Stripe integration
│   ├── identity/
│   │   ├── __init__.py
│   │   ├── bedrock.py       # Bedrock SDK integration (identity, consent, audit)
│   │   └── verification.py  # Node certificate verification
│   ├── model/
│   │   ├── __init__.py
│   │   ├── registry.py      # Model catalog, expert layouts
│   │   ├── download.py      # Expert weight distribution
│   │   └── formats.py       # GGUF, safetensors handling
│   └── config.py            # Configuration management
├── tests/
│   ├── test_router.py
│   ├── test_node.py
│   ├── test_token.py
│   ├── test_identity.py
│   └── test_model.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── HOSTING.md           # How to host an expert node
│   ├── MODELS.md            # Supported models and expert layouts
│   └── TOKEN_ECONOMICS.md   # Detailed token economics
└── site/
    └── index.html           # Landing page
```

## Installation

**Requires Python 3.11 or later.**

```bash
pip install sawyer-core
```

For GPU inference (hosting expert nodes):

```bash
pip install sawyer-core[inference]
```

Note: `vllm` and `llama-cpp-python` require CUDA and a C++ compiler. If installation fails, install them separately following their docs, then install sawyer-core without extras.

Or install from source for development:

```bash
git clone https://github.com/drc10101/sawyer.git
cd sawyer
pip install -e ".[dev]"
```

### Running

After install, Sawyer can be run either way:

```bash
sawyer serve                # if Python Scripts is on PATH
python -m sawyer serve      # works everywhere, no PATH needed
```

### One-Click Install (Windows)

Double-click `sawyer.bat` in the repo, or run the PowerShell installer for a desktop shortcut and Start Menu entry:

```powershell
irm https://infill.systems/install/sawyer.ps1 | iex
```

To uninstall: `.\install_sawyer.ps1 -Uninstall`

## Dependencies

- **Bedrock** (infill-bedrock): Node identity, consent tokens, audit chain
- **vLLM / llama.cpp**: Expert inference backend
- **gRPC / QUIC**: Low-latency inter-node communication
- **Stripe**: Subscription and host payout management
- **HuggingFace Hub**: Model weight distribution

## License

BSL-1.1 — free for non-production use. Production use requires a paid license. Converts to Apache 2.0 after the change date.

---

**Alpha milestone:** Single-router, two-node demo with one toy MoE model — real node registration, real health checks, real routing logs, fake economics. Prove the network behavior first, then graduate to larger quantized MoE weights.