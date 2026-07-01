[![PyPI](https://img.shields.io/pypi/v/sawyer-core?color=%2312c7ef&label=pypi%3A%20sawyer-core)](https://pypi.org/project/sawyer-core/)

# Sawyer — Distributed MoE Inference Network

> **Status: Active prototype** — Provider onboarding and APIs are evolving. Sawyer is under active development toward an alpha milestone.

**"The load is split. Friends help."**

<div align="center"><img src="sawyer_logo.png" alt="Sawyer on Bedrock" width="600"></div>

Named for Tom Sawyer, who turned an impossible chore into a community effort by making participation irresistible. Sawyer turns GPU inference — a credit-draining trap — into a distributed network where each node carries a piece of the load, and everyone benefits.

**Sawyer does not require providers to host full models.** Providers host isolated MoE expert workloads that the router activates only when needed. That is why Sawyer is not just another distributed inference project — it distributes only the sparse, independently activated sub-networks that MoE architectures make possible.

Built on [Bedrock](https://github.com/drc10101/bedrock) for node identity, consent-gated routing, and auditability. Sawyer runs on Bedrock. Sawyer does not own Bedrock.

---

## Use Cases

### For the Developer with a Laptop

You're building an app that calls LLM APIs. GPT-4 costs $0.03/1K tokens. Claude Haiku is cheaper but still adds up. Sawyer gives you 500K tokens for $5/mo — roughly $0.01/1K tokens — with chat and code models available from day one.

```bash
pip install sawyer-core
sawyer chat                    # Interactive chat UI
sawyer chat --model mixtral   # Pick a model
```

No GPU required. Your laptop connects to the network and inference happens on nodes that have the hardware. You just type.

### For the Gamer with a 4090

Your gaming rig sits idle most of the day. Sawyer puts those GPUs to work. You host expert weights and serve inference to the network. The more you contribute, the more you earn — your 4090 on Tier 4 earns 4x what a laptop on Tier 1 earns per token.

```bash
pip install sawyer-core
sawyer serve                   # Host experts, start earning
sawyer serve --model mixtral  # Pick a model to serve
sawyer status                  # Check your earnings
```

Tier 4 (24GB+ VRAM) can host any model's experts. Tier 1 (4GB) can still participate with Qwen1.5-MoE experts. Everyone earns.

### For the Small Team

Your startup needs inference but can't justify GPU costs. Subscribe at the Builder tier ($20/mo, 2M tokens) and route all your calls through Sawyer. Your inference cost drops by 3-6x compared to major API providers. No rate limits, no surprise bills.

### For the Hobbyist

You want to experiment with local models but only have one GPU. Sawyer lets you download experts, serve them locally or to the network, and use the chat client to interact. Free tier with 50K tokens to start, or subscribe for more.

---

## How You Earn

Sawyer is not volunteer computing. You get paid for the compute you contribute.

### The Pool

Every quarter, 70% of all subscription revenue goes into the provider pool:

```
Provider Pool = Total Subscribers x $5/month x 3 months x 70%
```

With 100 subscribers:
- Revenue: $1,500/quarter
- Provider pool: $1,050
- Platform: $450

With 1,000 subscribers:
- Revenue: $15,000/quarter
- Provider pool: $10,500
- Platform: $4,500

The pool grows as the network grows. More subscribers means more money for everyone.

### Distribution

The pool is split two ways:

| Share | What | Why |
|-------|------|-----|
| 90% | Throughput | Tokens served x tier multiplier — the work you do |
| 10% | Uptime | Just being available matters, even if traffic is low |

### Hardware Tiers

Your earnings depend on what you bring to the network. A 4090 earns more per token than a 1060 because it can host larger, more valuable experts.

| Tier | VRAM | Multiplier | Can Host | Monthly Estimate* |
|------|------|-----------|----------|-------------------|
| Tier 1 | 4 GB | 1x | Qwen1.5-MoE experts (0.5 GB) | $5-15 |
| Tier 2 | 8 GB | 2x | + DeepSeek-V2 experts (0.8 GB) | $15-40 |
| Tier 3 | 12 GB | 3x | + Mixtral experts (1.5 GB) | $30-80 |
| Tier 4 | 24 GB+ | 4x | All experts, local models | $60-200+ |

*Estimates based on 100 subscribers, varies with network size and utilization.

**Same 100K tokens served:**
- Tier 1 laptop: 100K x 1x = 100K weighted
- Tier 4 monster: 100K x 4x = 400K weighted

The monster PC earns 4x for the same token count because it invested in hardware that can do more for the network.

### Example: The Kid with the Monster PC

100 subscribers, one quarter:

| Provider | Tokens | Uptime | Tier | Weighted | Token Earnings | Uptime Earnings | Total |
|----------|--------|--------|------|----------|---------------|----------------|-------|
| Monster PC (4090) | 500K | 720h | Tier 4 (4x) | 2,000,000 | $893 | $35 | **$928** |
| Gaming rig (3060) | 200K | 500h | Tier 3 (3x) | 600,000 | $268 | $24 | **$292** |
| Midrange (1060) | 50K | 300h | Tier 1 (1x) | 50,000 | $22 | $15 | **$37** |

The kid with the 4090 doing most of the work gets the biggest slice. That's the point.

### Payouts

- Quarterly: January-March, April-June, July-September, October-December
- Minimum payout: $25. Below that, your earnings roll over to next quarter
- Methods: Stripe Connect (primary), PayPal
- Failed payouts roll over — nobody loses money

---

## Pricing

| Tier | Price | Tokens | Per 1K Tokens | Best For |
|------|-------|--------|---------------|----------|
| Explorer | $5/mo | 500K | $0.01 | Prototyping, experimentation |
| Builder | $20/mo | 2M | $0.01 | Development, testing |
| Operator | $50/mo | 5M | $0.01 | Production workloads |

~3-6x cheaper than GPT-4, roughly in line with Claude Haiku pricing. No rate limits. No surprise bills. Token budget resets monthly, unused tokens roll over (max 1 month).

---

## Supported Models

Use `sawyer models` to list available models, or filter by use case:

```bash
sawyer models              # All models
sawyer models --use chat   # Chat-focused models
sawyer models --use code   # Code-focused models
```

| Model | Params | Experts | Active/Token | Q4 Size | Expert Size | Best For |
|-------|--------|---------|-------------|---------|-------------|----------|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~24 GB | ~1.5 GB | Chat, Code |
| DeepSeek-V2 Lite | 15.7B | 64 | 6 | ~9 GB | ~0.8 GB | Chat |
| Qwen1.5-MoE | 14.3B | 60 | 4 | ~7 GB | ~0.5 GB | Chat (lightweight) |
| DBRX Instruct | 132B | 16 | 4 | ~65 GB | ~2.5 GB | Code |

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

Every node earns proportional to its hardware contribution. The 4090 in Dallas earns 4x per token because it can host the experts the network needs most.

---

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
- $5/mo subscription grants 500K tokens
- Tokens debit per inference request (input + output tokens)
- Token budget resets monthly, rolls over unused tokens (max 1 month)
- Provider pool = subscribers x $5 x months x 70%
- 90% distributed by weighted throughput, 10% by uptime
- Quarterly payouts, $25 minimum, rollover below threshold

### 4. `sawyer/provider/` — Provider Economics
- **Node Tiers**: 4 hardware tiers (4GB/8GB/12GB/24GB+) with 1x-4x earnings multipliers
- **Revenue Pool**: 70% of subscription revenue distributed to providers quarterly
- **Quarterly Payout**: Pool split by weighted contribution, $25 minimum, Stripe/PayPal
- **Rollover**: Below-threshold earnings carry forward — nobody loses money

### 5. `sawyer/identity/` — Bedrock Integration
- Every node holds a Bedrock cryptographic identity
- Router verifies node certificates before routing
- Consent tokens gate which models a node will serve
- Audit chain logs every inference request for compliance

### 6. `sawyer/model/` — Model Registry
- Catalog of supported MoE models tagged by use case (chat, code)
- Expert weight files versioned and checksummed
- Nodes download experts on registration or on-demand
- Supports Mixtral 8x7B, DeepSeek-V2, Qwen MoE, DBRX

---

## Protocol

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
   --> Provider pool = subscribers x $5 x 3 months x 70%
   --> 90% distributed by weighted throughput (tokens x tier multiplier)
   --> 10% distributed by uptime
   --> Payouts >= $25 via Stripe Connect
   --> Below $25 rolls over to next quarter
```

---

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

---

## Why It Works

- **MoE is more distributable than dense inference.** Experts are independent sub-networks. Unlike tensor parallelism (which splits a single matrix across GPUs), each expert runs its own forward pass. MoE is more distributable than dense tensor-parallel inference because experts are independently activated, but Sawyer's core engineering challenge is keeping routing, expert execution, and aggregation fast enough to feel local.
- **Sparsity means efficiency.** Only ~25% of parameters activate per token on Mixtral. The network doesn't pay for dormant compute.
- **Quantized models fit on consumer hardware.** Q4_K_M Mixtral expert ~1.5GB. A 3090 can host 2-3 experts comfortably alongside other workloads.
- **$5/mo is the sweet spot.** Below the psychological barrier of "another subscription." Enough tokens to prototype, test, and run real workloads. Revenue sustains the network without extracting from users.
- **Hardware investment is rewarded.** Tier 4 (24GB+) earns 4x per token compared to Tier 1 (4GB). Your 4090 pays for itself.

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