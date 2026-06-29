# Sawyer — Distributed MoE Inference Network

**The load is split. Friends help.**

Sawyer distributes Mixture-of-Experts (MoE) model inference across a network of volunteer-hosted nodes. Each node hosts one or more expert weight files, and a central router activates only the relevant experts per token. Users pay a low monthly subscription ($5/mo) for a token budget — cheap enough to experiment, paid enough to sustain the network.

## Quick Start

```bash
pip install sawyer-core
sawyer register --name my-node --gpu
sawyer serve
```

## How It Works

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

## Pricing

| Tier | Price | Token Budget | Use Case |
|------|-------|-------------|----------|
| Explorer | $5/mo | 500K tokens | Prototyping, experimentation |
| Builder | $20/mo | 2M tokens | Development, testing |
| Operator | $50/mo | 5M tokens | Production workloads |

## Supported Models

| Model | Params | Experts | Active/Token | Q4 Size |
|-------|--------|---------|-------------|---------|
| Mixtral 8x7B | 46.7B | 8 | 2 | ~24GB |
| DeepSeek-V2 Lite | 15.7B | 64 | 6 | ~9GB |
| Qwen1.5-MoE | 14.3B | 60 | 4 | ~7GB |
| DBRX | 132B | 16 | 4 | ~65GB |

## License

BSL-1.1 — free for non-production use. Production use requires a paid license. Converts to Apache 2.0 after the change date.

---

Built by [InFill Systems, LLC](https://infill.systems). Powered by [Bedrock](https://buildonbedrock.dev) identity and audit.