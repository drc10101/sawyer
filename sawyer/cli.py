"""Sawyer CLI — Register nodes, serve experts, check status, manage subscriptions.

Usage:
    sawyer register       Register this machine as a Sawyer node
    sawyer serve          Start serving expert inference requests
    sawyer run            One command: start Sawyer + Ollama + agent
    sawyer chat           Start the consumer chat client (web UI + API)
    sawyer bench          Benchmark MoE prefill speedup (before/after)
    sawyer fast-llama     Download Sawyer Fast Llama (optimized binary)
    sawyer status         Show network status and token balance
    sawyer models         List available models and expert layouts
    sawyer download       Download model weights to local cache
    sawyer account        Create or show token account
    sawyer quota          Check token quota before inference
    sawyer provider       Manage node provider registration and payouts
"""

import argparse
import asyncio
import os
import sys

import grpc
import httpx

from sawyer.bench import cmd_bench
from sawyer.config import SawyerConfig
from sawyer.download import ensure_llama_bench, get_binary_info
from sawyer.model.registry import get_model, list_models
from sawyer.node.agent import SawyerNode
from sawyer.node.weights import WeightLoader
from sawyer.token.accounting import TokenAccountant
from sawyer.token.budget import TIER_PRICING, SubscriptionTier


def cmd_register(args) -> int:
    """Register this machine as a Sawyer node."""
    config = SawyerConfig(
        node_name=args.name,
        bedrock_url=args.bedrock_url,
        bedrock_license_key=args.bedrock_key,
    )
    node = SawyerNode(config)
    node_id = asyncio.run(node.register(name=args.name or "sawyer-node"))
    print(f"Registered node: {node_id}")

    if args.gpu:
        print("  GPU detection: available (auto-detect in serve mode)")

    if args.experts:
        print(f"  Experts: {args.experts}")

    print(f"  Router: {config.router_url}")
    print(f"  Bedrock: {config.bedrock_url}")
    print(f"  Max experts: {config.max_experts}")
    return 0


def cmd_serve(args) -> int:
    """Start serving expert inference requests."""
    config = SawyerConfig(
        node_name=args.name,
        inference_port=args.port,
        router_url=args.router,
    )
    node = SawyerNode(config)
    print(f"Starting Sawyer node on port {args.port}")
    print(f"  Router: {args.router}")
    print(f"  Backend: {args.backend}")
    print(f"  Max experts: {config.max_experts}")
    if args.offline:
        print("  Mode: offline (no router connection)")

    # If a model is specified, set up the inference backend
    if args.model:
        model_name = args.model
        try:
            model = get_model(model_name)
            print(f"  Model: {model.display_name}")
            print(f"    {model.num_experts} experts, {model.active_experts} active")
        except ValueError:
            print(f"  Unknown model: {model_name}")
            print(f"  Available: {', '.join(m.name for m in list_models())}")
            return 1

        # Check cache
        loader = WeightLoader(config)
        if loader.is_cached(model_name):
            path = loader.get_cached_path(model_name)
            print(f"  Weights cached: {path}")
        else:
            print(f"  Weights not cached. Run 'sawyer download {model_name}' first.")
            return 1

    # Start the node
    try:
        asyncio.run(node.start(offline=args.offline))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except grpc.RpcError as e:
        print(f"\n  ERROR: Could not connect to router at {args.router}")
        print(f"  {e.details()}")
        print("\n  The Sawyer router is not yet available.")
        print("  Start in offline mode: sawyer serve --offline")
        print("  Or check your connection and try again.")
        return 1
    return 0


def cmd_run(args) -> int:
    """One command to start everything — Sawyer, Ollama, and your agent.

    Detects what's running, starts what isn't, and configures your agent
    to point at Sawyer (not Ollama directly) so you get routing, thinking
    model support, and multi-backend failover.

    Usage:
        sawyer run                      # Auto-detect best model, start everything
        sawyer run glm-5.1:cloud        # Use specific model
        sawyer run --no-agent            # Start Sawyer only, don't launch agent
        sawyer run --agent cursor        # Launch Cursor instead of Hermes
    """
    import shutil
    import subprocess
    import time
    import webbrowser

    from sawyer.client import LocalInference

    config = SawyerConfig()
    inference = LocalInference(config)
    model = args.model

    # ── Step 1: Check what's already running ──────────────────────
    print()
    print("  Sawyer — One command. Everything running.")
    print()

    # Check Ollama
    ollama_running = False
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{config.ollama_url}/api/tags")
            if resp.status_code == 200:
                ollama_running = True
    except Exception:
        pass

    if ollama_running:
        print(f"  [OK] Ollama running at {config.ollama_url}")
    else:
        print(f"  [--] Ollama not detected at {config.ollama_url}")
        # Try to start Ollama
        ollama_exe = shutil.which("ollama")
        if ollama_exe:
            print("  [..] Starting Ollama...")
            try:
                subprocess.Popen(
                    [ollama_exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                # Wait for Ollama to become available
                for _ in range(15):
                    time.sleep(1)
                    try:
                        with httpx.Client(timeout=1.0) as client:
                            resp = client.get(f"{config.ollama_url}/api/tags")
                            if resp.status_code == 200:
                                ollama_running = True
                                break
                    except Exception:
                        continue
                if ollama_running:
                    print(f"  [OK] Ollama started at {config.ollama_url}")
                else:
                    print("  [!!] Ollama didn't start in time. Run 'ollama serve' manually.")
            except Exception as e:
                print(f"  [!!] Could not start Ollama: {e}")
        else:
            print("  [!!] Ollama not installed. Install: https://ollama.com")

    # ── Step 2: Discover models ──────────────────────────────────
    print()
    models = inference.discover_models(force=True)
    if not models:
        print("  [!!] No models found.")
        if ollama_running:
            print("      Pull a model: ollama pull llama3")
            print("      Then run: sawyer run llama3")
        return 1

    print(f"  Models available ({len(models)}):")
    for m in models:
        size = m.details.get("parameter_size", "") if m.details else ""
        size_str = f" ({size})" if size else ""
        print(f"    {m.id}{size_str}")

    # ── Step 3: Resolve model ─────────────────────────────────────
    if not model:
        # Default: pick the first discovered model
        model = models[0].id
        print(f"\n  Using: {model} (first available)")
    else:
        # Try to resolve short name to full name
        resolved, _ = inference._resolve_model(model)
        if resolved != model:
            print(f"\n  Using: {resolved} (resolved from {model})")
            model = resolved
        else:
            print(f"\n  Using: {model}")

    # ── Step 4: Start Sawyer router ────────────────────────────────
    sawyer_url = f"http://{args.host}:{args.port}"
    sawyer_running = False
    try:
        with httpx.Client(timeout=1.0) as client:
            resp = client.get(f"{sawyer_url}/health")
            if resp.status_code == 200:
                sawyer_running = True
    except Exception:
        pass

    if sawyer_running:
        print(f"  [OK] Sawyer router running at {sawyer_url}")
    else:
        print(f"  [..] Starting Sawyer router on port {args.port}...")
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sawyer.cli",
                "chat",
                "--host",
                args.host,
                "--port",
                str(args.port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        # Wait for Sawyer to become available
        for _ in range(15):
            time.sleep(1)
            try:
                with httpx.Client(timeout=1.0) as client:
                    resp = client.get(f"{sawyer_url}/health")
                    if resp.status_code == 200:
                        sawyer_running = True
                        break
            except Exception:
                continue
        if sawyer_running:
            print(f"  [OK] Sawyer router started at {sawyer_url}")
        else:
            print(f"  [!!] Sawyer didn't start. Try: sawyer chat --port {args.port}")
            return 1

    # ── Step 5: Print config for agent frameworks ─────────────────
    print()
    print("  ── Agent Configuration ──────────────────────────")
    print()
    print(f"  Base URL:  {sawyer_url}/v1")
    print(f"  Model:     {model}")
    print("  API Key:   sawyer (or any non-empty string)")
    print()
    print("  Hermes (bash):")
    print(f"    hermes config set model.base_url {sawyer_url}/v1")
    print("    hermes config set model.provider openai_compatible")
    print(f"    hermes config set model.default {model}")
    print()
    print("  Hermes (PowerShell):")
    print(f'    hermes config set model.base_url "{sawyer_url}/v1"')
    print("    hermes config set model.provider openai_compatible")
    print(f'    hermes config set model.default "{model}"')
    print()
    print("  OpenAI-compatible agents (bash):")
    print(f"    OPENAI_API_KEY=sawyer OPENAI_BASE_URL={sawyer_url}/v1 <agent>")
    print()
    print("  OpenAI-compatible agents (PowerShell):")
    print(f'    $env:OPENAI_API_KEY="sawyer"; $env:OPENAI_BASE_URL="{sawyer_url}/v1"; <agent>')
    print()

    # ── Step 6: Open browser to chat UI ──────────────────────────
    if not args.no_browser:
        webbrowser.open(f"{sawyer_url}")

    # ── Step 7: Launch agent if requested ─────────────────────────
    if args.no_agent:
        print("  Ready. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")
        return 0

    agent = args.agent or "hermes"
    agent_cmd = shutil.which(agent)

    if not agent_cmd:
        # Try common paths
        common_paths = {
            "hermes": (
                os.path.expanduser("~/AppData/Local/hermes/hermes-agent/venv/Scripts/hermes")
                if os.name == "nt"
                else os.path.expanduser("~/.local/bin/hermes")
            ),
        }
        agent_cmd = common_paths.get(agent)

    if agent_cmd and os.path.isfile(agent_cmd):
        print(f"  [..] Launching {agent}...")
        # Set environment for the agent
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "sawyer"
        env["OPENAI_BASE_URL"] = f"{sawyer_url}/v1"

        if agent == "hermes":
            env["HERMES_MODEL"] = model
            env["HERMES_BASE_URL"] = f"{sawyer_url}/v1"

        try:
            proc = subprocess.Popen(
                [agent_cmd, "--model", model] if agent == "hermes" else [agent_cmd],
                env=env,
            )
            print(f"  [OK] {agent} launched (PID {proc.pid})")
            print()
            print(f"  Sawyer router: {sawyer_url}")
            print(f"  Chat UI:        {sawyer_url}")
            print(f"  {agent} PID:     {proc.pid}")
            print()
            print("  Press Ctrl+C to stop all services.")
            proc.wait()
        except KeyboardInterrupt:
            print(f"\n  Stopping {agent}...")
            proc.terminate()
        except Exception as e:
            print(f"  [!!] Could not launch {agent}: {e}")
    else:
        print(f"  [--] {agent} not found. Configure manually using the settings above.")
        print()
        print("  Ready. Press Ctrl+C to stop Sawyer.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")

    return 0


def cmd_chat(args) -> int:
    """Start the consumer chat client — web UI + OpenAI-compatible API.

    This is what the user with the 8GB laptop runs. Opens a web UI at
    localhost:8000 where they can chat with Sawyer-powered models.
    Also provides an OpenAI-compatible API so any tool (curl, OpenAI SDK,
    Ollama clients) can point at it.
    """
    from sawyer.client import LocalInference, serve_client

    print()
    print("  Sawyer — Distributed MoE Inference")
    print("  Cheaper than your provider. The load is split, friends help.")
    print()

    # Show what's available before starting the server
    config = SawyerConfig()
    inference = LocalInference(config)

    # discover_models() already probes all backends concurrently,
    # so we derive backend status from discovery results (no redundant is_available call)
    models = inference.discover_models()
    available_backends = {m.backend for m in models}

    print("  Checking backends...")
    for name in ["ollama", "llama_cpp", "vllm", "lm_studio"]:
        status = "[OK]" if name in available_backends else "[--]"
        print(f"    {status} {name}")

    if not available_backends:
        print()
        print("  No backends detected. Install Ollama: https://ollama.com")
        print("  Then: ollama pull llama3 && sawyer chat")
        print()

    if models:
        print(f"\n  Models available ({len(models)}):")
        for m in models[:5]:
            size = m.details.get("parameter_size", "") if m.details else ""
            size_str = f" ({size})" if size else ""
            print(f"    {m.id}{size_str}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")
    elif available_backends:
        print("\n  Backend detected but no models found.")
        print("  Pull a model: ollama pull llama3")

    print()

    if getattr(args, "ollama_bridge", False):
        print("  Ollama bridge enabled — serving local Ollama to the network.")
        print()

    try:
        serve_client(
            host=args.host,
            port=args.port,
            ollama_bridge=getattr(args, "ollama_bridge", False),
        )
    except KeyboardInterrupt:
        print("\n  Shutting down...")
    return 0


def cmd_status(args) -> int:
    """Show network status and token balance."""
    config = SawyerConfig()
    node = SawyerNode(config)

    print("Sawyer Network Status")
    print("=" * 40)

    # Node info
    if node.node_id:
        print(f"  Node ID: {node.node_id}")
    else:
        print("  Node: not registered (run 'sawyer register' first)")

    # Identity
    from sawyer.identity.bedrock import SawyerIdentity

    identity = SawyerIdentity(config)
    print(f"  Bedrock: {'connected' if identity.is_connected else 'local mode'}")

    # Token account
    accountant = TokenAccountant()
    account = accountant.get_account(args.user or "default")
    if account:
        print(f"\nToken Balance ({account.tier.value}):")
        print(f"  Monthly budget:  {account.balance.monthly_budget:,}")
        print(f"  Current balance: {account.balance.current_balance:,}")
        print(f"  Rollover:        {account.balance.rollover:,}")
        print(f"  Total available: {account.balance.total_available:,}")
        print(f"  Inferences:      {account.total_inferences}")
        print(f"  Tokens used:     {account.total_tokens_used:,}")
    else:
        print("\n  No token account (run 'sawyer account create')")

    # Cluster status
    from sawyer.router.scheduler import ExpertScheduler

    scheduler = ExpertScheduler()
    status = scheduler.get_cluster_status()
    print("\nCluster:")
    print(f"  Nodes: {status['total_nodes']}")
    print(f"  Healthy: {status['healthy_nodes']}")
    print(f"  Utilization: {status['utilization']}")

    return 0


def cmd_models(args) -> int:
    """List available models and expert layouts."""
    use = getattr(args, "use", None)
    models = list_models(use=use)

    if not models:
        print(f"No models found for use case: {use}")
        print("Available use cases: chat, code")
        return 1

    print("Available Models")
    print("=" * 60)

    for m in models:
        tags = ", ".join(m.tags)
        print(f"\n  {m.display_name} ({m.name})")
        print(f"    {m.description}")
        print(f"    Best for: {tags}")
        print(f"    Experts: {m.num_experts} total, {m.active_experts} active")
        print(f"    Parameters: {m.total_params_b:.1f}B total, ~{m.active_params_b:.1f}B active")
        print(
            f"    Q4 memory: {m.model_size_gb_q4:.1f} GB (full), "
            f"~{m.expert_size_gb_q4:.1f} GB per expert"
        )
        print(f"    Context: {m.context_length:,} tokens")
        print(
            f"    Min VRAM: {m.min_vram_gb:.0f} GB (full), "
            f"{m.min_vram_per_expert_gb:.0f} GB (one expert)"
        )

    print(f"\n  {len(models)} models available")
    print("\n  Tip: 'sawyer models --use chat' or '--use code' to filter")
    print("  Tip: 'sawyer extract mixtral-8x7b --experts 0,1' to host specific experts")
    return 0


def cmd_download(args) -> int:
    """Download model weights to local cache."""
    config = SawyerConfig()
    model_name = args.model

    try:
        model = get_model(model_name)
    except ValueError:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(m.name for m in list_models())}")
        return 1

    print(f"Downloading {model.display_name} weights...")
    print(f"  Size: {model.model_size_gb_q4:.1f} GB (Q4_K_M quantization)")

    loader = WeightLoader(config)
    if loader.is_cached(model_name) and not args.force:
        path = loader.get_cached_path(model_name)
        print(f"  Already cached: {path}")
        print("  Use --force to re-download")
        return 0

    try:
        wf = loader.download_weight(model_name, verify=not args.no_verify)
        print(f"  Downloaded: {wf.path}")
        print(f"  Size: {wf.size_bytes / (1024**3):.1f} GB")
        if wf.sha256:
            print(f"  SHA-256: {wf.sha256[:16]}...")
        print("  Done!")
    except Exception as e:
        print(f"  Download failed: {e}")
        return 1

    return 0


def cmd_fast_llama(args) -> int:
    """Download or update the Sawyer Fast Llama binary.

    Downloads the optimized llama.cpp binary from GitHub Releases to
    ~/.sawyer/bin/. This binary includes MoE prefill optimizations
    (pinned mmap + expert prefetch) for faster inference on MoE models.
    """
    try:
        binary = ensure_llama_bench(force=args.force)
        print(f"  Sawyer Fast Llama installed: {binary}")
        info = get_binary_info(binary)
        if info:
            print(f"  Version: {info.get('version', 'unknown')}")
            print(f"  Commit:  {info.get('commit', 'unknown')}")
        print()
        print("  Run 'sawyer bench -m model.gguf' to benchmark MoE speedup.")
        return 0
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return 1


def cmd_extract(args) -> int:
    """Extract per-expert weight shards from a GGUF model.

    Splits a downloaded MoE model into individual expert shards that can
    be loaded by separate nodes. Each shard contains the FFN weights for
    one expert across all layers, plus the gating network weights.
    """
    from sawyer.expert.extractor import ExpertExtractor

    model_name = args.model

    try:
        model = get_model(model_name)
    except ValueError:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(m.name for m in list_models())}")
        return 1

    # Resolve which experts to extract
    if args.experts:
        expert_ids = [int(e) for e in args.experts]
        max_expert = model.num_experts - 1
        for eid in expert_ids:
            if eid < 0 or eid > max_expert:
                print(f"Invalid expert {eid}: must be 0-{max_expert}")
                return 1
    else:
        expert_ids = None  # Extract all

    print(f"Extracting {model.display_name} expert shards")
    print(f"  Total experts: {model.num_experts}")
    print(f"  Per-expert size: ~{model.expert_size_gb_q4:.1f} GB")
    if expert_ids:
        print(f"  Extracting: experts {expert_ids}")
    else:
        print(f"  Extracting: all {model.num_experts} experts")

    extractor = ExpertExtractor()

    try:
        paths = extractor.extract_and_save(
            model_name=model_name,
            gguf_path=args.gguf,
            expert_ids=expert_ids,
            output_dir=args.output,
        )
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        return 1

    print(f"\n  Extracted {len(paths)} expert shards:")
    for path in paths:
        size_mb = path.stat().st_size / (1024**2)
        print(f"    {path.name}: {size_mb:.1f} MB")

    print("\n  Done! Use 'sawyer serve --expert <id>' to load a specific expert.")
    return 0


def cmd_account(args) -> int:
    """Create or show token account."""
    accountant = TokenAccountant()

    if args.action == "create":
        tier = SubscriptionTier(args.tier)
        try:
            account = accountant.create_account(
                user_id=args.user,
                tier=tier,
                rollover=args.rollover or 0,
            )
            print(f"Account created: {account.user_id}")
            print(f"  Tier: {account.tier.value}")
            print(f"  Monthly budget: {account.balance.monthly_budget:,} tokens")
            print(f"  Price: ${TIER_PRICING[tier]}/mo")
            print(f"  Total available: {account.balance.total_available:,} tokens")
        except Exception as e:
            print(f"Error: {e}")
            return 1

    elif args.action == "show":
        account = accountant.get_account(args.user)
        if not account:
            print(f"No account found for user '{args.user}'")
            return 1
        summary = accountant.get_usage_summary(args.user)
        print(f"Account: {summary['user_id']}")
        print(f"  Tier: {summary['tier']}")
        print(f"  Tokens used: {summary['tokens_used']:,}")
        print(f"  Tokens remaining: {summary['tokens_remaining']:,}")
        print(f"  Rollover: {summary['rollover_tokens']:,}")
        print(f"  Inferences: {summary['total_inferences']}")
        print(f"  Active: {summary['is_active']}")

    elif args.action == "list":
        summaries = accountant.get_all_summaries()
        if not summaries:
            print("No accounts found")
        for s in summaries:
            print(f"  {s['user_id']}: {s['tier']} — {s['tokens_remaining']:,} tokens remaining")

    return 0


def cmd_quota(args) -> int:
    """Check token quota before inference."""
    accountant = TokenAccountant()
    has_quota = accountant.check_quota(args.user, args.tokens)

    if has_quota:
        account = accountant.get_account(args.user)
        print(f"Quota OK: {account.balance.total_available:,} tokens available")
        print(f"  Requested: {args.tokens:,} tokens")
        print(f"  Remaining after: {account.balance.total_available - args.tokens:,} tokens")
    else:
        print(f"Quota exceeded: insufficient tokens for {args.tokens:,} token request")
        account = accountant.get_account(args.user)
        if account:
            print(f"  Available: {account.balance.total_available:,} tokens")
        return 1

    return 0


def cmd_provider(args) -> int:
    """Manage node provider registration and payouts."""
    from sawyer.provider.manager import PayoutSchedule, ProviderManager
    from sawyer.storage.database import SawyerStorage

    config = SawyerConfig()
    storage = SawyerStorage(config.database_url)
    mgr = ProviderManager(storage=storage)

    if args.provider_action == "register":
        provider = mgr.register(
            email=args.email,
            display_name=args.name,
            legal_name=args.legal_name or args.name,
            phone=args.phone or "",
            country=args.country,
            payout_schedule=(
                PayoutSchedule.QUARTERLY
                if args.schedule == "quarterly"
                else PayoutSchedule.MONTHLY
            ),
        )
        print("Provider registered!")
        print(f"  ID:          {provider.provider_id}")
        print(f"  Name:        {provider.display_name}")
        print(f"  Email:       {provider.email}")
        print(f"  Status:      {provider.status.value}")
        print(f"  Payout:      {provider.payout_schedule.value}")
        print("\nIMPORTANT: Providers must complete Stripe Connect onboarding to receive payouts.")
        print(
            "Earnings will accumulate but cannot be disbursed "
            "until Stripe onboarding is complete."
        )
        print(f"  sawyer provider onboarding {provider.provider_id}")

    elif args.provider_action == "status":
        provider = mgr.get_provider(args.provider_id)
        if provider is None:
            print(f"Provider {args.provider_id} not found")
            storage.close()
            return 1

        summary = mgr.get_provider_summary(args.provider_id)
        print(f"Provider Status: {summary['display_name']}")
        print(f"  ID:              {summary['provider_id']}")
        print(f"  Email:          {summary['email']}")
        print(f"  Status:         {summary['status']}")
        print(f"  Country:        {summary['country']}")
        print(f"  Nodes:          {summary['nodes']}")
        print(f"  Payout:         {summary['payout_schedule']}")
        print("\n  Earnings:")
        print(f"    Tokens served:  {summary['earnings']['total_tokens_served']:,}")
        print(f"    Total earned:   ${summary['earnings']['total_usd_earned']:.2f}")
        print(f"    Total paid:     ${summary['earnings']['total_usd_paid']:.2f}")
        print(f"    Available:      ${summary['earnings']['available_balance']:.2f}")
        print(f"    Pending:       ${summary['earnings']['pending_payouts']:.2f}")
        print("\n  Verification:")
        print(f"    Stripe:        {summary['verification']['stripe_connected']}")
        print(f"    Verified:      {summary['verification']['stripe_verified']}")
        print(f"    Tax ID:        {summary['verification']['tax_id_provided']}")

    elif args.provider_action == "onboarding":
        provider = mgr.get_provider(args.provider_id)
        if provider is None:
            print(f"Provider {args.provider_id} not found")
            storage.close()
            return 1
        mgr.verify_provider(
            args.provider_id,
            stripe_connect_id=f"acct_mock_{args.provider_id}",
        )
        print(f"Stripe Connect onboarding initiated for {args.provider_id}")
        print("  Status:    verified")
        print(f"  Stripe ID: acct_mock_{args.provider_id}")

    elif args.provider_action == "payouts":
        provider = mgr.get_provider(args.provider_id)
        if provider is None:
            print(f"Provider {args.provider_id} not found")
            storage.close()
            return 1

        payouts = mgr.get_payout_history(args.provider_id)
        if not payouts:
            print(f"No payout history for {args.provider_id}")
        else:
            print(f"Payout History for {provider.display_name}:")
            for p in payouts:
                print(
                    f"  {p.payout_id}  ${p.amount_usd:.2f}  " f"{p.status.value}  {p.period_label}"
                )

    elif args.provider_action == "network":
        summary = mgr.get_network_summary()
        print("Sawyer Provider Network:")
        print(f"  Total providers:    {summary['total_providers']}")
        print(f"  Active providers:   {summary['active_providers']}")
        print(f"  Verified providers: {summary['verified_providers']}")
        print(f"  Total nodes:        {summary['total_nodes']}")
        print(f"  Tokens served:      {summary['total_tokens_served']:,}")
        print(f"  Total earned:      ${summary['total_usd_earned']:.2f}")
        print(f"  Total paid:        ${summary['total_usd_paid']:.2f}")
        print(f"  Total pending:     ${summary['total_usd_pending']:.2f}")

    elif args.provider_action == "payout":
        provider = mgr.get_provider(args.provider_id)
        if provider is None:
            print(f"Provider {args.provider_id} not found")
            storage.close()
            return 1

        if not provider.is_eligible_for_payout:
            print("Not eligible for payout")
            print(f"  Available: ${provider.available_balance:.2f}")
            print(f"  Minimum:   ${provider.min_payout_usd:.2f}")
            print(f"  Verified:  {provider.stripe_account_verified}")
            storage.close()
            return 1

        payout = mgr.process_payout(args.provider_id)
        if payout is None:
            print(f"Could not process payout for {args.provider_id}")
            storage.close()
            return 1

        print("Payout processed!")
        print(f"  ID:       {payout.payout_id}")
        print(f"  Amount:   ${payout.amount_usd:.2f}")
        print(f"  Status:   {payout.status.value}")
        print(f"  Period:   {payout.period_label}")

    else:
        print(
            "Unknown provider action. "
            "Use: register, status, onboarding, payouts, network, payout"
        )
        storage.close()
        return 1

    storage.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sawyer",
        description="Sawyer — Distributed MoE Inference Network",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # register
    reg_parser = subparsers.add_parser("register", help="Register this machine as a Sawyer node")
    reg_parser.add_argument("--name", default=None, help="Node name (default: hostname)")
    reg_parser.add_argument("--gpu", action="store_true", help="Auto-detect GPU capabilities")
    reg_parser.add_argument(
        "--experts",
        nargs="+",
        help="Expert IDs to host (default: auto-assign)",
    )
    reg_parser.add_argument(
        "--bedrock-url",
        default="https://localhost:8443",
        help="Bedrock Core URL",
    )
    reg_parser.add_argument(
        "--bedrock-key",
        default=None,
        help="Bedrock license key",
    )

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start serving expert inference requests")
    serve_parser.add_argument("--name", default=None, help="Node name")
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8444,
        help="Inference port (default: 8444)",
    )
    serve_parser.add_argument(
        "--router",
        default="https://router.sawyer.dev",
        help="Router endpoint",
    )
    serve_parser.add_argument(
        "--model",
        default=None,
        help="Model to serve (e.g., mixtral-8x7b)",
    )
    serve_parser.add_argument(
        "--backend",
        choices=["subprocess", "http"],
        default="subprocess",
        help="Inference backend mode",
    )
    serve_parser.add_argument(
        "--offline",
        action="store_true",
        help="Start in offline mode (no router connection)",
    )

    # run — one command to start everything
    run_parser = subparsers.add_parser(
        "run",
        help="One command: start Sawyer + Ollama + agent",
    )
    run_parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Model to use (e.g., glm-5.1:cloud). Auto-detects if not specified.",
    )
    run_parser.add_argument(
        "--host",
        default="localhost",
        help="Host for Sawyer router (default: localhost)",
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for Sawyer router (default: 8000)",
    )
    run_parser.add_argument(
        "--agent",
        default="hermes",
        help="Agent to launch (default: hermes). Use --no-agent to skip.",
    )
    run_parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Don't launch an agent, just start Sawyer router",
    )
    run_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )

    # chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Start the consumer chat client (web UI + OpenAI-compatible API)",
    )
    chat_parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    chat_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    chat_parser.add_argument(
        "--ollama-bridge",
        action="store_true",
        help="Bridge local Ollama through Sawyer (serve Ollama to the network)",
    )

    # status
    status_parser = subparsers.add_parser("status", help="Show network status and token balance")
    status_parser.add_argument("--user", default="default", help="User ID for token balance")

    # models
    models_parser = subparsers.add_parser(
        "models", help="List available models and expert layouts"
    )
    models_parser.add_argument(
        "--use",
        choices=["chat", "code"],
        default=None,
        help="Filter models by use case",
    )

    # bench
    bench_parser = subparsers.add_parser(
        "bench",
        help="Benchmark MoE prefill speedup (before/after optimization comparison)",
    )
    bench_parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to GGUF model file",
    )
    bench_parser.add_argument(
        "--binary",
        default=None,
        help="Path to llama-bench binary (default: auto-detect)",
    )
    bench_parser.add_argument(
        "-p",
        "--prompt-sizes",
        default="512,1024,2048",
        help="Comma-separated prompt sizes to benchmark (default: 512,1024,2048)",
    )
    bench_parser.add_argument(
        "-ngl",
        "--ngl",
        type=int,
        default=99,
        help="Number of GPU layers to offload (default: 99, use 0 for CPU-only)",
    )
    bench_parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=-1,
        help="Number of CPU threads (default: auto)",
    )
    bench_parser.add_argument(
        "-r",
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per test (default: 3)",
    )
    bench_parser.add_argument(
        "--json",
        default=None,
        help="Save results as JSON to this path",
    )

    # download
    dl_parser = subparsers.add_parser("download", help="Download model weights to local cache")
    dl_parser.add_argument("model", help="Model to download (e.g., mixtral-8x7b)")
    dl_parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    dl_parser.add_argument("--no-verify", action="store_true", help="Skip SHA-256 verification")

    # fast-llama
    fl_parser = subparsers.add_parser(
        "fast-llama",
        help="Download Sawyer Fast Llama (optimized llama.cpp binary)",
    )
    fl_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already installed",
    )

    # extract
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract per-expert weight shards from a MoE model",
    )
    extract_parser.add_argument(
        "model",
        help="Model to extract experts from (e.g., mixtral-8x7b)",
    )
    extract_parser.add_argument(
        "--experts",
        nargs="+",
        type=int,
        help="Expert IDs to extract (default: all)",
    )
    extract_parser.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Path to GGUF file (default: use Sawyer cache)",
    )
    extract_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ~/.sawyer/experts/{model}/)",
    )

    # account
    acct_parser = subparsers.add_parser("account", help="Manage token accounts")
    acct_sub = acct_parser.add_subparsers(dest="action", help="Account actions")
    acct_create = acct_sub.add_parser("create", help="Create a token account")
    acct_create.add_argument("--user", default="default", help="User ID")
    acct_create.add_argument(
        "--tier",
        choices=["explorer", "builder", "operator"],
        default="explorer",
        help="Subscription tier",
    )
    acct_create.add_argument(
        "--rollover",
        type=int,
        default=0,
        help="Rollover tokens from previous period",
    )
    acct_show = acct_sub.add_parser("show", help="Show account details")
    acct_show.add_argument("--user", default="default", help="User ID")
    acct_sub.add_parser("list", help="List all accounts")

    # quota
    quota_parser = subparsers.add_parser("quota", help="Check token quota")
    quota_parser.add_argument("--user", default="default", help="User ID")
    quota_parser.add_argument("--tokens", type=int, required=True, help="Estimated tokens needed")

    # provider
    prov_parser = subparsers.add_parser(
        "provider", help="Manage node provider registration and payouts"
    )
    prov_sub = prov_parser.add_subparsers(dest="provider_action", help="Provider actions")

    prov_reg = prov_sub.add_parser("register", help="Register as a node provider")
    prov_reg.add_argument("--email", required=True, help="Your email address")
    prov_reg.add_argument("--name", required=True, help="Display name")
    prov_reg.add_argument("--legal-name", default=None, help="Legal name for payouts")
    prov_reg.add_argument("--phone", default=None, help="Phone number")
    prov_reg.add_argument("--country", default="US", help="Country code (default: US)")
    prov_reg.add_argument(
        "--schedule",
        choices=["monthly", "quarterly"],
        default="monthly",
        help="Payout schedule",
    )

    prov_stat = prov_sub.add_parser("status", help="Show provider status and earnings")
    prov_stat.add_argument("provider_id", help="Provider ID")

    prov_onb = prov_sub.add_parser("onboarding", help="Start Stripe Connect onboarding")
    prov_onb.add_argument("provider_id", help="Provider ID")

    prov_pay_hist = prov_sub.add_parser("payouts", help="Show payout history")
    prov_pay_hist.add_argument("provider_id", help="Provider ID")

    prov_pay = prov_sub.add_parser("payout", help="Trigger a payout")
    prov_pay.add_argument("provider_id", help="Provider ID")

    prov_sub.add_parser("network", help="Show network-wide provider stats")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "register": cmd_register,
        "serve": cmd_serve,
        "run": cmd_run,
        "chat": cmd_chat,
        "bench": cmd_bench,
        "status": cmd_status,
        "models": cmd_models,
        "download": cmd_download,
        "fast-llama": cmd_fast_llama,
        "extract": cmd_extract,
        "account": cmd_account,
        "quota": cmd_quota,
        "provider": cmd_provider,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
