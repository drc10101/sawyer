"""Sawyer CLI — Register nodes, serve experts, check status, manage subscriptions.

Usage:
    sawyer register       Register this machine as a Sawyer node
    sawyer serve          Start serving expert inference requests
    sawyer status         Show network status and token balance
    sawyer models         List available models and expert layouts
    sawyer download       Download model weights to local cache
    sawyer account        Create or show token account
    sawyer quota          Check token quota before inference
    sawyer provider       Manage node provider registration and payouts
"""

import argparse
import asyncio
import sys

import grpc

from sawyer.config import SawyerConfig
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
            print(f"  Available: {', '.join(m['name'] for m in list_models())}")
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
        print(f"\n  The Sawyer router is not yet available.")
        print(f"  Start in offline mode: sawyer serve --offline")
        print(f"  Or check your connection and try again.")
        return 1
    return 0


def cmd_chat(args) -> int:
    """Start the consumer chat client — web UI + OpenAI-compatible API.

    This is what the user with the 8GB laptop runs. Opens a web UI at
    localhost:8000 where they can chat with Sawyer-powered models.
    Also provides an OpenAI-compatible API so any tool (curl, OpenAI SDK,
    Ollama clients) can point at it.
    """
    from sawyer.client import serve_client

    print()
    print("  Sawyer — Distributed MoE Inference")
    print("  Cheaper than your provider. The load is split, friends help.")
    print()

    try:
        serve_client(host=args.host, port=args.port)
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
    models = list_models()
    print("Available Models")
    print("=" * 60)

    for m in models:
        print(f"\n  {m['display_name']} ({m['name']})")
        print(f"    Experts: {m['num_experts']} total, {m['active_experts']} active")
        print(
            f"    Parameters: {m['total_params_B']:.1f}B"
            f" total, {m['active_params_B']:.1f}B active"
        )
        print(f"    Q4 memory: {m['model_size_gb_q4']:.1f} GB")
        print(f"    Min VRAM: {m['min_vram_gb']:.0f} GB")

    print(f"\n  {len(models)} models available")
    return 0


def cmd_download(args) -> int:
    """Download model weights to local cache."""
    config = SawyerConfig()
    model_name = args.model

    try:
        model = get_model(model_name)
    except ValueError:
        print(f"Unknown model: {model_name}")
        print(f"Available: {', '.join(m['name'] for m in list_models())}")
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
        size_mb = path.stat().st_size / (1024 ** 2)
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
        print("\nNext step: Complete Stripe Connect onboarding")
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
                    f"  {p.payout_id}  ${p.amount_usd:.2f}  "
                    f"{p.status.value}  {p.period_label}"
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

    # status
    status_parser = subparsers.add_parser("status", help="Show network status and token balance")
    status_parser.add_argument("--user", default="default", help="User ID for token balance")

    # models
    subparsers.add_parser("models", help="List available models and expert layouts")

    # download
    dl_parser = subparsers.add_parser("download", help="Download model weights to local cache")
    dl_parser.add_argument("model", help="Model to download (e.g., mixtral-8x7b)")
    dl_parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    dl_parser.add_argument("--no-verify", action="store_true", help="Skip SHA-256 verification")

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
    prov_sub = prov_parser.add_subparsers(
        dest="provider_action", help="Provider actions"
    )

    prov_reg = prov_sub.add_parser("register", help="Register as a node provider")
    prov_reg.add_argument("--email", required=True, help="Your email address")
    prov_reg.add_argument("--name", required=True, help="Display name")
    prov_reg.add_argument("--legal-name", default=None, help="Legal name for payouts")
    prov_reg.add_argument("--phone", default=None, help="Phone number")
    prov_reg.add_argument(
        "--country", default="US", help="Country code (default: US)"
    )
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
        "chat": cmd_chat,
        "status": cmd_status,
        "models": cmd_models,
        "download": cmd_download,
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
