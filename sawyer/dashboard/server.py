"""Sawyer Dashboard — FastAPI web interface for cluster status.

Provides:
- GET / — Cluster overview (nodes, health, utilization)
- GET /nodes — List all registered nodes
- GET /nodes/{node_id} — Node detail
- GET /accounts — List all token accounts
- GET /accounts/{user_id} — Account detail and usage
- GET /inference/history/{user_id} — Inference history for a user
- GET /providers — List all providers
- GET /providers/{provider_id} — Provider detail + earnings
- POST /providers/register — Register a new provider
- POST /providers/{provider_id}/onboarding — Start Stripe Connect onboarding
- GET /providers/{provider_id}/verification — Check verification status
- GET /providers/{provider_id}/payouts — Payout history
- POST /providers/{provider_id}/payout — Trigger a payout
- GET /providers/network/summary — Network-wide provider stats
- GET /health — Health check
- GET /stats — Aggregate statistics
- GET /audit — Audit log query
"""

import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request

from sawyer.auth.api import InvalidAPIKey, SawyerAuth
from sawyer.provider.manager import PayoutSchedule, ProviderManager, ProviderStatus
from sawyer.router.scheduler import ExpertScheduler
from sawyer.storage.database import SawyerStorage
from sawyer.token.budget import SubscriptionTier

# ── App Factory ────────────────────────────────────────────────────

_default_storage: SawyerStorage | None = None
_default_provider_mgr: ProviderManager | None = None


def create_app(
    storage: SawyerStorage | None = None,
    provider_mgr: ProviderManager | None = None,
) -> FastAPI:
    """Create a FastAPI app with optional storage injection."""
    global _default_storage, _default_provider_mgr
    _default_storage = storage
    _default_provider_mgr = provider_mgr

    api = FastAPI(
        title="Sawyer Dashboard",
        description="Distributed MoE Inference Network — Cluster Status API",
        version="0.1.0",
    )

    # Register all routes
    api.add_api_route("/", cluster_overview, methods=["GET"])
    api.add_api_route("/health", health_check, methods=["GET"])
    api.add_api_route("/nodes", list_nodes, methods=["GET"])
    api.add_api_route("/nodes/{node_id}", get_node, methods=["GET"])
    api.add_api_route("/accounts", list_accounts, methods=["GET"])
    api.add_api_route("/accounts/{user_id}", get_account, methods=["GET"])
    api.add_api_route("/inference/history/{user_id}", inference_history, methods=["GET"])
    api.add_api_route("/providers", list_providers, methods=["GET"])
    api.add_api_route("/providers/{provider_id}", get_provider, methods=["GET"])
    api.add_api_route("/providers/register", register_provider, methods=["POST"])
    api.add_api_route(
        "/providers/{provider_id}/onboarding",
        start_onboarding,
        methods=["POST"],
    )
    api.add_api_route(
        "/providers/{provider_id}/verification",
        check_verification,
        methods=["GET"],
    )
    api.add_api_route(
        "/providers/{provider_id}/payouts",
        provider_payout_history,
        methods=["GET"],
    )
    api.add_api_route("/providers/{provider_id}/payout", trigger_payout, methods=["POST"])
    api.add_api_route("/providers/network/summary", network_summary, methods=["GET"])
    api.add_api_route("/stats", aggregate_stats, methods=["GET"])
    api.add_api_route("/audit", audit_log, methods=["GET"])

    return api


# ── Dependencies ──────────────────────────────────────────────────


def _get_storage() -> SawyerStorage:
    """Get the storage instance."""
    if _default_storage is not None:
        return _default_storage
    from sawyer.config import SawyerConfig

    return SawyerStorage(SawyerConfig().database_url)


def _get_auth() -> SawyerAuth:
    """Get or create a SawyerAuth instance."""
    return SawyerAuth(_get_storage())


def _get_scheduler() -> ExpertScheduler:
    """Get or create an ExpertScheduler instance."""
    return ExpertScheduler()


async def verify_api_key(request: Request) -> Any:
    """Verify API key from X-API-Key header or api_key query param."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        api_key = request.query_params.get("api_key")

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. " "Pass X-API-Key header or api_key query param.",
        )

    auth = _get_auth()
    try:
        return auth.validate_key(api_key)
    except InvalidAPIKey as e:
        raise HTTPException(status_code=401, detail=str(e)) from None


# ── Routes ─────────────────────────────────────────────────────────


async def cluster_overview() -> dict[str, Any]:
    """Cluster overview: node count, health, utilization."""
    storage = _get_storage()
    scheduler = _get_scheduler()
    status = scheduler.get_cluster_status()
    accounts = storage.list_accounts()
    nodes = storage.list_nodes()

    total_tokens_used = sum(a.total_tokens_used for a in accounts)
    active_accounts = sum(1 for a in accounts if a.is_active)

    return {
        "cluster": status,
        "nodes": {
            "registered": len(nodes),
            "healthy": sum(1 for n in nodes if n.get("healthy", False)),
        },
        "accounts": {
            "total": len(accounts),
            "active": active_accounts,
        },
        "tokens": {
            "total_used": total_tokens_used,
        },
        "timestamp": time.time(),
    }


async def health_check() -> dict[str, str]:
    """Simple health check."""
    return {"status": "healthy", "service": "sawyer-dashboard"}


async def list_nodes(api_key=Depends(verify_api_key)) -> list[dict[str, Any]]:
    """List all registered nodes."""
    storage = _get_storage()
    return storage.list_nodes()


async def get_node(node_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Get details for a specific node."""
    storage = _get_storage()
    node = storage.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return node


async def list_accounts(api_key=Depends(verify_api_key)) -> list[dict[str, Any]]:
    """List all token accounts."""
    storage = _get_storage()
    accounts = storage.list_accounts()
    return [
        {
            "user_id": a.user_id,
            "tier": a.tier.value,
            "balance": a.balance.total_available,
            "tokens_used": a.total_tokens_used,
            "inferences": a.total_inferences,
            "is_active": a.is_active,
        }
        for a in accounts
    ]


async def get_account(user_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Get account details and usage summary."""
    storage = _get_storage()
    account = storage.load_account(user_id)
    if account is None:
        raise HTTPException(status_code=404, detail=f"Account {user_id} not found")

    return {
        "user_id": account.user_id,
        "tier": account.tier.value,
        "balance": {
            "monthly_budget": account.balance.monthly_budget,
            "current_balance": account.balance.current_balance,
            "rollover": account.balance.rollover,
            "total_available": account.balance.total_available,
        },
        "usage": {
            "tokens_used": account.total_tokens_used,
            "inferences": account.total_inferences,
            "last_inference_at": account.last_inference_at,
        },
        "is_active": account.is_active,
    }


async def inference_history(
    user_id: str,
    limit: int = 50,
    api_key=Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Get inference history for a user."""
    storage = _get_storage()
    records = storage.get_inference_records(user_id, limit=limit)
    return [
        {
            "record_id": r.record_id,
            "model": r.model_name,
            "expert_ids": r.expert_ids,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "total_tokens": r.total_tokens,
            "latency_ms": r.latency_ms,
            "node_id": r.node_id,
            "routing_strategy": r.routing_strategy,
            "timestamp": r.timestamp,
        }
        for r in records
    ]


async def aggregate_stats(api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Aggregate cluster statistics."""
    storage = _get_storage()
    accounts = storage.list_accounts()
    nodes = storage.list_nodes()

    total_tokens = sum(a.total_tokens_used for a in accounts)
    total_inferences = sum(a.total_inferences for a in accounts)

    return {
        "nodes": {
            "total": len(nodes),
            "healthy": sum(1 for n in nodes if n.get("healthy", False)),
            "total_vram_gb": sum(n.get("vram_gb", 0) for n in nodes),
        },
        "accounts": {
            "total": len(accounts),
            "by_tier": {
                tier.value: sum(1 for a in accounts if a.tier == tier) for tier in SubscriptionTier
            },
        },
        "tokens": {
            "total_used": total_tokens,
            "total_inferences": total_inferences,
        },
        "timestamp": time.time(),
    }


async def audit_log(
    actor_id: str | None = None,
    action: str | None = None,
    limit: int = 100,
    api_key=Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Query the audit log."""
    storage = _get_storage()
    return storage.query_audit(actor_id=actor_id, action=action, limit=limit)


# ── Provider Routes ────────────────────────────────────────────────


def _get_provider_mgr() -> ProviderManager:
    """Get or create a ProviderManager instance."""
    if _default_provider_mgr is not None:
        return _default_provider_mgr
    return ProviderManager(storage=_get_storage())


async def list_providers(
    status: str | None = None,
    api_key=Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """List all providers, optionally filtered by status."""
    mgr = _get_provider_mgr()
    provider_status = None
    if status:
        try:
            provider_status = ProviderStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}") from None

    providers = mgr.list_providers(status=provider_status)
    return [
        {
            "provider_id": p.provider_id,
            "display_name": p.display_name,
            "email": p.email,
            "status": p.status.value,
            "country": p.country,
            "nodes": len(p.node_ids),
            "total_usd_earned": p.total_usd_earned,
            "available_balance": p.available_balance,
            "payout_schedule": p.payout_schedule.value,
            "registered_at": p.registered_at,
        }
        for p in providers
    ]


async def get_provider(provider_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Get a provider's full details including earnings."""
    mgr = _get_provider_mgr()
    provider = mgr.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    return mgr.get_provider_summary(provider_id)


async def register_provider(request: Request) -> dict[str, Any]:
    """Register a new node provider."""
    body = await request.json()
    mgr = _get_provider_mgr()

    email = body.get("email")
    display_name = body.get("display_name")
    if not email or not display_name:
        raise HTTPException(status_code=400, detail="email and display_name are required")

    # Check for duplicate email
    existing = mgr.find_provider_by_email(email)
    if existing:
        raise HTTPException(status_code=409, detail=f"Provider already registered: {email}")

    payout_schedule = PayoutSchedule.MONTHLY
    if body.get("payout_schedule") == "quarterly":
        payout_schedule = PayoutSchedule.QUARTERLY

    provider = mgr.register(
        email=email,
        display_name=display_name,
        legal_name=body.get("legal_name", ""),
        phone=body.get("phone", ""),
        country=body.get("country", "US"),
        payout_schedule=payout_schedule,
    )

    return {
        "provider_id": provider.provider_id,
        "status": provider.status.value,
        "message": "Provider registered. Complete Stripe Connect onboarding to start earning.",
    }


async def start_onboarding(provider_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Start Stripe Connect onboarding for a provider."""
    mgr = _get_provider_mgr()
    provider = mgr.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    # In production, this calls SawyerProviderStripe.create_connect_account()
    # For now, mark as onboarding
    provider.status = ProviderStatus.ONBOARDING
    provider.updated_at = time.time()

    return {
        "provider_id": provider_id,
        "status": provider.status.value,
        "message": "Stripe Connect onboarding initiated. "
        "Provider will receive an onboarding link via email.",
        "next_step": "GET /providers/{provider_id}/verification to check status",
    }


async def check_verification(provider_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Check a provider's Stripe Connect verification status."""
    mgr = _get_provider_mgr()
    provider = mgr.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    return {
        "provider_id": provider_id,
        "status": provider.status.value,
        "stripe_connected": bool(provider.stripe_connect_id),
        "stripe_verified": provider.stripe_account_verified,
        "tax_id_provided": provider.tax_id_provided,
        "is_eligible_for_payout": provider.is_eligible_for_payout,
        "available_balance": provider.available_balance,
    }


async def provider_payout_history(
    provider_id: str,
    limit: int = 50,
    api_key=Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Get payout history for a provider."""
    mgr = _get_provider_mgr()
    provider = mgr.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    payouts = mgr.get_payout_history(provider_id, limit=limit)
    return [
        {
            "payout_id": p.payout_id,
            "amount_usd": p.amount_usd,
            "status": p.status.value,
            "period": p.period_label,
            "tokens_in_period": p.tokens_in_period,
            "created_at": p.created_at,
            "paid_at": p.paid_at,
        }
        for p in payouts
    ]


async def trigger_payout(provider_id: str, api_key=Depends(verify_api_key)) -> dict[str, Any]:
    """Trigger a payout for an eligible provider."""
    mgr = _get_provider_mgr()
    provider = mgr.get_provider(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")

    if not provider.is_eligible_for_payout:
        return {
            "provider_id": provider_id,
            "eligible": False,
            "reason": (
                f"Balance ${provider.available_balance:.2f} below "
                f"minimum ${provider.min_payout_usd:.2f} or account not verified"
            ),
        }

    payout = mgr.process_payout(provider_id)
    if payout is None:
        return {
            "provider_id": provider_id,
            "eligible": False,
            "reason": "Could not process payout",
        }

    return {
        "provider_id": provider_id,
        "eligible": True,
        "payout_id": payout.payout_id,
        "amount_usd": payout.amount_usd,
        "status": payout.status.value,
        "period": payout.period_label,
    }


async def network_summary(
    api_key=Depends(verify_api_key),
) -> dict[str, Any]:
    """Get network-wide provider statistics."""
    mgr = _get_provider_mgr()
    return mgr.get_network_summary()


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the dashboard server."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)


# Default app instance for uvicorn
app = create_app()
