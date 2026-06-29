"""Sawyer Provider — Node provider registration, onboarding, and payout management.

A Provider is someone who runs Sawyer nodes and earns money for the compute
they contribute. This module handles:

- Provider registration (contact info, bank details via Stripe Connect)
- KYC/verification status tracking
- Payout account linking (Stripe Connect Express)
- Earnings aggregation and payout scheduling (monthly/quarterly)
- Payout history and tax reporting support (1099-K)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

# Platform takes 30%, provider gets 70%
PLATFORM_FEE_PERCENT = 30.0
PROVIDER_SHARE_PERCENT = 70.0

# Minimum payout amounts
MIN_PAYOUT_USD_MONTHLY = 10.00
MIN_PAYOUT_USD_QUARTERLY = 25.00

# Earnings rate: $0.002 per 1K tokens served
EARNINGS_RATE_PER_1K_TOKENS = 0.002


class ProviderStatus(Enum):
    """Provider verification status."""

    PENDING = "pending"  # Registered, not verified
    ONBOARDING = "onboarding"  # Stripe Connect onboarding started
    VERIFIED = "verified"  # Stripe account verified
    ACTIVE = "active"  # Running nodes, earning
    SUSPENDED = "suspended"  # Suspended by platform
    OFFBOARDED = "offboarded"  # Voluntarily removed


class PayoutSchedule(Enum):
    """How often a provider receives payouts."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class PayoutStatus(Enum):
    """Status of an individual payout."""

    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    PAID = "paid"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Data Models ────────────────────────────────────────────────────


@dataclass
class Provider:
    """A Sawyer node provider who earns money for compute contributed."""

    provider_id: str
    email: str
    display_name: str
    status: ProviderStatus = ProviderStatus.PENDING
    stripe_connect_id: str = ""  # Stripe Connect account ID
    stripe_account_verified: bool = False

    # Contact info
    legal_name: str = ""
    phone: str = ""
    country: str = "US"
    tax_id_provided: bool = False

    # Payout settings
    payout_schedule: PayoutSchedule = PayoutSchedule.MONTHLY
    payout_method: str = "stripe"  # stripe or paypal
    min_payout_usd: float = MIN_PAYOUT_USD_MONTHLY

    # Earnings tracking
    total_tokens_served: int = 0
    total_usd_earned: float = 0.0
    total_usd_paid: float = 0.0
    total_usd_pending: float = 0.0

    # Node association
    node_ids: list[str] = field(default_factory=list)

    # Timestamps
    registered_at: float = field(default_factory=time.time)
    verified_at: float = 0.0
    last_payout_at: float = 0.0
    updated_at: float = field(default_factory=time.time)

    @property
    def available_balance(self) -> float:
        """USD available for next payout."""
        return round(self.total_usd_earned - self.total_usd_paid - self.total_usd_pending, 2)

    @property
    def is_eligible_for_payout(self) -> bool:
        """Whether provider has enough earnings and is verified."""
        return (
            self.status in (ProviderStatus.VERIFIED, ProviderStatus.ACTIVE)
            and self.stripe_account_verified
            and self.available_balance >= self.min_payout_usd
        )


@dataclass
class Payout:
    """A single payout to a provider."""

    payout_id: str
    provider_id: str
    amount_usd: float
    status: PayoutStatus = PayoutStatus.PENDING
    stripe_transfer_id: str = ""
    period_start: float = 0.0
    period_end: float = 0.0
    tokens_in_period: int = 0
    created_at: float = field(default_factory=time.time)
    paid_at: float = 0.0
    failure_reason: str = ""

    @property
    def period_label(self) -> str:
        """Human-readable period label."""
        import datetime

        start = datetime.datetime.fromtimestamp(self.period_start)
        end = datetime.datetime.fromtimestamp(self.period_end)
        return f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"


# ── Provider Manager ───────────────────────────────────────────────


class ProviderManager:
    """Manages provider registration, verification, and payouts.

    Providers register, complete Stripe Connect onboarding, run nodes,
    and receive payouts based on compute contributed.
    """

    def __init__(self, storage: Any = None) -> None:
        self._storage = storage
        self._providers: dict[str, Provider] = {}
        self._payouts: list[Payout] = []
        self._payout_counter = 0

    # ── Registration ─────────────────────────────────────────────

    def register(
        self,
        email: str,
        display_name: str,
        legal_name: str = "",
        phone: str = "",
        country: str = "US",
        payout_schedule: PayoutSchedule = PayoutSchedule.MONTHLY,
    ) -> Provider:
        """Register a new provider.

        Args:
            email: Provider email address
            display_name: Public-facing name
            legal_name: Legal name for tax/payout purposes
            phone: Phone number
            country: ISO country code
            payout_schedule: Monthly or quarterly payouts

        Returns:
            The new Provider object
        """
        provider_id = f"prov_{len(self._providers) + 1:06d}"

        provider = Provider(
            provider_id=provider_id,
            email=email,
            display_name=display_name,
            legal_name=legal_name or display_name,
            phone=phone,
            country=country,
            payout_schedule=payout_schedule,
            min_payout_usd=(
                MIN_PAYOUT_USD_MONTHLY
                if payout_schedule == PayoutSchedule.MONTHLY
                else MIN_PAYOUT_USD_QUARTERLY
            ),
        )

        self._providers[provider_id] = provider

        if self._storage:
            self._storage.append_audit(
                action="provider.register",
                actor_id=provider_id,
                target_id=provider_id,
                silo="provider",
                details={"display_name": display_name, "email": email},
            )

        logger.info(
            "Registered provider %s: %s (%s)",
            provider_id,
            display_name,
            email,
        )
        return provider

    def get_provider(self, provider_id: str) -> Provider | None:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def find_provider_by_email(self, email: str) -> Provider | None:
        """Find a provider by email address."""
        for p in self._providers.values():
            if p.email == email:
                return p
        return None

    def find_provider_by_node(self, node_id: str) -> Provider | None:
        """Find the provider that owns a given node."""
        for p in self._providers.values():
            if node_id in p.node_ids:
                return p
        return None

    def list_providers(self, status: ProviderStatus | None = None) -> list[Provider]:
        """List all providers, optionally filtered by status."""
        providers = list(self._providers.values())
        if status:
            providers = [p for p in providers if p.status == status]
        return providers

    # ── Node Association ─────────────────────────────────────────

    def add_node(self, provider_id: str, node_id: str) -> None:
        """Associate a node with a provider."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")
        if node_id not in provider.node_ids:
            provider.node_ids.append(node_id)
            provider.updated_at = time.time()

    def remove_node(self, provider_id: str, node_id: str) -> None:
        """Disassociate a node from a provider."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")
        if node_id in provider.node_ids:
            provider.node_ids.remove(node_id)
            provider.updated_at = time.time()

    # ── Earnings ──────────────────────────────────────────────────

    def credit_earnings(self, provider_id: str, tokens_served: int) -> float:
        """Credit a provider for tokens served.

        The provider receives 70% of the gross earnings.
        Earnings rate: $0.002 per 1K tokens.

        Args:
            provider_id: Provider to credit
            tokens_served: Number of tokens served

        Returns:
            USD credited to the provider
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")

        gross_earnings = (tokens_served / 1000.0) * EARNINGS_RATE_PER_1K_TOKENS
        provider_share = gross_earnings * (PROVIDER_SHARE_PERCENT / 100.0)

        provider.total_tokens_served += tokens_served
        provider.total_usd_earned += round(provider_share, 6)
        provider.updated_at = time.time()

        logger.info(
            "Credited provider %s: %d tokens → $%.4f (70/30 split)",
            provider_id,
            tokens_served,
            provider_share,
        )
        return round(provider_share, 4)

    # ── Payouts ────────────────────────────────────────────────────

    def process_payout(self, provider_id: str, stripe_transfer_id: str = "") -> Payout | None:
        """Process a payout for an eligible provider.

        Args:
            provider_id: Provider to pay
            stripe_transfer_id: Stripe transfer ID if already created

        Returns:
            Payout object if eligible, None if not
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")

        if not provider.is_eligible_for_payout:
            logger.info(
                "Provider %s not eligible for payout " "(balance=$%.2f, min=$%.2f, verified=%s)",
                provider_id,
                provider.available_balance,
                provider.min_payout_usd,
                provider.stripe_account_verified,
            )
            return None

        self._payout_counter += 1
        payout_amount = provider.available_balance

        provider.total_usd_pending += payout_amount
        provider.updated_at = time.time()

        payout = Payout(
            payout_id=f"pay_{self._payout_counter:08d}",
            provider_id=provider_id,
            amount_usd=payout_amount,
            status=PayoutStatus.PENDING,
            stripe_transfer_id=stripe_transfer_id,
            period_start=provider.last_payout_at or provider.registered_at,
            period_end=time.time(),
            tokens_in_period=0,  # Calculated from records
        )

        self._payouts.append(payout)

        logger.info(
            "Created payout %s for provider %s: $%.2f",
            payout.payout_id,
            provider_id,
            payout_amount,
        )
        return payout

    def mark_payout_paid(self, payout_id: str, stripe_transfer_id: str = "") -> Payout | None:
        """Mark a payout as paid after successful transfer."""
        for payout in self._payouts:
            if payout.payout_id == payout_id:
                payout.status = PayoutStatus.PAID
                payout.paid_at = time.time()
                if stripe_transfer_id:
                    payout.stripe_transfer_id = stripe_transfer_id

                # Update provider totals
                provider = self._providers.get(payout.provider_id)
                if provider:
                    provider.total_usd_paid += payout.amount_usd
                    provider.total_usd_pending -= payout.amount_usd
                    provider.last_payout_at = time.time()

                logger.info(
                    "Payout %s marked paid: $%.2f to %s",
                    payout_id,
                    payout.amount_usd,
                    payout.provider_id,
                )
                return payout
        return None

    def mark_payout_failed(self, payout_id: str, reason: str = "") -> Payout | None:
        """Mark a payout as failed."""
        for payout in self._payouts:
            if payout.payout_id == payout_id:
                payout.status = PayoutStatus.FAILED
                payout.failure_reason = reason

                # Release the pending amount
                provider = self._providers.get(payout.provider_id)
                if provider:
                    provider.total_usd_pending -= payout.amount_usd

                logger.warning("Payout %s failed: %s", payout_id, reason)
                return payout
        return None

    def get_payout_history(self, provider_id: str, limit: int = 50) -> list[Payout]:
        """Get payout history for a provider."""
        return [p for p in reversed(self._payouts) if p.provider_id == provider_id][:limit]

    def get_pending_payouts(self) -> list[Payout]:
        """Get all payouts in PENDING status."""
        return [p for p in self._payouts if p.status == PayoutStatus.PENDING]

    # ── Status Updates ─────────────────────────────────────────────

    def verify_provider(self, provider_id: str, stripe_connect_id: str = "") -> Provider:
        """Mark a provider as verified after Stripe Connect onboarding.

        Args:
            provider_id: Provider to verify
            stripe_connect_id: Stripe Connect account ID

        Returns:
            Updated provider
        """
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")

        provider.status = ProviderStatus.VERIFIED
        provider.stripe_connect_id = stripe_connect_id
        provider.stripe_account_verified = True
        provider.verified_at = time.time()
        provider.updated_at = time.time()

        logger.info(
            "Verified provider %s (stripe: %s)",
            provider_id,
            stripe_connect_id or "N/A",
        )
        return provider

    def activate_provider(self, provider_id: str) -> Provider:
        """Activate a verified provider to start earning."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")
        if provider.status != ProviderStatus.VERIFIED:
            raise ValueError(f"Provider {provider_id} must be verified before activation")

        provider.status = ProviderStatus.ACTIVE
        provider.updated_at = time.time()
        return provider

    def suspend_provider(self, provider_id: str, reason: str = "") -> Provider:
        """Suspend a provider."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")

        provider.status = ProviderStatus.SUSPENDED
        provider.updated_at = time.time()

        if self._storage:
            self._storage.append_audit(
                action="provider.suspend",
                actor_id=provider_id,
                target_id=provider_id,
                silo="provider",
                details={"reason": reason or "suspended"},
            )

        return provider

    def offboard_provider(self, provider_id: str) -> Provider:
        """Offboard a provider (voluntary removal)."""
        provider = self._providers.get(provider_id)
        if provider is None:
            raise ValueError(f"Provider {provider_id} not found")

        provider.status = ProviderStatus.OFFBOARDED
        provider.updated_at = time.time()

        # Process final payout if there's a balance
        if provider.available_balance >= MIN_PAYOUT_USD_MONTHLY:
            self.process_payout(provider_id)

        logger.info("Offboarded provider %s", provider_id)
        return provider

    # ── Statistics ─────────────────────────────────────────────────

    def get_provider_summary(self, provider_id: str) -> dict[str, Any]:
        """Get a summary of a provider's account and earnings."""
        provider = self._providers.get(provider_id)
        if provider is None:
            return {"error": f"Provider {provider_id} not found"}

        payouts = self.get_payout_history(provider_id)
        return {
            "provider_id": provider.provider_id,
            "display_name": provider.display_name,
            "email": provider.email,
            "status": provider.status.value,
            "country": provider.country,
            "nodes": len(provider.node_ids),
            "payout_schedule": provider.payout_schedule.value,
            "earnings": {
                "total_tokens_served": provider.total_tokens_served,
                "total_usd_earned": provider.total_usd_earned,
                "total_usd_paid": provider.total_usd_paid,
                "available_balance": provider.available_balance,
                "pending_payouts": provider.total_usd_pending,
            },
            "verification": {
                "stripe_connected": bool(provider.stripe_connect_id),
                "stripe_verified": provider.stripe_account_verified,
                "tax_id_provided": provider.tax_id_provided,
                "verified_at": provider.verified_at,
            },
            "payouts": {
                "total_count": len(payouts),
                "last_payout_at": provider.last_payout_at,
                "next_payout_eligible": provider.is_eligible_for_payout,
            },
            "registered_at": provider.registered_at,
        }

    def get_network_summary(self) -> dict[str, Any]:
        """Get aggregate statistics across all providers."""
        providers = list(self._providers.values())
        active = [p for p in providers if p.status == ProviderStatus.ACTIVE]
        verified = [
            p for p in providers if p.status in (ProviderStatus.VERIFIED, ProviderStatus.ACTIVE)
        ]

        return {
            "total_providers": len(providers),
            "active_providers": len(active),
            "verified_providers": len(verified),
            "total_nodes": sum(len(p.node_ids) for p in providers),
            "total_tokens_served": sum(p.total_tokens_served for p in providers),
            "total_usd_earned": round(sum(p.total_usd_earned for p in providers), 2),
            "total_usd_paid": round(sum(p.total_usd_paid for p in providers), 2),
            "total_usd_pending": round(sum(p.total_usd_pending for p in providers), 2),
        }
