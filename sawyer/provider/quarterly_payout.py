"""Sawyer Quarterly Payout — distribute the provider pool and pay providers.

Pool calculation:
    total_subscribers x $5 x 0.70 = provider_pool

Distribution:
    Each provider's share = (their weighted_contribution / total_weighted) x pool
    Weighted contribution = tokens_served x tier_multiplier

Payout schedule:
    Quarterly: Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec
    Minimum payout: $25 (providers below threshold roll over to next quarter)
    Payout method: Stripe Connect or PayPal

The pool is fixed per quarter. More subscribers = bigger pool = more money
for everyone. The kid with the 4090 doing most of the work gets the
biggest slice. The laptop kid still earns, just less.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sawyer.provider.node_tiers import (
    NodeTier,
    compute_weighted_contribution,
    get_tier_config,
)
from sawyer.provider.revenue_pool import RevenuePool

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────

SUBSCRIPTION_PRICE = 5.00  # $5/month
PROVIDER_SHARE = 0.70  # 70% to providers
MIN_PAYOUT_USD = 25.00  # Minimum payout per quarter
QUARTERS = {
    1: ("Q1", "January-March"),
    2: ("Q2", "April-June"),
    3: ("Q3", "July-September"),
    4: ("Q4", "October-December"),
}


class PayoutMethod(Enum):
    """How a provider receives their payout."""

    STRIPE = "stripe"
    PAYPAL = "paypal"


class PayoutStatus(Enum):
    """Status of a quarterly payout."""

    PENDING = "pending"
    PROCESSING = "processing"
    PAID = "paid"
    FAILED = "failed"
    ROLLED_OVER = "rolled_over"  # Below minimum, rolled to next quarter


@dataclass
class ProviderPayout:
    """One provider's quarterly payout."""

    payout_id: str
    provider_id: str
    period_id: str
    quarter: str  # e.g. "Q2 2026"
    quarter_label: str  # e.g. "April-June"

    # Pool context
    total_subscribers: int
    total_revenue_usd: float
    provider_pool_usd: float

    # This provider's contribution
    tokens_served: int
    uptime_hours: float
    tier: str
    multiplier: float
    weighted_contribution: float
    share_percent: float

    # Earnings
    token_earnings_usd: float
    uptime_earnings_usd: float
    total_earnings_usd: float

    # Payout
    status: PayoutStatus = PayoutStatus.PENDING
    payout_method: PayoutMethod = PayoutMethod.STRIPE
    stripe_transfer_id: str = ""
    paypal_payout_id: str = ""
    paid_at: float = 0.0
    failure_reason: str = ""


class QuarterlyPayout:
    """Manages quarterly payout distribution.

    Each quarter:
    1. Count total subscribers
    2. Calculate provider pool = subscribers x $5/month x 3 months x 70%
    3. Distribute pool proportionally by weighted contribution
    4. Queue payouts above $25 minimum
    5. Roll over balances below $25 to next quarter
    """

    def __init__(self) -> None:
        self._revenue_pool = RevenuePool()
        self._payouts: list[ProviderPayout] = []
        self._payout_counter = 0
        self._rollover: dict[str, float] = {}  # provider_id -> rolled over USD

    def calculate_quarterly_pool(
        self,
        total_subscribers: int,
        months_in_quarter: int = 3,
    ) -> dict[str, Any]:
        """Calculate the provider pool for a quarter.

        Args:
            total_subscribers: Number of active subscribers this quarter
            months_in_quarter: Months in the quarter (default: 3)

        Returns:
            Pool breakdown dict
        """
        total_revenue = total_subscribers * SUBSCRIPTION_PRICE * months_in_quarter
        provider_pool = total_revenue * PROVIDER_SHARE
        platform_pool = total_revenue * (1 - PROVIDER_SHARE)

        return {
            "total_subscribers": total_subscribers,
            "months_in_quarter": months_in_quarter,
            "subscription_price": SUBSCRIPTION_PRICE,
            "total_revenue_usd": round(total_revenue, 2),
            "provider_pool_usd": round(provider_pool, 2),
            "platform_pool_usd": round(platform_pool, 2),
            "provider_share_percent": PROVIDER_SHARE * 100,
            "per_subscriber_per_month": SUBSCRIPTION_PRICE,
            "per_subscriber_per_quarter": round(SUBSCRIPTION_PRICE * months_in_quarter, 2),
        }

    def distribute_quarterly(
        self,
        total_subscribers: int,
        provider_stats: list[dict[str, Any]],
        quarter: int | None = None,
        year: int | None = None,
    ) -> list[ProviderPayout]:
        """Distribute the quarterly provider pool to all providers.

        Args:
            total_subscribers: Active subscribers this quarter
            provider_stats: List of dicts with:
                - provider_id: str
                - tokens_served: int
                - uptime_hours: float
                - vram_gb: float (for tier classification)
                - payout_method: str ("stripe" or "paypal")
            quarter: Quarter number 1-4 (default: current)
            year: Year (default: current)

        Returns:
            List of ProviderPayout objects
        """
        # Determine quarter
        now = time.localtime()
        if quarter is None:
            quarter = (now.tm_mon - 1) // 3 + 1
        if year is None:
            year = now.tm_year

        quarter_name, quarter_label = QUARTERS[quarter]
        period_label = f"{quarter_name} {year}"

        # Calculate pool
        pool_info = self.calculate_quarterly_pool(total_subscribers)
        provider_pool = pool_info["provider_pool_usd"]

        # Open a revenue pool period
        period = self._revenue_pool.open_period(
            revenue_usd=pool_info["total_revenue_usd"],
        )

        # Calculate weighted contributions
        total_weighted = 0.0
        total_uptime = 0.0
        provider_data = []

        for stats in provider_stats:
            vram = stats.get("vram_gb", 4.0)
            tier = NodeTier.from_vram(vram)
            config = get_tier_config(tier)
            tokens = stats.get("tokens_served", 0)
            uptime = stats.get("uptime_hours", 0.0)
            weighted = compute_weighted_contribution(tokens, vram)

            total_weighted += weighted
            total_uptime += uptime

            provider_data.append({
                "provider_id": stats["provider_id"],
                "tokens_served": tokens,
                "uptime_hours": uptime,
                "vram_gb": vram,
                "tier": tier,
                "config": config,
                "weighted": weighted,
            })

        # Distribute pool proportionally
        # 90% by weighted throughput, 10% by uptime
        throughput_pool = provider_pool * 0.90
        uptime_pool = provider_pool * 0.10

        payouts = []
        total_distributed = 0.0

        for pd in provider_data:
            # Token earnings: weighted share of throughput pool
            if total_weighted > 0:
                token_share = (pd["weighted"] / total_weighted) * throughput_pool
            else:
                token_share = 0.0

            # Uptime earnings: proportional share of uptime pool
            if total_uptime > 0:
                uptime_share = (pd["uptime_hours"] / total_uptime) * uptime_pool
            else:
                uptime_share = 0.0

            # Add any rollover from previous quarters
            rollover = self._rollover.get(pd["provider_id"], 0.0)

            total_earnings = token_share + uptime_share + rollover
            share_pct = (pd["weighted"] / total_weighted * 100) if total_weighted > 0 else 0

            self._payout_counter += 1
            payout = ProviderPayout(
                payout_id=f"qpay-{self._payout_counter:06d}",
                provider_id=pd["provider_id"],
                period_id=period.period_id,
                quarter=period_label,
                quarter_label=quarter_label,
                total_subscribers=total_subscribers,
                total_revenue_usd=pool_info["total_revenue_usd"],
                provider_pool_usd=provider_pool,
                tokens_served=pd["tokens_served"],
                uptime_hours=pd["uptime_hours"],
                tier=pd["tier"].value,
                multiplier=pd["config"].multiplier,
                weighted_contribution=pd["weighted"],
                share_percent=round(share_pct, 2),
                token_earnings_usd=round(token_share, 4),
                uptime_earnings_usd=round(uptime_share, 4),
                total_earnings_usd=round(total_earnings, 4),
                payout_method=PayoutMethod(
                    pd.get("payout_method", "stripe")
                ),
            )

            # Check minimum payout threshold
            if total_earnings < MIN_PAYOUT_USD:
                payout.status = PayoutStatus.ROLLED_OVER
                # Roll over to next quarter
                self._rollover[pd["provider_id"]] = total_earnings
                logger.info(
                    "Rolled over $%.2f for %s (below $%.2f minimum)",
                    total_earnings,
                    pd["provider_id"],
                    MIN_PAYOUT_USD,
                )
            else:
                # Clear any previous rollover since it's included
                if pd["provider_id"] in self._rollover:
                    del self._rollover[pd["provider_id"]]

            payouts.append(payout)
            total_distributed += payout.total_earnings_usd

            logger.info(
                "Quarterly payout for %s: $%.2f (%s, %s, %.1f%% share)",
                pd["provider_id"],
                payout.total_earnings_usd,
                pd["tier"].value,
                pd["config"].description,
                share_pct,
            )

        # Close the revenue pool period
        self._revenue_pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {
                    "provider_id": pd["provider_id"],
                    "tokens_served": pd["tokens_served"],
                    "uptime_hours": pd["uptime_hours"],
                }
                for pd in provider_data
            ],
        )

        self._payouts.extend(payouts)

        logger.info(
            "Quarterly distribution complete: %d providers, "
            "$%.2f distributed, $%.2f rolled over",
            len(payouts),
            total_distributed,
            sum(self._rollover.values()),
        )

        return payouts

    def get_payout(self, payout_id: str) -> ProviderPayout | None:
        """Get a payout by ID."""
        for p in self._payouts:
            if p.payout_id == payout_id:
                return p
        return None

    def get_provider_payouts(
        self,
        provider_id: str,
    ) -> list[ProviderPayout]:
        """Get all payouts for a provider."""
        return [p for p in self._payouts if p.provider_id == provider_id]

    def get_pending_payouts(self) -> list[ProviderPayout]:
        """Get all payouts ready to be processed (above minimum, not yet paid)."""
        return [
            p for p in self._payouts
            if p.status == PayoutStatus.PENDING
            and p.total_earnings_usd >= MIN_PAYOUT_USD
        ]

    def get_rolled_over(self) -> dict[str, float]:
        """Get current rollover balances."""
        return dict(self._rollover)

    def mark_paid(
        self,
        payout_id: str,
        stripe_transfer_id: str = "",
        paypal_payout_id: str = "",
    ) -> ProviderPayout | None:
        """Mark a payout as paid after successful transfer."""
        payout = self.get_payout(payout_id)
        if payout is None:
            return None

        payout.status = PayoutStatus.PAID
        payout.paid_at = time.time()
        payout.stripe_transfer_id = stripe_transfer_id
        payout.paypal_payout_id = paypal_payout_id

        logger.info(
            "Marked payout %s as paid: $%.2f to %s",
            payout_id,
            payout.total_earnings_usd,
            payout.provider_id,
        )
        return payout

    def mark_failed(
        self,
        payout_id: str,
        reason: str = "",
    ) -> ProviderPayout | None:
        """Mark a payout as failed. Earnings roll back to next quarter."""
        payout = self.get_payout(payout_id)
        if payout is None:
            return None

        payout.status = PayoutStatus.FAILED
        payout.failure_reason = reason

        # Roll the earnings back for next quarter
        self._rollover[payout.provider_id] = (
            self._rollover.get(payout.provider_id, 0.0)
            + payout.total_earnings_usd
        )

        logger.warning(
            "Payout %s failed: %s. $%.2f rolled over to next quarter for %s",
            payout_id,
            reason,
            payout.total_earnings_usd,
            payout.provider_id,
        )
        return payout

    def get_quarterly_summary(
        self,
        total_subscribers: int,
        provider_stats: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Preview a quarterly distribution without creating payouts.

        Useful for showing providers what they'll earn before payout.
        """
        pool_info = self.calculate_quarterly_pool(total_subscribers)

        total_weighted = 0.0
        total_uptime = 0.0
        previews = []

        for stats in provider_stats:
            vram = stats.get("vram_gb", 4.0)
            tokens = stats.get("tokens_served", 0)
            uptime = stats.get("uptime_hours", 0.0)
            weighted = compute_weighted_contribution(tokens, vram)
            tier = NodeTier.from_vram(vram)
            config = get_tier_config(tier)

            total_weighted += weighted
            total_uptime += uptime

            previews.append({
                "provider_id": stats["provider_id"],
                "tier": tier.value,
                "multiplier": config.multiplier,
                "tokens_served": tokens,
                "uptime_hours": uptime,
                "weighted_contribution": weighted,
            })

        # Calculate each provider's share
        throughput_pool = pool_info["provider_pool_usd"] * 0.90
        uptime_pool = pool_info["provider_pool_usd"] * 0.10

        for preview in previews:
            if total_weighted > 0:
                token_share = (preview["weighted_contribution"] / total_weighted) * throughput_pool
                share_pct = (preview["weighted_contribution"] / total_weighted) * 100
            else:
                token_share = 0.0
                share_pct = 0.0

            if total_uptime > 0:
                uptime_share = (preview["uptime_hours"] / total_uptime) * uptime_pool
            else:
                uptime_share = 0.0

            preview["share_percent"] = round(share_pct, 2)
            preview["token_earnings_usd"] = round(token_share, 4)
            preview["uptime_earnings_usd"] = round(uptime_share, 4)
            preview["total_earnings_usd"] = round(token_share + uptime_share, 4)

        return {
            **pool_info,
            "total_weighted_contribution": total_weighted,
            "total_uptime_hours": total_uptime,
            "providers": previews,
        }