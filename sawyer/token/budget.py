"""Sawyer Token Economics — subscription tiers, budgets, and accounting."""

from dataclasses import dataclass
from enum import Enum


class SubscriptionTier(Enum):
    """Sawyer subscription tiers."""

    EXPLORER = "explorer"  # Free trial — unlimited tokens
    PRO = "pro"  # $15/mo — 2M tokens
    PIONEER = "pioneer"  # $40/mo — 5M tokens
    ENTERPRISE = "enterprise"  # $100/mo — 25M tokens


# Token budgets per tier (monthly)
TIER_TOKENS = {
    SubscriptionTier.EXPLORER: 0,  # Unlimited during trial
    SubscriptionTier.PRO: 2_000_000,
    SubscriptionTier.PIONEER: 5_000_000,
    SubscriptionTier.ENTERPRISE: 25_000_000,
}

TIER_PRICING = {
    SubscriptionTier.EXPLORER: 0,  # Free trial
    SubscriptionTier.PRO: 15,
    SubscriptionTier.PIONEER: 40,
    SubscriptionTier.ENTERPRISE: 100,
}

# Maximum token rollover (1 month's worth)
MAX_ROLLOVER = {
    SubscriptionTier.EXPLORER: 0,
    SubscriptionTier.PRO: 2_000_000,
    SubscriptionTier.PIONEER: 5_000_000,
    SubscriptionTier.ENTERPRISE: 25_000_000,
}


@dataclass
class TokenBalance:
    """User token balance with rollover support."""

    tier: SubscriptionTier
    monthly_budget: int
    current_balance: int
    rollover: int = 0

    @property
    def total_available(self) -> int:
        """Total tokens available including rollover."""
        return self.current_balance + self.rollover

    def debit(self, tokens: int) -> int:
        """Debit tokens from the balance. Returns remaining balance.

        Uses rollover first, then monthly budget.
        """
        if tokens > self.total_available:
            raise ValueError(f"Insufficient tokens: need {tokens}, have {self.total_available}")

        # Debit from rollover first
        if self.rollover >= tokens:
            self.rollover -= tokens
            return self.total_available

        # Debit remaining from current balance
        remaining = tokens - self.rollover
        self.rollover = 0
        self.current_balance -= remaining
        return self.total_available

    def credit_rollover(self) -> None:
        """Roll over unused monthly tokens (up to max)."""
        max_rollover = MAX_ROLLOVER[self.tier]
        self.rollover = min(self.current_balance, max_rollover)
        self.current_balance = self.monthly_budget


@dataclass
class HostEarnings:
    """Earnings for a node hosting experts."""

    node_id: str
    tokens_served: int
    credits_earned: float
    usd_earned: float
    payout_threshold: float = 10.0  # Minimum USD for payout

    @property
    def eligible_for_payout(self) -> bool:
        """Check if earnings exceed the payout threshold."""
        return self.usd_earned >= self.payout_threshold
