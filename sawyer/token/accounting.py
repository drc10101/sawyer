"""Sawyer Token Accounting — tracks inference usage and debits token budgets.

Connects the inference pipeline to the token budget system:
- Records every inference request with token counts
- Debits the user's token budget per request
- Handles rollover logic at billing cycle boundaries
- Provides usage summaries and quota enforcement
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sawyer.token.budget import (
    MAX_ROLLOVER,
    TIER_TOKENS,
    HostEarnings,
    SubscriptionTier,
    TokenBalance,
)

logger = logging.getLogger(__name__)


class AccountingError(Exception):
    """Raised when token accounting fails."""

    pass


class InsufficientTokens(AccountingError):
    """Raised when a user doesn't have enough tokens for an inference request."""

    pass


@dataclass
class InferenceRecord:
    """Record of a single inference request."""

    record_id: str
    user_id: str
    model_name: str
    expert_ids: list[int]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    timestamp: float
    node_id: str = ""
    routing_strategy: str = ""
    finish_reason: str = "stop"


@dataclass
class UserAccount:
    """A user's token account with balance and usage history."""

    user_id: str
    tier: SubscriptionTier
    balance: TokenBalance
    records: list[InferenceRecord] = field(default_factory=list)
    total_tokens_used: int = 0
    total_inferences: int = 0
    created_at: float = field(default_factory=time.time)
    last_inference_at: float = 0.0

    @property
    def is_active(self) -> bool:
        """Whether the user has tokens available."""
        return self.balance.total_available > 0


class TokenAccountant:
    """Manages token accounting across all users.

    Tracks inference usage, debits token budgets, handles rollover,
    and provides usage summaries.
    """

    def __init__(self) -> None:
        self._accounts: dict[str, UserAccount] = {}
        self._host_earnings: dict[str, HostEarnings] = {}
        self._record_counter = 0

    def create_account(
        self,
        user_id: str,
        tier: SubscriptionTier,
        rollover: int = 0,
    ) -> UserAccount:
        """Create a new user account with a token budget.

        Args:
            user_id: Unique user identifier
            tier: Subscription tier determining token budget
            rollover: Rollover tokens from previous billing period

        Returns:
            The new UserAccount
        """
        if user_id in self._accounts:
            raise AccountingError(f"Account already exists for user {user_id}")

        monthly_budget = TIER_TOKENS[tier]
        balance = TokenBalance(
            tier=tier,
            monthly_budget=monthly_budget,
            current_balance=monthly_budget,
            rollover=min(rollover, MAX_ROLLOVER[tier]),
        )

        account = UserAccount(
            user_id=user_id,
            tier=tier,
            balance=balance,
        )
        self._accounts[user_id] = account
        logger.info(
            "Created account for %s: tier=%s budget=%d rollover=%d",
            user_id,
            tier.value,
            monthly_budget,
            rollover,
        )
        return account

    def get_account(self, user_id: str) -> UserAccount | None:
        """Get a user's account, or None if not found."""
        return self._accounts.get(user_id)

    def record_inference(
        self,
        user_id: str,
        model_name: str,
        expert_ids: list[int],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        node_id: str = "",
        routing_strategy: str = "",
        finish_reason: str = "stop",
    ) -> InferenceRecord:
        """Record an inference request and debit the user's token budget.

        Args:
            user_id: User making the request
            model_name: Model used for inference
            expert_ids: Which experts were activated
            input_tokens: Tokens in the prompt
            output_tokens: Tokens generated
            latency_ms: Request latency in milliseconds
            node_id: Node that served the request
            routing_strategy: Routing strategy used
            finish_reason: Why generation stopped

        Returns:
            InferenceRecord with details

        Raises:
            InsufficientTokens: If user doesn't have enough tokens
            AccountingError: If user account doesn't exist
        """
        account = self._accounts.get(user_id)
        if account is None:
            raise AccountingError(f"No account for user {user_id}")

        total_tokens = input_tokens + output_tokens

        # Check balance before debiting
        if total_tokens > account.balance.total_available:
            logger.warning(
                "Insufficient tokens for user %s: need %d, have %d",
                user_id,
                total_tokens,
                account.balance.total_available,
            )
            raise InsufficientTokens(
                f"User {user_id} needs {total_tokens} tokens but has "
                f"{account.balance.total_available}"
            )

        # Debit the balance
        account.balance.debit(total_tokens)

        # Create the record
        self._record_counter += 1
        record = InferenceRecord(
            record_id=f"inf-{self._record_counter:08d}",
            user_id=user_id,
            model_name=model_name,
            expert_ids=expert_ids,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            timestamp=time.time(),
            node_id=node_id,
            routing_strategy=routing_strategy,
            finish_reason=finish_reason,
        )

        # Update account stats
        account.records.append(record)
        account.total_tokens_used += total_tokens
        account.total_inferences += 1
        account.last_inference_at = time.time()

        # Credit the hosting node
        if node_id:
            self._credit_host(node_id, total_tokens)

        logger.info(
            "Recorded inference %s: user=%s model=%s tokens=%d balance=%d",
            record.record_id,
            user_id,
            model_name,
            total_tokens,
            account.balance.total_available,
        )
        return record

    def check_quota(self, user_id: str, estimated_tokens: int) -> bool:
        """Check if a user has enough tokens for an estimated request.

        Args:
            user_id: User to check
            estimated_tokens: Estimated token count for the request

        Returns:
            True if the user has enough tokens
        """
        account = self._accounts.get(user_id)
        if account is None:
            return False
        return account.balance.total_available >= estimated_tokens

    def process_billing_cycle(self, user_id: str) -> TokenBalance:
        """Process end-of-billing-cycle rollover for a user.

        Rolls over unused tokens (up to the tier max) and resets
        the monthly budget.

        Args:
            user_id: User whose billing cycle to process

        Returns:
            Updated TokenBalance
        """
        account = self._accounts.get(user_id)
        if account is None:
            raise AccountingError(f"No account for user {user_id}")

        old_balance = account.balance.total_available
        account.balance.credit_rollover()
        new_balance = account.balance.total_available

        logger.info(
            "Billing cycle processed for %s: balance %d → %d (rollover=%d)",
            user_id,
            old_balance,
            new_balance,
            account.balance.rollover,
        )
        return account.balance

    def get_usage_summary(self, user_id: str) -> dict[str, Any]:
        """Get a usage summary for a user.

        Args:
            user_id: User to summarize

        Returns:
            Dict with usage stats
        """
        account = self._accounts.get(user_id)
        if account is None:
            return {"error": f"No account for user {user_id}"}

        return {
            "user_id": user_id,
            "tier": account.tier.value,
            "tokens_used": account.total_tokens_used,
            "tokens_remaining": account.balance.total_available,
            "rollover_tokens": account.balance.rollover,
            "total_inferences": account.total_inferences,
            "last_inference_at": account.last_inference_at,
            "is_active": account.is_active,
        }

    def get_all_summaries(self) -> list[dict[str, Any]]:
        """Get usage summaries for all users."""
        return [self.get_usage_summary(uid) for uid in self._accounts]

    def _credit_host(self, node_id: str, tokens: int) -> None:
        """Credit a hosting node for tokens served.

        Earnings rate: $0.002 per 1K tokens served (covers compute cost).
        """
        if node_id not in self._host_earnings:
            self._host_earnings[node_id] = HostEarnings(
                node_id=node_id,
                tokens_served=0,
                credits_earned=0.0,
                usd_earned=0.0,
            )

        earnings = self._host_earnings[node_id]
        earnings.tokens_served += tokens
        earnings.credits_earned += tokens / 1000.0
        earnings.usd_earned += tokens * 0.002 / 1000.0

    def get_host_earnings(self, node_id: str) -> HostEarnings | None:
        """Get earnings for a hosting node."""
        return self._host_earnings.get(node_id)

    def get_all_host_earnings(self) -> list[HostEarnings]:
        """Get earnings for all hosting nodes."""
        return list(self._host_earnings.values())
