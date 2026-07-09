"""Tests for Sawyer Token Accounting."""

import pytest

from sawyer.token.accounting import (
    AccountingError,
    InsufficientTokens,
    TokenAccountant,
    UserAccount,
)
from sawyer.token.budget import SubscriptionTier


class TestTokenAccountant:
    """Test TokenAccountant — the core token accounting engine."""

    def setup_method(self):
        self.accountant = TokenAccountant()

    def test_create_account(self):
        """Create an account with Explorer tier."""
        account = self.accountant.create_account("user-1", SubscriptionTier.PRO)
        assert account.user_id == "user-1"
        assert account.tier == SubscriptionTier.PRO
        assert account.balance.total_available == 2_000_000
        assert account.total_tokens_used == 0

    def test_create_account_with_rollover(self):
        """Create an account with rollover tokens."""
        account = self.accountant.create_account(
            "user-2", SubscriptionTier.PRO, rollover=100_000
        )
        assert account.balance.total_available == 2_100_000  # 2M + 100K

    def test_create_account_duplicate_raises(self):
        """Creating a duplicate account raises AccountingError."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        with pytest.raises(AccountingError, match="already exists"):
            self.accountant.create_account("user-1", SubscriptionTier.PRO)

    def test_get_account(self):
        """Get an existing account."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        account = self.accountant.get_account("user-1")
        assert account is not None
        assert account.user_id == "user-1"

    def test_get_account_not_found(self):
        """Get returns None for non-existent account."""
        assert self.accountant.get_account("nonexistent") is None

    def test_record_inference(self):
        """Record an inference and debit tokens."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        record = self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=100,
            output_tokens=50,
            latency_ms=120.0,
            node_id="node-a",
        )
        assert record.total_tokens == 150
        assert record.model_name == "mixtral-8x7b"
        assert record.record_id.startswith("inf-")

        # Check balance was debited
        account = self.accountant.get_account("user-1")
        assert account.balance.total_available == 2_000_000 - 150
        assert account.total_tokens_used == 150
        assert account.total_inferences == 1

    def test_record_inference_insufficient_tokens(self):
        """Raise InsufficientTokens when balance is too low."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        with pytest.raises(InsufficientTokens):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=2_000_000,
                output_tokens=1,
                latency_ms=100.0,
            )

    def test_record_inference_unknown_user(self):
        """Raise AccountingError for non-existent user."""
        with pytest.raises(AccountingError, match="No account"):
            self.accountant.record_inference(
                user_id="nonexistent",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=10,
                output_tokens=10,
                latency_ms=100.0,
            )

    def test_uses_rollover_first(self):
        """Token debit uses rollover tokens before monthly budget."""
        account = self.accountant.create_account(
            "user-1", SubscriptionTier.PRO, rollover=2_000_000
        )
        assert account.balance.total_available == 4_000_000  # 2M + 2M rollover

        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=300_000,
            output_tokens=100_000,
            latency_ms=100.0,
        )

        # 400K debited from rollover first: 2M - 400K = 1.6M remaining
        account = self.accountant.get_account("user-1")
        assert account.balance.rollover == 1_600_000
        assert account.balance.current_balance == 2_000_000

    def test_check_quota(self):
        """Check if a user has enough tokens."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        assert self.accountant.check_quota("user-1", 100) is True
        assert self.accountant.check_quota("user-1", 5_000_000) is False

    def test_check_quota_unknown_user(self):
        """Check quota returns False for unknown user."""
        assert self.accountant.check_quota("nonexistent", 100) is False

    def test_process_billing_cycle(self):
        """Process billing cycle rolls over unused tokens."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)

        # Use some tokens
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=100_000,
            output_tokens=50_000,
            latency_ms=100.0,
        )

        # Process billing cycle — 1.85M remaining rolls over
        balance = self.accountant.process_billing_cycle("user-1")
        assert balance.rollover == 1_850_000
        assert balance.current_balance == 2_000_000  # Reset to monthly budget
        assert balance.total_available == 3_850_000

    def test_usage_summary(self):
        """Get a usage summary for a user."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)

        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=500,
            output_tokens=200,
            latency_ms=150.0,
        )

        summary = self.accountant.get_usage_summary("user-1")
        assert summary["tier"] == "pro"
        assert summary["tokens_used"] == 700
        assert summary["total_inferences"] == 1
        assert summary["is_active"] is True

    def test_usage_summary_nonexistent(self):
        """Usage summary for non-existent user returns error."""
        summary = self.accountant.get_usage_summary("nonexistent")
        assert "error" in summary

    def test_get_all_summaries(self):
        """Get summaries for all users."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        self.accountant.create_account("user-2", SubscriptionTier.PRO)

        summaries = self.accountant.get_all_summaries()
        assert len(summaries) == 2

    def test_host_earnings(self):
        """Recording inference credits the hosting node."""
        self.accountant.create_account("user-1", SubscriptionTier.PRO)
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=1000,
            output_tokens=500,
            latency_ms=100.0,
            node_id="node-a",
        )

        earnings = self.accountant.get_host_earnings("node-a")
        assert earnings is not None
        assert earnings.tokens_served == 1500
        assert earnings.usd_earned > 0

    def test_host_payout_threshold(self):
        """Host earnings track eligibility for payout."""
        self.accountant.create_account("user-1", SubscriptionTier.ENTERPRISE)
        # Use enough tokens to accumulate earnings
        for _ in range(10):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=100_000,
                output_tokens=50_000,
                latency_ms=100.0,
                node_id="node-a",
            )

        earnings = self.accountant.get_host_earnings("node-a")
        assert earnings is not None
        assert earnings.tokens_served == 1_500_000


class TestUserAccount:
    """Test UserAccount dataclass."""

    def test_is_active_with_tokens(self):
        """Account is active when it has tokens."""
        from sawyer.token.budget import TokenBalance

        balance = TokenBalance(
            tier=SubscriptionTier.PRO,
            monthly_budget=2_000_000,
            current_balance=2_000_000,
        )
        account = UserAccount(
            user_id="test",
            tier=SubscriptionTier.PRO,
            balance=balance,
        )
        assert account.is_active is True

    def test_is_active_zero_tokens(self):
        """Account is not active when tokens are depleted."""
        from sawyer.token.budget import TokenBalance

        balance = TokenBalance(
            tier=SubscriptionTier.PRO,
            monthly_budget=2_000_000,
            current_balance=0,
        )
        account = UserAccount(
            user_id="test",
            tier=SubscriptionTier.PRO,
            balance=balance,
        )
        assert account.is_active is False
