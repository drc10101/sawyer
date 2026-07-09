"""Tests for ProviderEarningsSync — wiring inference to provider payouts."""

from sawyer.provider.earnings_sync import ProviderEarningsSync
from sawyer.provider.manager import ProviderManager
from sawyer.token.accounting import TokenAccountant
from sawyer.token.budget import SubscriptionTier


class TestEarningsSync:
    """Test syncing token earnings to provider payouts."""

    def setup_method(self):
        self.accountant = TokenAccountant()
        self.provider_mgr = ProviderManager()
        self.sync = ProviderEarningsSync(self.accountant, self.provider_mgr)

        # Set up a provider with a node
        self.provider = self.provider_mgr.register(email="host@example.com", display_name="HostCo")
        self.provider_mgr.verify_provider(self.provider.provider_id, "acct_test")
        self.provider_mgr.activate_provider(self.provider.provider_id)
        self.provider_mgr.add_node(self.provider.provider_id, "node-alpha")

        # Create a user account for inference
        self.accountant.create_account("user-1", SubscriptionTier.PRO)

    def test_flush_host_earnings_to_provider(self):
        """Tokens served by a node credit the owning provider."""
        # Run inference through the node
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=800,
            output_tokens=200,
            latency_ms=45.0,
            node_id="node-alpha",
        )

        # Flush should credit the provider
        credited = self.sync.flush_host_earnings()
        assert self.provider.provider_id in credited
        assert credited[self.provider.provider_id] > 0

        # Provider should have earned money
        assert self.provider.total_usd_earned > 0
        assert self.provider.total_tokens_served == 1000

    def test_flush_only_credits_new_tokens(self):
        """Second flush only credits tokens since last flush."""
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=500,
            output_tokens=500,
            latency_ms=30.0,
            node_id="node-alpha",
        )

        # First flush
        credited1 = self.sync.flush_host_earnings()
        assert credited1[self.provider.provider_id] > 0

        # Second flush with no new inferences — should credit 0
        credited2 = self.sync.flush_host_earnings()
        assert (
            self.provider.provider_id not in credited2
            or credited2[self.provider.provider_id] == 0.0
        )

    def test_flush_accumulates_across_multiple_inferences(self):
        """Multiple inferences before flush all get credited."""
        for _ in range(5):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=100,
                output_tokens=100,
                latency_ms=10.0,
                node_id="node-alpha",
            )

        credited = self.sync.flush_host_earnings()
        # 5 * 200 = 1000 tokens
        assert self.provider.total_tokens_served == 1000
        assert credited[self.provider.provider_id] > 0

    def test_node_without_provider_is_skipped(self):
        """Nodes without a registered provider are skipped during flush."""
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=500,
            output_tokens=500,
            latency_ms=30.0,
            node_id="orphan-node",
        )

        # Should not crash, just skip
        credited = self.sync.flush_host_earnings()
        assert "orphan-node" not in credited

    def test_flush_single_node(self):
        """Flush earnings for a single node."""
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=300,
            output_tokens=700,
            latency_ms=20.0,
            node_id="node-alpha",
        )

        usd = self.sync.flush_single_node("node-alpha")
        assert usd is not None
        assert usd > 0
        assert self.provider.total_usd_earned > 0

    def test_get_eligible_providers(self):
        """Get providers eligible for payout."""
        # Provider starts with $0 — not eligible
        eligible = self.sync.get_eligible_providers()
        assert len(eligible) == 0

        # Serve a lot of tokens to get above $10 minimum
        for _ in range(20):
            self.accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=500,
                output_tokens=500,
                latency_ms=10.0,
                node_id="node-alpha",
            )

        self.sync.flush_host_earnings()

        # 20 * 1000 = 20K tokens * $0.002/1K = $0.04 * 70% = $0.028
        # Still under $10 — need way more
        # Let's just directly credit to get above threshold
        self.provider_mgr.credit_earnings(self.provider.provider_id, tokens_served=15_000_000)

        eligible = self.sync.get_eligible_providers()
        assert len(eligible) >= 1
        assert eligible[0]["provider_id"] == self.provider.provider_id

    def test_get_provider_for_node(self):
        """Get provider info for a specific node."""
        self.accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=200,
            output_tokens=300,
            latency_ms=15.0,
            node_id="node-alpha",
        )

        info = self.sync.get_provider_for_node("node-alpha")
        assert info is not None
        assert info["provider_id"] == self.provider.provider_id
        assert info["node_tokens_served"] == 500

    def test_get_provider_for_unknown_node(self):
        """Unknown node returns None."""
        info = self.sync.get_provider_for_node("unknown-node")
        assert info is None
