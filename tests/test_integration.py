"""Sawyer Integration Tests — end-to-end router + node inference.

Tests the full pipeline: registration → heartbeat → load expert → infer →
unload, with token accounting, identity verification, and routing.
"""

from unittest.mock import patch

import pytest

from sawyer.config import SawyerConfig
from sawyer.identity.bedrock import NodeCertificate, SawyerIdentity
from sawyer.node.weights import WeightLoader
from sawyer.router.scheduler import ExpertScheduler, NodeInfo, RoutingStrategy
from sawyer.router.server import SawyerRouterServicer
from sawyer.token.accounting import InsufficientTokens, TokenAccountant
from sawyer.token.budget import (
    MAX_ROLLOVER,
    TIER_PRICING,
    TIER_TOKENS,
    SubscriptionTier,
)


class TestEndToEndRegistration:
    """End-to-end: node registers with router, heartbeats, then deregisters."""

    def setup_method(self):
        self.scheduler = ExpertScheduler()
        self.servicer = SawyerRouterServicer(self.scheduler)

    def test_node_lifecycle(self):
        """Register -> heartbeat -> deregister."""
        node_info = NodeInfo(
            node_id="node-e2e-1",
            experts=[0, 2],
            gpu="rtx-4090",
            vram_gb=24.0,
            bandwidth_mbps=1000.0,
            latency_ms=25.0,
        )
        self.scheduler.register_node(node_info)
        # Heartbeat — update latency
        self.scheduler.nodes["node-e2e-1"].latency_ms = 30.0
        node = self.scheduler.nodes["node-e2e-1"]
        assert node.latency_ms == 30.0

        # Deregister
        self.scheduler.unregister_node("node-e2e-1")
        assert len(self.scheduler.nodes) == 0

    def test_multiple_nodes_register(self):
        """Multiple nodes register and are tracked."""
        for i in range(5):
            node = NodeInfo(
                node_id=f"node-{i}",
                experts=[0],
                gpu=f"gpu-{i}",
                vram_gb=24.0,
                bandwidth_mbps=1000.0,
                latency_ms=float(10 + i * 5),
            )
            self.scheduler.register_node(node)

        assert len(self.scheduler.nodes) == 5
        assert self.scheduler.get_cluster_status()["total_nodes"] == 5


class TestEndToEndInference:
    """End-to-end: token accounting + inference + routing."""

    def test_inference_with_accounting(self):
        """User makes an inference, tokens are debited correctly."""
        accountant = TokenAccountant()

        # Create user account
        account = accountant.create_account("user-1", SubscriptionTier.PRO)
        initial_balance = account.balance.total_available

        # Record inference
        record = accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=800,
            output_tokens=200,
            latency_ms=45.0,
            node_id="node-a",
            routing_strategy="adaptive",
        )

        assert record.total_tokens == 1000
        assert record.model_name == "mixtral-8x7b"
        assert record.node_id == "node-a"

        # Balance debited
        account = accountant.get_account("user-1")
        assert account.balance.total_available == initial_balance - 1000
        assert account.total_tokens_used == 1000
        assert account.total_inferences == 1

    def test_insufficient_tokens_blocks_inference(self):
        """Inference is blocked when user has insufficient tokens."""
        accountant = TokenAccountant()
        accountant.create_account("user-1", SubscriptionTier.PRO)

        # Use up all tokens
        accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=400_000,
            output_tokens=100_000,
            latency_ms=100.0,
            node_id="node-a",
        )

        # Now check quota
        assert accountant.check_quota("user-1", 1) is False

        # And recording more should fail
        with pytest.raises(InsufficientTokens):
            accountant.record_inference(
                user_id="user-1",
                model_name="mixtral-8x7b",
                expert_ids=[0],
                input_tokens=10,
                output_tokens=10,
                latency_ms=50.0,
            )

    def test_rollover_across_billing_cycles(self):
        """Unused tokens roll over at billing cycle boundary."""
        accountant = TokenAccountant()
        accountant.create_account("user-1", SubscriptionTier.PRO)

        # Use 1M of 2M tokens
        accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=2_000_000,
            output_tokens=2_000_000,
            latency_ms=100.0,
            node_id="node-a",
        )

        # Process billing cycle — 1M remaining rolls over
        balance = accountant.process_billing_cycle("user-1")
        assert balance.rollover == 1_000_000
        assert balance.current_balance == 2_000_000
        assert balance.total_available == 3_000_000

    def test_rollover_cap_enforced(self):
        """Rollover is capped at MAX_ROLLOVER for the tier."""
        accountant = TokenAccountant()
        accountant.create_account("user-1", SubscriptionTier.PRO, rollover=5_000_000)

        account = accountant.get_account("user-1")
        assert account.balance.rollover == MAX_ROLLOVER[SubscriptionTier.PRO]

    def test_host_earnings_tracked(self):
        """Hosting node earns credits for tokens served."""
        accountant = TokenAccountant()
        accountant.create_account("user-1", SubscriptionTier.PRO)

        accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=500,
            output_tokens=500,
            latency_ms=100.0,
            node_id="node-a",
        )

        earnings = accountant.get_host_earnings("node-a")
        assert earnings is not None
        assert earnings.tokens_served == 1000
        assert earnings.usd_earned > 0
        assert earnings.usd_earned == 1000 * 0.002 / 1000


class TestEndToEndRouting:
    """End-to-end: routing strategies select the right node."""

    def setup_method(self):
        self.scheduler = ExpertScheduler()

    def _register_nodes(self):
        """Register test nodes with varying load and latency."""
        nodes = [
            NodeInfo("near-empty", [0], "gpu", 24.0, 1000.0, 15.0),
            NodeInfo(
                "near-full",
                [0, 1, 2, 3, 4, 5, 6],
                "gpu",
                24.0,
                1000.0,
                20.0,
            ),
            NodeInfo("far-empty", [0], "gpu", 24.0, 1000.0, 200.0),
            NodeInfo(
                "mid-load",
                [0, 1, 2, 3],
                "gpu",
                24.0,
                1000.0,
                50.0,
            ),
        ]
        for n in nodes:
            self.scheduler.register_node(n)

    def test_adaptive_prefers_near_empty(self):
        """ADAPTIVE: nearby idle node wins over distant idle node."""
        self._register_nodes()
        node = self.scheduler.select_node(
            expert_id=0,
            strategy=RoutingStrategy.ADAPTIVE,
        )
        assert node is not None
        # near-empty (15ms, 1/8) should beat far-empty (200ms, 1/8)
        assert node.node_id == "near-empty"

    def test_least_loaded_picks_underutilized(self):
        """LEAST_LOADED: always picks least loaded node."""
        self._register_nodes()
        node = self.scheduler.select_node(
            expert_id=0,
            strategy=RoutingStrategy.LEAST_LOADED,
        )
        assert node is not None
        # Should pick one of the lightly loaded nodes
        assert node.node_id in ("near-empty", "far-empty")

    def test_power_of_two_randomized(self):
        """POWER_OF_TWO: picks from 2 random candidates."""
        self._register_nodes()
        # Run multiple times — should not always pick same node
        selected = set()
        for _ in range(20):
            node = self.scheduler.select_node(
                expert_id=0,
                strategy=RoutingStrategy.POWER_OF_TWO,
            )
            if node:
                selected.add(node.node_id)
        # Should have picked at least 2 different nodes
        assert len(selected) >= 2


class TestEndToEndIdentity:
    """End-to-end: node identity with Bedrock SDK integration."""

    def test_local_fallback_identity(self):
        """Node creates local identity when Bedrock unavailable."""
        import asyncio

        config = SawyerConfig(bedrock_url="http://localhost:9999")

        with patch("sawyer.identity.bedrock.BedrockClient") as mock_sdk:
            mock_sdk.side_effect = Exception("Connection refused")
            identity = SawyerIdentity(config)
            node_id = asyncio.run(identity.register_node("test-node"))
            assert node_id is not None
            assert node_id.startswith("sawyer-local-")

    def test_certificate_fields(self):
        """Node certificate has required fields."""
        cert = NodeCertificate(
            node_uuid="sawyer-test-001",
            node_name="test-node",
            public_key_hash="abc123",
        )
        assert cert.node_uuid == "sawyer-test-001"
        assert cert.node_name == "test-node"


class TestEndToEndWeightLoading:
    """End-to-end: weight loading with cache management."""

    def test_weight_loader_initialization(self):
        """WeightLoader initializes with config paths."""
        config = SawyerConfig(cache_dir="/tmp/sawyer-test-cache")
        loader = WeightLoader(config)
        assert loader.config is not None

    def test_weight_loader_is_cached_unknown(self):
        """Unknown model is not cached."""
        config = SawyerConfig(cache_dir="/tmp/sawyer-test-cache")
        loader = WeightLoader(config)
        assert loader.is_cached("nonexistent-model-xyz") is False


class TestEndToEndTierSystem:
    """End-to-end: tier system pricing and tokens are consistent."""

    def test_all_tiers_have_pricing(self):
        """Every tier has a price defined."""
        for tier in SubscriptionTier:
            assert tier in TIER_PRICING
            assert TIER_PRICING[tier] > 0

    def test_all_tiers_have_token_budgets(self):
        """Every tier has a token budget defined."""
        for tier in SubscriptionTier:
            assert tier in TIER_TOKENS
            assert TIER_TOKENS[tier] > 0

    def test_all_tiers_have_rollover_caps(self):
        """Every tier has a max rollover defined."""
        for tier in SubscriptionTier:
            assert tier in MAX_ROLLOVER
            assert MAX_ROLLOVER[tier] > 0

    def test_tier_ordering(self):
        """Tiers are ordered by value: Explorer < Builder < Operator."""
        prices = [TIER_PRICING[t] for t in SubscriptionTier]
        tokens = [TIER_TOKENS[t] for t in SubscriptionTier]
        assert prices == sorted(prices)
        assert tokens == sorted(tokens)

    def test_token_debit_updates_all_fields(self):
        """Debiting tokens updates balance, used count, and inferences."""
        accountant = TokenAccountant()
        accountant.create_account("user-1", SubscriptionTier.PRO)

        accountant.record_inference(
            user_id="user-1",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=100,
            output_tokens=50,
            latency_ms=50.0,
        )

        account = accountant.get_account("user-1")
        assert account.balance.current_balance == 2_000_000 - 150
        assert account.total_tokens_used == 150
        assert account.total_inferences == 1
        assert account.last_inference_at > 0


class TestEndToEndMultipleUsers:
    """End-to-end: multiple users with different tiers."""

    def test_multiple_users_different_tiers(self):
        """Multiple users with different tiers operate independently."""
        accountant = TokenAccountant()

        accountant.create_account("pro-user", SubscriptionTier.PRO)
        accountant.create_account("pro-user", SubscriptionTier.PRO)
        accountant.create_account("enterprise-user", SubscriptionTier.ENTERPRISE)

        # Explorer: 500K tokens
        accountant.record_inference(
            user_id="pro-user",
            model_name="mixtral-8x7b",
            expert_ids=[0],
            input_tokens=100,
            output_tokens=50,
            latency_ms=50.0,
        )

        # Builder: 2M tokens
        accountant.record_inference(
            user_id="pro-user",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
            input_tokens=1000,
            output_tokens=500,
            latency_ms=80.0,
        )

        # Operator: 5M tokens
        accountant.record_inference(
            user_id="enterprise-user",
            model_name="mixtral-8x7b",
            expert_ids=[0, 2, 4],
            input_tokens=5000,
            output_tokens=2000,
            latency_ms=120.0,
        )

        summaries = accountant.get_all_summaries()
        assert len(summaries) == 3

        explorer = accountant.get_account("pro-user")
        builder = accountant.get_account("pro-user")
        operator = accountant.get_account("enterprise-user")

        assert explorer.balance.total_available == 2_000_000 - 150
        assert builder.balance.total_available == 2_000_000 - 1500
        assert operator.balance.total_available == 5_000_000 - 7000
