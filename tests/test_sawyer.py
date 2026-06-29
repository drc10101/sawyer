"""Tests for Sawyer — Distributed MoE Inference Network.

SPDX-License-Identifier: BSL-1.1 — See LICENSE for details.
"""

import pytest

from sawyer.config import SawyerConfig
from sawyer.model.registry import (
    can_host_expert,
    get_model,
    list_models,
)
from sawyer.node.agent import SawyerNode
from sawyer.router.scheduler import ExpertScheduler, NodeInfo, RoutingStrategy
from sawyer.token.budget import (
    TIER_PRICING,
    TIER_TOKENS,
    HostEarnings,
    SubscriptionTier,
    TokenBalance,
)

# ── Config ──


class TestSawyerConfig:
    def test_default_config(self):
        config = SawyerConfig()
        assert config.inference_port == 8444
        assert config.max_experts == 2
        assert config.inference_backend == "auto"

    def test_validate_valid_config(self):
        config = SawyerConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_port(self):
        config = SawyerConfig(inference_port=0)
        with pytest.raises(ValueError, match="Invalid port"):
            config.validate()

    def test_validate_invalid_max_experts(self):
        config = SawyerConfig(max_experts=0)
        with pytest.raises(ValueError, match="max_experts"):
            config.validate()


# ── Model Registry ──


class TestModelRegistry:
    def test_list_models(self):
        models = list_models()
        assert len(models) >= 4
        assert any(m.name == "mixtral-8x7b" for m in models)

    def test_get_model(self):
        model = get_model("mixtral-8x7b")
        assert model.display_name == "Mixtral 8x7B"
        assert model.num_experts == 8
        assert model.active_experts == 2

    def test_get_model_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("nonexistent-model")

    def test_can_host_expert(self):
        model = get_model("mixtral-8x7b")
        assert can_host_expert(model, available_vram_gb=8.0) is True
        assert can_host_expert(model, available_vram_gb=1.0) is False

    def test_deepseek_v2_lite(self):
        model = get_model("deepseek-v2-lite")
        assert model.num_experts == 64
        assert model.active_experts == 6

    def test_dbrx(self):
        model = get_model("dbrx")
        assert model.num_experts == 16
        assert model.active_experts == 4


# ── Expert Scheduler ──


class TestExpertScheduler:
    def setup_method(self):
        self.scheduler = ExpertScheduler()
        self.node_a = NodeInfo(
            node_id="node-a",
            experts=[0, 2],
            gpu="RTX 3090",
            vram_gb=24.0,
            bandwidth_mbps=100.0,
            latency_ms=15.0,
        )
        self.node_b = NodeInfo(
            node_id="node-b",
            experts=[0, 1, 3],
            gpu="A100",
            vram_gb=80.0,
            bandwidth_mbps=200.0,
            latency_ms=30.0,
        )

    def test_register_node(self):
        self.scheduler.register_node(self.node_a)
        assert "node-a" in self.scheduler.nodes

    def test_unregister_node(self):
        self.scheduler.register_node(self.node_a)
        self.scheduler.unregister_node("node-a")
        assert "node-a" not in self.scheduler.nodes

    def test_find_expert_nodes(self):
        self.scheduler.register_node(self.node_a)
        self.scheduler.register_node(self.node_b)
        nodes = self.scheduler.find_expert_nodes(0)
        assert len(nodes) == 2  # Both nodes host expert 0

    def test_find_expert_nodes_no_match(self):
        self.scheduler.register_node(self.node_a)
        nodes = self.scheduler.find_expert_nodes(5)  # No node has expert 5
        assert len(nodes) == 0

    def test_select_node_prefers_low_latency(self):
        self.scheduler.register_node(self.node_a)
        self.scheduler.register_node(self.node_b)
        node = self.scheduler.select_node(
            0, strategy=RoutingStrategy.LOWEST_LATENCY
        )
        assert node.node_id == "node-a"  # Lower latency

    def test_select_node_no_candidates(self):
        node = self.scheduler.select_node(99)
        assert node is None


# ── Routing Strategies ──


class TestRoutingStrategies:
    """Test all four routing strategies."""

    def setup_method(self):
        self.scheduler = ExpertScheduler(strategy=RoutingStrategy.ADAPTIVE)
        self.node_near_idle = NodeInfo(
            node_id="near-idle",
            experts=[0],
            gpu="RTX 3090",
            vram_gb=24.0,
            bandwidth_mbps=100.0,
            latency_ms=15.0,
            active_requests=0,
            max_concurrent_requests=10,
        )
        self.node_near_busy = NodeInfo(
            node_id="near-busy",
            experts=[0],
            gpu="RTX 4090",
            vram_gb=24.0,
            bandwidth_mbps=100.0,
            latency_ms=20.0,
            active_requests=9,
            max_concurrent_requests=10,
        )
        self.node_far_idle = NodeInfo(
            node_id="far-idle",
            experts=[0],
            gpu="A100",
            vram_gb=80.0,
            bandwidth_mbps=200.0,
            latency_ms=200.0,
            active_requests=0,
            max_concurrent_requests=20,
        )

    def test_lowest_latency_picks_nearest(self):
        """LOWEST_LATENCY always picks the closest node regardless of load."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_near_busy)
        self.scheduler.register_node(self.node_far_idle)

        node = self.scheduler.select_node(0, strategy=RoutingStrategy.LOWEST_LATENCY)
        assert node.node_id == "near-idle"  # 15ms wins

    def test_least_loaded_picks_underutilized(self):
        """LEAST_LOADED picks the least busy node regardless of latency."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_near_busy)
        self.scheduler.register_node(self.node_far_idle)

        node = self.scheduler.select_node(0, strategy=RoutingStrategy.LEAST_LOADED)
        # far-idle and near-idle both 0% load, but least_loaded sorts by load
        # then by position — one of the idle ones wins
        assert node.node_id in ("near-idle", "far-idle")

    def test_adaptive_prefers_close_idle_over_far_idle(self):
        """ADAPTIVE: nearby idle beats far idle because latency matters."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_far_idle)

        node = self.scheduler.select_node(0, strategy=RoutingStrategy.ADAPTIVE)
        assert node.node_id == "near-idle"

    def test_adaptive_sends_to_underutilized_over_nearby_busy(self):
        """ADAPTIVE: an underutilized server can beat a nearby overloaded one."""
        self.scheduler.register_node(self.node_near_busy)
        self.scheduler.register_node(self.node_far_idle)

        node = self.scheduler.select_node(0, strategy=RoutingStrategy.ADAPTIVE)
        # far-idle: latency=200ms (0.4), load=0% (0.0) → score = 0.4*0.4 + 0.6*0.0 = 0.16
        # near-busy: latency=20ms (0.04), load=90% (0.9) → score = 0.4*0.04 + 0.6*0.9 = 0.556
        # far-idle wins because load_weight=0.6 dominates
        assert node.node_id == "far-idle"

    def test_adaptive_near_idle_beats_near_busy(self):
        """ADAPTIVE: nearby idle is the best of all worlds."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_near_busy)
        self.scheduler.register_node(self.node_far_idle)

        node = self.scheduler.select_node(0, strategy=RoutingStrategy.ADAPTIVE)
        assert node.node_id == "near-idle"

    def test_power_of_two_returns_valid_node(self):
        """POWER_OF_TWO always picks from the candidate pool."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_near_busy)
        self.scheduler.register_node(self.node_far_idle)

        # Run multiple times — should always return a valid node
        for _ in range(20):
            node = self.scheduler.select_node(
                0, strategy=RoutingStrategy.POWER_OF_TWO
            )
            assert node.node_id in ("near-idle", "near-busy", "far-idle")

    def test_complete_request_updates_metrics(self):
        """complete_request decrements active_requests and updates avg_response."""
        self.scheduler.register_node(self.node_near_idle)
        self.node_near_idle.active_requests = 3

        self.scheduler.complete_request("near-idle", response_ms=50.0)
        assert self.node_near_idle.active_requests == 2
        assert self.node_near_idle.requests_served == 1
        assert self.node_near_idle.avg_response_ms > 0

    def test_cluster_status(self):
        """cluster_status returns summary of all nodes."""
        self.scheduler.register_node(self.node_near_idle)
        self.scheduler.register_node(self.node_near_busy)
        status = self.scheduler.get_cluster_status()
        assert status["total_nodes"] == 2
        assert status["healthy_nodes"] == 2
        assert "near-idle" in status["nodes"]
        assert "near-busy" in status["nodes"]

    def test_load_fraction(self):
        """_load_fraction computes active/capacity ratio."""
        assert self.scheduler._load_fraction(self.node_near_idle) == 0.0
        self.node_near_idle.active_requests = 5
        assert self.scheduler._load_fraction(self.node_near_idle) == 0.5
        self.node_near_idle.active_requests = 0


# ── Node Agent ──


class TestSawyerNode:
    def setup_method(self):
        self.config = SawyerConfig(node_name="test-node")
        self.node = SawyerNode(self.config)

    def test_init(self):
        assert self.node.node_id is None
        assert len(self.node.experts) == 0

    @pytest.mark.asyncio
    async def test_register(self):
        """Test register by mocking the router client — no real gRPC connection."""
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.register.return_value = "sawyer-node-42"
        mock_client.connect = MagicMock()

        with (
            patch.object(self.node, "_router_client", None),
            patch("sawyer.node.agent.RouterClient", return_value=mock_client),
        ):
            node_id = await self.node.register(name="test-node")
            assert node_id == "sawyer-node-42"
            assert self.node.node_id == "sawyer-node-42"

    @pytest.mark.asyncio
    async def test_load_expert(self):
        await self.node.load_expert("mixtral-8x7b", 0)
        assert "mixtral-8x7b:0" in self.node.experts

    @pytest.mark.asyncio
    async def test_load_expert_idempotent(self):
        await self.node.load_expert("mixtral-8x7b", 0)
        await self.node.load_expert("mixtral-8x7b", 0)  # Should not duplicate
        assert len(self.node.experts) == 1

    @pytest.mark.asyncio
    async def test_unload_expert(self):
        await self.node.load_expert("mixtral-8x7b", 0)
        await self.node.unload_expert("mixtral-8x7b", 0)
        assert "mixtral-8x7b:0" not in self.node.experts


# ── Token Economics ──


class TestTokenBalance:
    def test_explorer_tier(self):
        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=500_000,
            current_balance=500_000,
        )
        assert balance.total_available == 500_000

    def test_debit_from_monthly(self):
        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=500_000,
            current_balance=500_000,
        )
        remaining = balance.debit(100_000)
        assert remaining == 400_000
        assert balance.current_balance == 400_000

    def test_debit_uses_rollover_first(self):
        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=500_000,
            current_balance=500_000,
            rollover=50_000,
        )
        total_before = balance.total_available
        remaining = balance.debit(30_000)
        assert remaining == total_before - 30_000
        assert balance.rollover == 20_000
        assert balance.current_balance == 500_000

    def test_debit_insufficient(self):
        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=500_000,
            current_balance=100_000,
        )
        with pytest.raises(ValueError, match="Insufficient"):
            balance.debit(200_000)

    def test_credit_rollover(self):
        balance = TokenBalance(
            tier=SubscriptionTier.EXPLORER,
            monthly_budget=500_000,
            current_balance=200_000,
        )
        balance.credit_rollover()
        assert balance.rollover == 200_000
        assert balance.current_balance == 500_000


class TestHostEarnings:
    def test_payout_threshold(self):
        earnings = HostEarnings(
            node_id="node-a",
            tokens_served=100_000,
            credits_earned=10.0,
            usd_earned=5.0,
        )
        assert not earnings.eligible_for_payout

        earnings.usd_earned = 10.0
        assert earnings.eligible_for_payout


class TestTierPricing:
    def test_explorer_price(self):
        assert TIER_PRICING[SubscriptionTier.EXPLORER] == 5

    def test_builder_price(self):
        assert TIER_PRICING[SubscriptionTier.BUILDER] == 20

    def test_operator_price(self):
        assert TIER_PRICING[SubscriptionTier.OPERATOR] == 50

    def test_explorer_tokens(self):
        assert TIER_TOKENS[SubscriptionTier.EXPLORER] == 500_000

    def test_builder_tokens(self):
        assert TIER_TOKENS[SubscriptionTier.BUILDER] == 2_000_000

    def test_operator_tokens(self):
        assert TIER_TOKENS[SubscriptionTier.OPERATOR] == 5_000_000
