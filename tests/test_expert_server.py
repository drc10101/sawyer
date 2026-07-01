"""Tests for Sawyer Expert Server — load, forward pass, health reporting."""

import asyncio
import time

import pytest

from sawyer.config import SawyerConfig
from sawyer.node.expert_server import ExpertServer, ExpertSlot, HealthReport
from sawyer.node.inference import BackendMode, InferenceResult, LlamaCppBackend


# --- Fixtures ---

@pytest.fixture
def config():
    return SawyerConfig(max_vram_gb=24.0, max_experts=4)


@pytest.fixture
def server(config):
    return ExpertServer(config)


@pytest.fixture
def slot():
    return ExpertSlot(
        model_name="mixtral-8x7b",
        expert_id=0,
        vram_bytes=int(4.9 * 1024**3),  # 4.9 GB Q4_K_M
    )


# --- ExpertSlot tests ---

class TestExpertSlot:
    def test_key(self, slot):
        assert slot.key == "mixtral-8x7b:0"

    def test_avg_latency_zero(self, slot):
        assert slot.avg_latency_ms == 0.0

    def test_avg_latency_with_data(self, slot):
        slot.inference_count = 3
        slot.total_latency_ms = 150.0
        assert slot.avg_latency_ms == 50.0

    def test_default_status(self, slot):
        assert slot.status == "loading"


# --- ExpertServer tests ---

class TestExpertServer:
    def test_init(self, server):
        assert server.config.max_vram_gb == 24.0
        assert len(server.experts) == 0

    def test_set_node_id(self, server):
        server.set_node_id("node-001")
        assert server._node_id == "node-001"

    def test_set_gpu_info(self, server):
        server.set_gpu_info("RTX 4090", 24 * 1024**3)
        assert server._gpu_name == "RTX 4090"
        assert server._vram_total == 24 * 1024**3

    def test_health_report_empty(self, server):
        server.set_node_id("node-001")
        report = server.health_report()
        assert isinstance(report, HealthReport)
        assert report.node_id == "node-001"
        assert report.experts_loaded == 0
        assert report.experts_ready == 0
        assert report.total_inferences == 0
        assert report.is_healthy is False  # No experts loaded

    def test_health_report_with_expert(self, server):
        server.set_node_id("node-001")
        server.set_gpu_info("RTX 4090", 24 * 1024**3)

        # Manually add a ready expert
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
        )
        server.experts["mixtral-8x7b:0"] = slot

        report = server.health_report()
        assert report.experts_loaded == 1
        assert report.experts_ready == 1
        assert report.is_healthy is True
        assert report.vram_used_bytes > 0

    def test_health_report_to_dict(self, server):
        server.set_node_id("node-001")
        report = server.health_report()
        d = report.to_dict()
        assert "node_id" in d
        assert "is_healthy" in d
        assert d["node_id"] == "node-001"

    def test_load_expert_already_loaded(self, server):
        """Loading an already-ready expert should return the existing slot."""
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
        )
        server.experts["mixtral-8x7b:0"] = slot

        result = asyncio.run(server.load_expert("mixtral-8x7b", 0))
        assert result is slot

    def test_load_expert_vram_exceeded(self):
        """Loading an expert when VRAM is full should raise ValueError."""
        small_config = SawyerConfig(max_vram_gb=2.0, max_experts=1)
        server = ExpertServer(small_config)

        # Pre-fill VRAM
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(2.0 * 1024**3),
            status="ready",
        )
        server.experts["mixtral-8x7b:0"] = slot

        # mixtral expert is ~4.9 GB, only ~0 GB remaining
        with pytest.raises(ValueError, match="Insufficient VRAM"):
            asyncio.run(server.load_expert("mixtral-8x7b", 1))

    def test_unload_expert(self, server):
        """Unloading an expert should remove it from the server."""
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
        )
        server.experts["mixtral-8x7b:0"] = slot

        asyncio.run(server.unload_expert("mixtral-8x7b", 0))
        assert "mixtral-8x7b:0" not in server.experts

    def test_unload_nonexistent_expert(self, server):
        """Unloading a non-existent expert should not raise."""
        asyncio.run(server.unload_expert("mixtral-8x7b", 99))

    def test_get_expert_status(self, server):
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
        )
        server.experts["mixtral-8x7b:0"] = slot

        assert server.get_expert_status("mixtral-8x7b", 0) == "ready"
        assert server.get_expert_status("mixtral-8x7b", 1) is None

    def test_list_experts(self, server):
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
            inference_count=5,
            total_latency_ms=250.0,
        )
        server.experts["mixtral-8x7b:0"] = slot

        experts = server.list_experts()
        assert len(experts) == 1
        assert experts[0]["model"] == "mixtral-8x7b"
        assert experts[0]["expert_id"] == 0
        assert experts[0]["status"] == "ready"
        assert experts[0]["inferences"] == 5

    def test_forward_pass_not_loaded(self, server):
        """Forward pass on a non-loaded expert should raise ValueError."""
        with pytest.raises(ValueError, match="not loaded"):
            asyncio.run(server.forward_pass("mixtral-8x7b", 0, "Hello"))

    def test_forward_pass_not_ready(self, server):
        """Forward pass on a loading expert should raise ValueError."""
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="loading",
        )
        server.experts["mixtral-8x7b:0"] = slot

        with pytest.raises(ValueError, match="not ready"):
            asyncio.run(server.forward_pass("mixtral-8x7b", 0, "Hello"))


# --- SawyerNode integration tests ---

class TestSawyerNodeIntegration:
    def test_node_offline_mode(self):
        from sawyer.node.agent import SawyerNode
        node = SawyerNode(SawyerConfig())
        asyncio.run(node.start(offline=True))
        assert node.node_id == "offline"
        asyncio.run(node.stop())

    def test_node_has_expert_server(self):
        from sawyer.node.agent import SawyerNode
        node = SawyerNode(SawyerConfig())
        assert isinstance(node.expert_server, ExpertServer)

    def test_node_load_and_list_experts(self):
        from sawyer.node.agent import SawyerNode
        config = SawyerConfig(max_vram_gb=24.0, max_experts=4)
        node = SawyerNode(config)
        asyncio.run(node.start(offline=True))

        # Manually add an expert slot (no real backend download)
        slot = ExpertSlot(
            model_name="mixtral-8x7b",
            expert_id=0,
            vram_bytes=int(4.9 * 1024**3),
            status="ready",
        )
        node.expert_server.experts["mixtral-8x7b:0"] = slot

        experts = node.expert_server.list_experts()
        assert len(experts) == 1

        # Health report
        report = node.expert_server.health_report()
        assert report.experts_loaded == 1
        assert report.is_healthy is True

        asyncio.run(node.stop())

    def test_node_health_report(self):
        from sawyer.node.agent import SawyerNode
        config = SawyerConfig(max_vram_gb=24.0, max_experts=4)
        node = SawyerNode(config)
        asyncio.run(node.start(offline=True))

        # Add experts manually
        for i in range(3):
            slot = ExpertSlot(
                model_name="mixtral-8x7b",
                expert_id=i,
                vram_bytes=int(4.9 * 1024**3),
                status="ready",
            )
            node.expert_server.experts[f"mixtral-8x7b:{i}"] = slot

        report = asyncio.run(node.heartbeat())
        assert report.experts_loaded == 3
        assert report.experts_ready == 3
        assert report.is_healthy is True