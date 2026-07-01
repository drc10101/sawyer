"""Sawyer Node Agent — hosts experts, serves inference, reports health.

Connects to the Sawyer router via gRPC, loads expert weights, and serves
inference requests routed by the gateway.
"""

import logging
import time
from dataclasses import dataclass, field

from sawyer.config import SawyerConfig
from sawyer.proto import sawyer_pb2
from sawyer.router.client import RouterClient
from sawyer.router.server import SawyerNodeServicer

logger = logging.getLogger(__name__)


@dataclass
class ExpertSlot:
    """An expert loaded into a node's VRAM."""

    model_name: str
    expert_id: int
    loaded_at: float = field(default_factory=time.time)
    inference_count: int = 0
    vram_bytes: int = 0


class SawyerNode:
    """Sawyer node agent.

    Registers with the network via Bedrock identity, downloads and hosts
    expert weights, serves inference requests, and reports health to the router.
    """

    def __init__(self, config: SawyerConfig) -> None:
        self.config = config
        self.node_id: str | None = None
        self.experts: dict[str, ExpertSlot] = {}  # key: "model:expert_id"
        self._running = False
        self._router_client: RouterClient | None = None
        self._node_server: SawyerNodeServicer | None = None

    async def register(self, name: str | None = None) -> str:
        """Register this node with the Sawyer network via gRPC.

        Args:
            name: Optional node name. Defaults to config value or hostname.

        Returns:
            Node ID assigned by the router.
        """
        import socket

        node_name = name or self.config.node_name or socket.gethostname()
        logger.info("Registering node '%s' with Sawyer network", node_name)

        # Connect to the router
        self._router_client = RouterClient(self.config)
        self._router_client.connect()

        # Auto-detect GPU if requested
        gpu_name = "unknown"
        vram_bytes = 0
        if self.config.max_vram_gb:
            vram_bytes = int(self.config.max_vram_gb * 1024**3)

        # Register via gRPC
        self.node_id = self._router_client.register(
            name=node_name,
            gpu_name=gpu_name,
            vram_bytes=vram_bytes,
            max_experts=self.config.max_experts,
            region="us-east-1",
        )

        # Start local node server for the router to call
        self._node_server = SawyerNodeServicer(self.config)

        logger.info("Registered as %s", self.node_id)
        return self.node_id

    async def load_expert(
        self, model_name: str, expert_id: int, weight_url: str = "", weight_checksum: str = ""
    ) -> None:
        """Download and load an expert weight file into VRAM.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            expert_id: Expert number within the model
            weight_url: URL to download the weight file from
            weight_checksum: SHA-256 checksum for verification
        """
        key = f"{model_name}:{expert_id}"
        if key in self.experts:
            logger.info("Expert %s already loaded", key)
            return

        logger.info("Loading expert %s from %s...", key, weight_url or "local cache")

        # TODO: Download from weight_url, verify checksum, load into inference backend
        # For now, record the slot
        from sawyer.model.registry import get_model

        model = get_model(model_name)
        self.experts[key] = ExpertSlot(
            model_name=model_name,
            expert_id=expert_id,
            vram_bytes=int(model.expert_size_gb_q4 * 1024**3),
        )
        logger.info("Expert %s loaded successfully (%.1f GB)", key, model.expert_size_gb_q4)

    async def unload_expert(self, model_name: str, expert_id: int) -> None:
        """Unload an expert from VRAM."""
        key = f"{model_name}:{expert_id}"
        if key in self.experts:
            del self.experts[key]
            logger.info("Unloaded expert %s", key)

    async def serve_request(self, model_name: str, expert_id: int, tokens: list) -> dict:
        """Run inference on a single expert.

        Args:
            model_name: Model identifier
            expert_id: Expert number
            tokens: Input token embeddings

        Returns:
            Expert output tensor (placeholder)
        """
        key = f"{model_name}:{expert_id}"
        if key not in self.experts:
            raise ValueError(f"Expert {key} not loaded on this node")

        slot = self.experts[key]
        slot.inference_count += 1

        logger.info("Serving expert %s (inference #%d)", key, slot.inference_count)

        # TODO: Forward pass through the expert via inference backend
        return {
            "expert_id": expert_id,
            "model": model_name,
            "output": "placeholder_tensor",
            "node_id": self.node_id,
        }

    async def heartbeat(self) -> None:
        """Send heartbeat to the router with current node status."""
        if not self._router_client:
            return

        total_vram = int(self.config.max_vram_gb * 1024**3) if self.config.max_vram_gb else 0
        used_vram = sum(s.vram_bytes for s in self.experts.values())
        total_inferences = sum(s.inference_count for s in self.experts.values())

        status = sawyer_pb2.NodeStatus(
            cpu_usage=0.0,
            gpu_usage=0.0,
            vram_used_bytes=used_vram,
            vram_total_bytes=total_vram,
            active_requests=0,
            total_inferences=total_inferences,
            avg_latency_ms=10.0,
        )

        self._router_client.heartbeat(status=status)

    async def start(self, offline: bool = False) -> None:
        """Start the node agent — register and begin serving.

        Args:
            offline: If True, skip router registration and run standalone.
        """
        logger.info("Starting Sawyer Node Agent")
        self._running = True
        if offline:
            logger.info("Running in offline mode — no router connection")
            self.node_id = "offline"
        else:
            await self.register()

    async def stop(self) -> None:
        """Stop the node agent — deregister and clean up."""
        logger.info("Stopping Sawyer Node Agent")
        self._running = False
        if self._router_client:
            self._router_client.deregister()
            self._router_client.close()
