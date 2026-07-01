"""Sawyer Node Agent — hosts experts, serves inference, reports health.

Connects to the Sawyer router via gRPC, loads expert weights, and serves
inference requests routed by the gateway. The ExpertServer handles the
actual inference lifecycle; this module handles registration, heartbeat,
and coordination.
"""

import logging
import time

from sawyer.config import SawyerConfig
from sawyer.node.expert_server import ExpertServer, HealthReport
from sawyer.node.inference import LlamaCppBackend
from sawyer.proto import sawyer_pb2
from sawyer.router.client import RouterClient
from sawyer.router.server import SawyerNodeServicer

logger = logging.getLogger(__name__)


class SawyerNode:
    """Sawyer node agent.

    Registers with the network via Bedrock identity, delegates expert
    management to ExpertServer, and reports health to the router.
    """

    def __init__(self, config: SawyerConfig | None = None) -> None:
        self.config = config or SawyerConfig()
        self.node_id: str | None = None
        self.expert_server = ExpertServer(self.config)
        self._running = False
        self._router_client: RouterClient | None = None
        self._node_server: SawyerNodeServicer | None = None
        self._last_heartbeat: float = 0.0

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

        # Set node ID and GPU info on expert server
        self.expert_server.set_node_id(self.node_id)
        self.expert_server.set_gpu_info(gpu_name, vram_bytes)

        # Start local node server for the router to call
        self._node_server = SawyerNodeServicer(self.config)

        logger.info("Registered as %s", self.node_id)
        return self.node_id

    async def load_expert(
        self, model_name: str, expert_id: int, weight_url: str = "", weight_checksum: str = ""
    ) -> None:
        """Download and load an expert weight file into VRAM.

        Delegates to ExpertServer.load_expert().
        """
        await self.expert_server.load_expert(
            model_name=model_name,
            expert_id=expert_id,
            weight_url=weight_url or None,
            weight_checksum=weight_checksum,
        )

    async def unload_expert(self, model_name: str, expert_id: int) -> None:
        """Unload an expert from VRAM.

        Delegates to ExpertServer.unload_expert().
        """
        await self.expert_server.unload_expert(model_name, expert_id)

    async def serve_request(self, model_name: str, expert_id: int, prompt: str, **kwargs) -> dict:
        """Run inference on a single expert.

        Delegates to ExpertServer.forward_pass().

        Args:
            model_name: Model identifier
            expert_id: Expert number
            prompt: Input text prompt
            **kwargs: Additional inference parameters (max_tokens, temperature, etc.)

        Returns:
            Dict with inference result
        """
        result = await self.expert_server.forward_pass(
            model_name=model_name,
            expert_id=expert_id,
            prompt=prompt,
            request_id=kwargs.pop("request_id", ""),
            **kwargs,
        )
        return {
            "expert_id": expert_id,
            "model": model_name,
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "node_id": self.node_id,
        }

    async def heartbeat(self) -> HealthReport:
        """Send a health report to the router.

        Returns:
            The HealthReport that was sent.
        """
        report = self.expert_server.health_report()
        self._last_heartbeat = time.time()

        if not self._router_client:
            logger.warning("Cannot send heartbeat: no router connection")
            return report

        # Convert HealthReport to gRPC NodeStatus
        status = sawyer_pb2.NodeStatus(
            cpu_usage=0.0,
            gpu_usage=0.0,
            vram_used_bytes=report.vram_used_bytes,
            vram_total_bytes=report.vram_total_bytes,
            active_requests=report.active_requests,
            total_inferences=report.total_inferences,
            avg_latency_ms=report.avg_latency_ms,
        )

        self._router_client.heartbeat(status=status)
        logger.debug(
            "Heartbeat sent: %d experts ready, %d inferences, %.1fms avg latency",
            report.experts_ready, report.total_inferences, report.avg_latency_ms,
        )

        return report

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
            self.expert_server.set_node_id(self.node_id)
        else:
            await self.register()

    async def stop(self) -> None:
        """Stop the node agent — deregister and clean up."""
        logger.info("Stopping Sawyer Node Agent")
        self._running = False

        # Gracefully shut down expert server
        await self.expert_server.shutdown()

        if self._router_client:
            self._router_client.deregister()
            self._router_client.close()