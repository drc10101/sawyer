"""Sawyer Node Agent — hosts experts, serves inference, reports health."""

import logging
import time
from dataclasses import dataclass, field

from sawyer.config import SawyerConfig

logger = logging.getLogger(__name__)


@dataclass
class ExpertSlot:
    """An expert loaded into a node's VRAM."""

    model_name: str
    expert_id: int
    loaded_at: float = field(default_factory=time.time)
    inference_count: int = 0


class SawyerNode:
    """Sawyer node agent.

    Registers with the network, downloads and hosts expert weights,
    serves inference requests, and reports health to the router.
    """

    def __init__(self, config: SawyerConfig) -> None:
        self.config = config
        self.node_id: str | None = None
        self.experts: dict[str, ExpertSlot] = {}  # key: "model:expert_id"
        self._running = False

    async def register(self, name: str | None = None) -> str:
        """Register this node with the Sawyer network via Bedrock identity.

        Args:
            name: Optional node name. Defaults to hostname.

        Returns:
            Node ID assigned by the network.
        """
        import socket

        node_name = name or self.config.node_name or socket.gethostname()
        logger.info("Registering node %s with Sawyer network", node_name)

        # TODO: Call Bedrock node registration API
        # For now, generate a placeholder ID
        self.node_id = f"sawyer-node-{node_name}"
        logger.info("Registered as %s", self.node_id)
        return self.node_id

    async def load_expert(self, model_name: str, expert_id: int) -> None:
        """Download and load an expert weight file into VRAM.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            expert_id: Expert number within the model
        """
        key = f"{model_name}:{expert_id}"
        if key in self.experts:
            logger.info("Expert %s already loaded", key)
            return

        logger.info("Loading expert %s...", key)
        # TODO: Download from HuggingFace, load into inference backend
        self.experts[key] = ExpertSlot(model_name=model_name, expert_id=expert_id)
        logger.info("Expert %s loaded successfully", key)

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

        # TODO: Forward pass through the expert via vLLM or llama.cpp
        return {
            "expert_id": expert_id,
            "model": model_name,
            "output": "placeholder_tensor",
            "node_id": self.node_id,
        }

    async def start(self) -> None:
        """Start the node agent — register and begin serving."""
        logger.info("Starting Sawyer Node Agent")
        self._running = True
        await self.register()

    async def stop(self) -> None:
        """Stop the node agent."""
        logger.info("Stopping Sawyer Node Agent")
        self._running = False
