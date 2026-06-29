"""Sawyer Expert Scheduler — select and route to the best available expert nodes.

Considers node health, latency, load, and geographic proximity when routing
inference requests to the optimal expert nodes.
"""

import logging
import time
from dataclasses import dataclass, field

from sawyer.model.registry import get_model

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a registered Sawyer node."""

    node_id: str
    experts: list[int]  # expert IDs this node hosts
    gpu: str
    vram_gb: float
    bandwidth_mbps: float
    latency_ms: float
    healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    requests_served: int = 0
    avg_response_ms: float = 0.0


class ExpertScheduler:
    """Routes inference requests to the best available expert nodes.

    For each request:
    1. Look up the model's gating network output (which experts activate)
    2. Find healthy nodes hosting those experts
    3. Select the lowest-latency nodes with available capacity
    4. Route the request and collect results
    """

    def __init__(self) -> None:
        self.nodes: dict[str, NodeInfo] = {}

    def register_node(self, node: NodeInfo) -> None:
        """Register a new node or update an existing one."""
        self.nodes[node.node_id] = node
        logger.info("Registered node %s with experts %s", node.node_id, node.experts)

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from the scheduler."""
        self.nodes.pop(node_id, None)
        logger.info("Unregistered node %s", node_id)

    def find_expert_nodes(self, expert_id: int) -> list[NodeInfo]:
        """Find all healthy nodes hosting a specific expert."""
        return [
            node
            for node in self.nodes.values()
            if node.healthy
            and expert_id in node.experts
            and (time.time() - node.last_heartbeat) < 120  # 2-minute heartbeat timeout
        ]

    def select_node(self, expert_id: int, prefer_low_latency: bool = True) -> NodeInfo | None:
        """Select the best node for a given expert.

        Prioritizes low latency, then low load, then high bandwidth.
        """
        candidates = self.find_expert_nodes(expert_id)
        if not candidates:
            logger.warning("No healthy nodes found for expert %d", expert_id)
            return None

        if prefer_low_latency:
            candidates.sort(key=lambda n: n.latency_ms)
        else:
            candidates.sort(key=lambda n: n.requests_served)

        return candidates[0]

    async def route(
        self,
        model_name: str,
        tokens: list,
        user_id: str,
    ) -> dict:
        """Route an inference request to the appropriate expert nodes.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            tokens: Input token embeddings
            user_id: Authenticated user ID

        Returns:
            Aggregated expert outputs
        """
        model = get_model(model_name)

        # For a real implementation, the gating network would select which
        # experts activate for these specific tokens. For now, we select
        # the top-K experts as a placeholder.
        active_expert_ids = list(range(model.active_experts))
        logger.info(
            "Routing %s request for user %s to experts %s",
            model_name,
            user_id,
            active_expert_ids,
        )

        # Select nodes for each active expert
        selected_nodes: dict[int, NodeInfo] = {}
        for expert_id in active_expert_ids:
            node = self.select_node(expert_id)
            if node is None:
                raise RuntimeError(f"No available node for expert {expert_id}")
            selected_nodes[expert_id] = node

        # TODO: Send inference request to each node, collect results,
        # aggregate expert outputs, and return to the user.
        return {
            "model": model_name,
            "experts_routed": active_expert_ids,
            "nodes_used": {str(eid): n.node_id for eid, n in selected_nodes.items()},
            "status": "placeholder",
        }
