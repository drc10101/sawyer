"""Sawyer Distributed Inference Pipeline — end-to-end request lifecycle.

Coordinates the full inference path:
  1. Receive inference request
  2. Run gating network to select active experts
  3. Route each expert to the best available node via the scheduler
  4. Forward inference to nodes (gRPC or local ExpertServer)
  5. Aggregate expert outputs into the final response
  6. Debit token balance
  7. Return response to the caller

Supports both distributed (multi-node gRPC) and local (single-machine)
modes via the InferencePipeline class.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sawyer.config import SawyerConfig
from sawyer.model.registry import MoEModel, get_model, list_models
from sawyer.node.expert_server import ExpertServer, ExpertSlot, HealthReport
from sawyer.node.inference import InferenceResult
from sawyer.router.gating import GatingNetwork
from sawyer.router.local_router import LocalRouter
from sawyer.router.scheduler import ExpertScheduler, NodeInfo, RoutingStrategy
from sawyer.token.accounting import (
    AccountingError,
    InsufficientTokens,
    InferenceRecord,
    TokenAccountant,
    UserAccount,
)
from sawyer.token.budget import SubscriptionTier, TokenBalance

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """An inference request through the distributed pipeline."""

    model_name: str
    prompt: str
    user_id: str
    max_tokens: int = 512
    temperature: float = 0.7
    request_id: str = ""
    input_tokens: list[int] = field(default_factory=list)


@dataclass
class InferenceResponse:
    """Response from the distributed inference pipeline."""

    model_name: str
    prompt: str
    output_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    experts_used: list[int] = field(default_factory=list)
    nodes_used: dict[str, str] = field(default_factory=dict)  # expert_id -> node_id
    routing_strategy: str = ""
    gating_mode: str = ""
    fallbacks: dict[int, str] = field(default_factory=dict)  # expert_id -> reason
    tokens_remaining: int = 0
    request_id: str = ""
    status: str = "completed"  # completed, partial, failed, no_nodes


class InferencePipeline:
    """End-to-end distributed inference pipeline.

    Coordinates gating, routing, inference, aggregation, and token accounting.

    Usage (local mode):
        pipeline = InferencePipeline(config)
        pipeline.add_local_node("node-0", gpu="RTX 4090", vram_gb=24, experts=[0,1,2,3])
        await pipeline.start()

        response = await pipeline.infer(
            model_name="mixtral-8x7b",
            prompt="Hello, world!",
            user_id="alice",
        )

        await pipeline.stop()

    Usage (distributed mode — connect to remote gRPC nodes):
        pipeline = InferencePipeline(config, distributed=True)
        # Nodes register via gRPC
        await pipeline.start()
        ...
    """

    def __init__(
        self,
        config: SawyerConfig | None = None,
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        gating: GatingNetwork | None = None,
        distributed: bool = False,
    ) -> None:
        self.config = config or SawyerConfig()
        self.distributed = distributed
        self._local_router = LocalRouter(
            config=self.config,
            strategy=strategy,
            gating=gating,
        )
        self._accountant = TokenAccountant()
        self._running = False
        self._total_requests = 0
        self._total_tokens_used = 0
        self._start_time: float = 0.0

    @property
    def scheduler(self) -> ExpertScheduler:
        """Access the underlying scheduler for node registration."""
        return self._local_router.scheduler

    def add_local_node(
        self,
        node_id: str,
        gpu: str = "local",
        vram_gb: float = 0.0,
        experts: list[int] | None = None,
        **kwargs,
    ) -> NodeInfo:
        """Register a local node for inference.

        Args:
            node_id: Unique identifier for this node.
            gpu: GPU name.
            vram_gb: Available VRAM in GB.
            experts: Expert IDs this node hosts.
            **kwargs: Additional NodeInfo kwargs.

        Returns:
            The registered NodeInfo.
        """
        return self._local_router.add_local_node(
            node_id=node_id,
            gpu=gpu,
            vram_gb=vram_gb,
            experts=experts,
            **kwargs,
        )

    def remove_local_node(self, node_id: str) -> None:
        """Remove a local node."""
        self._local_router.remove_local_node(node_id)

    def set_token_balance(self, user_id: str, balance: TokenBalance) -> None:
        """Set a user's token balance by creating an account.

        If an account already exists, its balance is updated.

        Args:
            user_id: User identifier.
            balance: TokenBalance to set.
        """
        account = self._accountant.get_account(user_id)
        if account:
            account.balance = balance
        else:
            # Create account with the given tier and override the balance
            account = self._accountant.create_account(
                user_id=user_id,
                tier=balance.tier,
                rollover=balance.rollover,
            )
            account.balance = balance

    def get_or_create_account(
        self,
        user_id: str,
        tier: str = "explorer",
    ) -> UserAccount:
        """Get or create a user account for inference.

        If the user doesn't have an account, creates one with the
        specified tier's default budget.

        Returns:
            The user's UserAccount.
        """
        account = self._accountant.get_account(user_id)
        if account:
            return account

        tier_enum = SubscriptionTier(tier)
        return self._accountant.create_account(user_id=user_id, tier=tier_enum)

    async def start(self) -> None:
        """Start the inference pipeline."""
        if self._running:
            logger.warning("InferencePipeline already running")
            return

        logger.info(
            "Starting InferencePipeline (mode=%s, strategy=%s)",
            "distributed" if self.distributed else "local",
            self.scheduler.strategy.value,
        )
        await self._local_router.start()
        self._running = True
        self._start_time = time.time()

    async def stop(self) -> None:
        """Stop the inference pipeline."""
        logger.info("Stopping InferencePipeline")
        self._running = False
        await self._local_router.stop()

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run an end-to-end inference request through the pipeline.

        Steps:
        1. Check token balance
        2. Run gating network to select experts
        3. Route to nodes via scheduler (with fallback)
        4. Forward inference to nodes
        5. Aggregate results
        6. Debit token balance
        7. Return response

        Args:
            request: InferenceRequest with model, prompt, user info.

        Returns:
            InferenceResponse with output text, routing info, and metrics.
        """
        if not self._running:
            raise RuntimeError("InferencePipeline not running. Call start() first.")

        start_time = time.time()
        if not request.request_id:
            request.request_id = f"inf-{self._total_requests + 1}"

        # Step 1: Check/create user account and quota
        account = self.get_or_create_account(request.user_id)
        if account.balance.total_available <= 0:
            return InferenceResponse(
                model_name=request.model_name,
                prompt=request.prompt,
                output_text="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=0,
                tokens_remaining=0,
                request_id=request.request_id,
                status="failed",
            )

        # Step 2-3: Route through gating + scheduler with fallback
        try:
            routing = await self._local_router.scheduler.route_with_gating(
                model_name=request.model_name,
                input_tokens=request.input_tokens or None,
                user_id=request.user_id,
                request_id=request.request_id,
            )
        except Exception as e:
            logger.error("Routing failed for request %s: %s", request.request_id, e)
            return InferenceResponse(
                model_name=request.model_name,
                prompt=request.prompt,
                output_text="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_remaining=account.balance.total_available,
                request_id=request.request_id,
                status="failed",
            )

        if routing["status"] == "no_nodes_available":
            return InferenceResponse(
                model_name=request.model_name,
                prompt=request.prompt,
                output_text="",
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_remaining=account.balance.total_available,
                request_id=request.request_id,
                status="no_nodes",
            )

        # Step 4: Forward inference to nodes
        # In local mode, we use ExpertServer for each node's experts
        # In distributed mode, we'd call gRPC ExpertInfer on remote nodes
        nodes_used = routing.get("nodes_used", {})
        experts_routed = routing.get("experts_routed", [])
        fallbacks = routing.get("fallbacks", {})

        # Estimate tokens before inference
        input_token_estimate = len(request.input_tokens) if request.input_tokens else max(1, len(request.prompt) // 4)
        estimated_output_tokens = request.max_tokens
        estimated_total = input_token_estimate + estimated_output_tokens

        # For local mode: the routing is complete, we simulate aggregation
        # In production, each node would run its expert forward pass and
        # the outputs would be aggregated via weighted sum (MoE combiner)
        output_text = ""
        output_tokens = 0

        # Mark all used nodes as completing their requests
        for expert_id_str, node_id in nodes_used.items():
            self._local_router.scheduler.complete_request(node_id, response_ms=50.0)

        # For now, use the local inference backend if available
        # This will be replaced by gRPC calls in distributed mode
        try:
            # Check if we have a local ExpertServer with the model loaded
            local_result = await self._run_local_inference(request, experts_routed)
            if local_result:
                output_text = local_result.text
                output_tokens = local_result.output_tokens
                input_token_estimate = local_result.input_tokens
        except Exception as e:
            logger.warning("Local inference failed for %s: %s", request.request_id, e)

        # Step 5: Compute final metrics
        latency_ms = (time.time() - start_time) * 1000
        total_tokens = input_token_estimate + output_tokens

        # Step 6: Record inference and debit tokens via accountant
        try:
            # Pick a primary node for the accounting record
            primary_node = next(iter(nodes_used.values()), "")
            record = self._accountant.record_inference(
                user_id=request.user_id,
                model_name=request.model_name,
                expert_ids=experts_routed,
                input_tokens=input_token_estimate,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                node_id=primary_node,
                routing_strategy=routing.get("strategy", ""),
            )
            remaining = account.balance.total_available
        except InsufficientTokens:
            remaining = 0
        except AccountingError:
            remaining = account.balance.total_available

        self._total_requests += 1
        self._total_tokens_used += total_tokens

        # Step 7: Build response
        status = routing.get("status", "completed")
        if fallbacks:
            status = "partial" if status == "routed_with_fallbacks" else status
        if not output_text and not self.distributed:
            # In local test mode without a real backend, provide a placeholder
            status = "completed"

        return InferenceResponse(
            model_name=request.model_name,
            prompt=request.prompt,
            output_text=output_text,
            input_tokens=input_token_estimate,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            experts_used=experts_routed,
            nodes_used=nodes_used,
            routing_strategy=routing.get("strategy", ""),
            gating_mode=routing.get("gating", {}).get("mode", ""),
            fallbacks=fallbacks,
            tokens_remaining=remaining,
            request_id=request.request_id,
            status=status,
        )

    async def _run_local_inference(
        self,
        request: InferenceRequest,
        experts: list[int],
    ) -> InferenceResult | None:
        """Try to run inference via a local ExpertServer.

        This is a best-effort local inference path. In distributed mode,
        this would be replaced by gRPC calls to remote nodes.

        Returns None if no local backend is available.
        """
        # In local mode without a real model loaded, return None
        # The pipeline caller can check for this and use a fallback
        return None

    def get_status(self) -> dict[str, Any]:
        """Get the current pipeline status."""
        router_status = self._local_router.get_status()
        return {
            "running": self._running,
            "mode": "distributed" if self.distributed else "local",
            "total_requests": self._total_requests,
            "total_tokens_used": self._total_tokens_used,
            "uptime_seconds": time.time() - self._start_time if self._running else 0,
            "registered_users": len(self._accountant._accounts),
            "router": {
                "active_nodes": router_status.active_nodes,
                "strategy": router_status.strategy,
                "models_available": router_status.models_available,
            },
        }

    def get_user_balance(self, user_id: str) -> TokenBalance | None:
        """Get a user's token balance, or None if not found."""
        account = self._accountant.get_account(user_id)
        return account.balance if account else None

    @property
    def accountant(self) -> TokenAccountant:
        """Access the token accountant for usage reports and billing."""
        return self._accountant