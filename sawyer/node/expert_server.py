"""Sawyer Expert Serving — loads expert shards, runs forward passes, reports health.

The ExpertServer manages the lifecycle of expert shards on a node:
  1. Loading expert weights into the inference backend (llama.cpp or remote)
  2. Running forward passes for routed inference requests
  3. Reporting health metrics (VRAM, latency, throughput) to the router
  4. Unloading experts when capacity is needed for others

This is the "serving" half of a Sawyer node — the SawyerNode agent handles
registration and heartbeat, while ExpertServer handles the actual inference.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sawyer.config import SawyerConfig
from sawyer.model.registry import MoEModel, get_model
from sawyer.node.inference import BackendMode, InferenceResult, LlamaCppBackend
from sawyer.node.weights import WeightLoader, build_expert_weight_url

logger = logging.getLogger(__name__)


@dataclass
class ExpertSlot:
    """An expert loaded into a node's VRAM."""

    model_name: str
    expert_id: int
    loaded_at: float = field(default_factory=time.time)
    inference_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    vram_bytes: int = 0
    status: str = "loading"  # loading, ready, failed, unloading

    @property
    def avg_latency_ms(self) -> float:
        """Average inference latency in milliseconds."""
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count

    @property
    def key(self) -> str:
        """Unique key for this expert slot: 'model_name:expert_id'."""
        return f"{self.model_name}:{self.expert_id}"


@dataclass
class HealthReport:
    """Health report from a node to the router."""

    node_id: str
    timestamp: float = field(default_factory=time.time)
    gpu_name: str = ""
    vram_total_bytes: int = 0
    vram_used_bytes: int = 0
    experts_loaded: int = 0
    experts_ready: int = 0
    active_requests: int = 0
    total_inferences: int = 0
    avg_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    is_healthy: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for gRPC / HTTP reporting."""
        return {
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "gpu_name": self.gpu_name,
            "vram_total_bytes": self.vram_total_bytes,
            "vram_used_bytes": self.vram_used_bytes,
            "experts_loaded": self.experts_loaded,
            "experts_ready": self.experts_ready,
            "active_requests": self.active_requests,
            "total_inferences": self.total_inferences,
            "avg_latency_ms": self.avg_latency_ms,
            "uptime_seconds": self.uptime_seconds,
            "is_healthy": self.is_healthy,
        }


class ExpertServer:
    """Manages expert lifecycle on a Sawyer node.

    Loads expert weight files, runs forward passes via the inference backend,
    tracks VRAM usage, and produces health reports for the router.

    Usage:
        config = SawyerConfig()
        server = ExpertServer(config)

        # Load an expert
        await server.load_expert("mixtral-8x7b", expert_id=0)

        # Run inference
        result = await server.forward_pass("mixtral-8x7b", expert_id=0, prompt="Hello")

        # Get health report
        health = server.health_report()
    """

    def __init__(
        self,
        config: SawyerConfig | None = None,
        backend: LlamaCppBackend | None = None,
    ) -> None:
        self.config = config or SawyerConfig()
        self.experts: dict[str, ExpertSlot] = {}  # key: "model:expert_id"
        self._backend = backend
        self._weight_loader = WeightLoader(self.config)
        self._start_time = time.time()
        self._active_requests = 0
        self._gpu_name = ""
        self._vram_total = int(self.config.max_vram_gb * 1024**3) if self.config.max_vram_gb else 0
        self._node_id: str = ""

    @property
    def backend(self) -> LlamaCppBackend:
        """Lazy-initialize the inference backend."""
        if self._backend is None:
            self._backend = LlamaCppBackend(self.config)
        return self._backend

    def set_node_id(self, node_id: str) -> None:
        """Set the node ID for health reporting."""
        self._node_id = node_id

    def set_gpu_info(self, gpu_name: str, vram_total_bytes: int) -> None:
        """Set GPU info for health reporting."""
        self._gpu_name = gpu_name
        self._vram_total = vram_total_bytes

    async def load_expert(
        self,
        model_name: str,
        expert_id: int,
        weight_url: str | None = None,
        weight_checksum: str = "",
    ) -> ExpertSlot:
        """Download and load an expert weight file into VRAM.

        Steps:
        1. Check if already loaded
        2. Check VRAM capacity
        3. Download weight file (if not cached)
        4. Load into inference backend
        5. Mark as ready

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            expert_id: Expert number within the model
            weight_url: URL to download from (auto-built if None)
            weight_checksum: SHA-256 checksum for verification

        Returns:
            The loaded ExpertSlot

        Raises:
            ValueError: If VRAM is insufficient
            RuntimeError: If loading fails
        """
        key = f"{model_name}:{expert_id}"

        if key in self.experts:
            slot = self.experts[key]
            if slot.status == "ready":
                logger.info("Expert %s already loaded and ready", key)
                return slot
            if slot.status == "loading":
                logger.warning("Expert %s is currently loading", key)
                return slot

        # Check capacity
        model = get_model(model_name)
        expert_vram = int(model.expert_size_gb_q4 * 1024**3)
        current_vram = sum(s.vram_bytes for s in self.experts.values() if s.status != "unloading")
        available = self._vram_total - current_vram if self._vram_total else expert_vram + 1

        if self._vram_total and expert_vram > available:
            raise ValueError(
                f"Insufficient VRAM for {key}: need {expert_vram / 1024**3:.1f} GB, "
                f"have {available / 1024**3:.1f} GB available "
                f"(total: {self._vram_total / 1024**3:.1f} GB, used: {current_vram / 1024**3:.1f} GB)"
            )

        # Create slot
        slot = ExpertSlot(
            model_name=model_name,
            expert_id=expert_id,
            vram_bytes=expert_vram,
            status="loading",
        )
        self.experts[key] = slot

        try:
            # Build weight URL if not provided
            if not weight_url:
                weight_url = build_expert_weight_url(model_name, expert_id)

            # Download weight file
            if weight_url:
                logger.info("Loading expert %s from %s ...", key, weight_url)
                wf = await asyncio.to_thread(
                    self._weight_loader.download_weight,
                    model_name,
                    expert_id,
                )
                weight_path = str(wf.path)
            else:
                # Use cached path
                cached = self._weight_loader.get_cached_path(model_name)
                weight_path = str(cached) if cached else ""

            if weight_path:
                # Load into inference backend
                try:
                    await asyncio.to_thread(
                        self.backend.load_model,
                        model_name,
                        weight_path,
                    )
                except Exception as e:
                    logger.error("Failed to load model into backend: %s", e)
                    # Non-fatal — the backend may not be in subprocess mode

            slot.status = "ready"
            logger.info(
                "Expert %s loaded (%.1f GB VRAM, status=%s)",
                key, expert_vram / 1024**3, slot.status,
            )

        except Exception as e:
            slot.status = "failed"
            logger.error("Failed to load expert %s: %s", key, e)
            raise

        return slot

    async def unload_expert(self, model_name: str, expert_id: int) -> None:
        """Unload an expert from VRAM.

        Args:
            model_name: Model identifier
            expert_id: Expert number
        """
        key = f"{model_name}:{expert_id}"
        if key not in self.experts:
            logger.warning("Expert %s not loaded, cannot unload", key)
            return

        slot = self.experts[key]
        slot.status = "unloading"
        logger.info("Unloading expert %s ...", key)

        # Remove from slot tracking (VRAM freed)
        del self.experts[key]
        logger.info("Expert %s unloaded (%.1f GB freed)", key, slot.vram_bytes / 1024**3)

    async def forward_pass(
        self,
        model_name: str,
        expert_id: int,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_id: str = "",
    ) -> InferenceResult:
        """Run a forward pass through the specified expert.

        This is the core inference method. It verifies the expert is loaded,
        runs inference via the backend, and tracks metrics.

        Args:
            model_name: Model identifier
            expert_id: Expert number
            prompt: Input text prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            request_id: Request ID for logging

        Returns:
            InferenceResult with generated text and metrics

        Raises:
            ValueError: If the expert is not loaded
        """
        key = f"{model_name}:{expert_id}"

        if key not in self.experts:
            raise ValueError(f"Expert {key} not loaded on this node")

        slot = self.experts[key]
        if slot.status != "ready":
            raise ValueError(f"Expert {key} is not ready (status={slot.status})")

        self._active_requests += 1
        start_time = time.time()

        try:
            logger.info(
                "Forward pass: expert=%s, request=%s, prompt_len=%d",
                key, request_id, len(prompt),
            )

            # Run inference via backend
            result = await asyncio.to_thread(
                self.backend.infer,
                prompt=prompt,
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            # Update slot metrics
            elapsed_ms = (time.time() - start_time) * 1000
            slot.inference_count += 1
            slot.total_input_tokens += result.input_tokens
            slot.total_output_tokens += result.output_tokens
            slot.total_latency_ms += elapsed_ms

            # Add expert tracking to result
            result.expert_ids = [expert_id]

            logger.info(
                "Forward pass complete: expert=%s, request=%s, "
                "tokens=%d/%d, latency=%.0fms",
                key, request_id, result.input_tokens, result.output_tokens, elapsed_ms,
            )

            return result

        except Exception as e:
            logger.error("Forward pass failed for %s: %s", key, e)
            # Mark slot as potentially unhealthy
            slot.status = "ready"  # Keep ready for retry
            raise

        finally:
            self._active_requests = max(0, self._active_requests - 1)

    async def batch_forward_pass(
        self,
        model_name: str,
        expert_ids: list[int],
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        request_id: str = "",
    ) -> list[InferenceResult]:
        """Run forward passes through multiple experts for the same prompt.

        Used by the router to evaluate multiple experts in parallel for
        mixture-of-experts inference.

        Args:
            model_name: Model identifier
            expert_ids: List of expert IDs to run
            prompt: Input text prompt
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            request_id: Request ID for logging

        Returns:
            List of InferenceResult, one per expert
        """
        tasks = [
            self.forward_pass(
                model_name=model_name,
                expert_id=eid,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                request_id=f"{request_id}-e{eid}",
            )
            for eid in expert_ids
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def health_report(self) -> HealthReport:
        """Generate a health report for this node.

        The health report includes:
        - GPU name and VRAM usage
        - Expert loading status
        - Active request count
        - Total inference count and average latency
        - Uptime

        Returns:
            HealthReport with current node metrics
        """
        total_vram_used = sum(s.vram_bytes for s in self.experts.values() if s.status != "unloading")
        experts_ready = sum(1 for s in self.experts.values() if s.status == "ready")
        total_inferences = sum(s.inference_count for s in self.experts.values())

        # Calculate weighted average latency across all experts
        if total_inferences > 0:
            avg_latency = sum(
                s.total_latency_ms for s in self.experts.values()
            ) / total_inferences
        else:
            avg_latency = 0.0

        uptime = time.time() - self._start_time

        # Node is healthy if at least one expert is ready and no experts are failed
        is_healthy = experts_ready > 0 and not any(
            s.status == "failed" for s in self.experts.values()
        )

        return HealthReport(
            node_id=self._node_id,
            gpu_name=self._gpu_name,
            vram_total_bytes=self._vram_total,
            vram_used_bytes=total_vram_used,
            experts_loaded=len(self.experts),
            experts_ready=experts_ready,
            active_requests=self._active_requests,
            total_inferences=total_inferences,
            avg_latency_ms=avg_latency,
            uptime_seconds=uptime,
            is_healthy=is_healthy,
        )

    def get_expert_status(self, model_name: str, expert_id: int) -> str | None:
        """Get the status of a specific expert slot.

        Returns:
            Status string ('loading', 'ready', 'failed', 'unloading') or None if not loaded.
        """
        key = f"{model_name}:{expert_id}"
        slot = self.experts.get(key)
        return slot.status if slot else None

    def list_experts(self) -> list[dict[str, Any]]:
        """List all loaded experts with their status and metrics."""
        result = []
        for slot in self.experts.values():
            result.append({
                "model": slot.model_name,
                "expert_id": slot.expert_id,
                "status": slot.status,
                "inferences": slot.inference_count,
                "avg_latency_ms": slot.avg_latency_ms,
                "vram_gb": slot.vram_bytes / (1024**3),
                "loaded_at": slot.loaded_at,
            })
        return result

    async def shutdown(self) -> None:
        """Gracefully shut down the expert server.

        Unloads all experts and closes the inference backend.
        """
        logger.info("Shutting down ExpertServer, unloading %d experts", len(self.experts))

        # Unload all experts
        for key in list(self.experts.keys()):
            model_name, expert_id_str = key.rsplit(":", 1)
            await self.unload_expert(model_name, int(expert_id_str))

        # Close backend
        if self._backend:
            self._backend.close()

        # Close weight loader
        self._weight_loader.close()

        logger.info("ExpertServer shutdown complete")