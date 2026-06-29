"""Sawyer Inference Backend — interfaces with llama.cpp for MoE expert execution.

Supports two modes:
1. Subprocess: manages a llama.cpp server process locally
2. HTTP: connects to a remote llama.cpp server via OpenAI-compatible API

The inference backend handles:
- Loading GGUF model weights into VRAM
- Routing inference requests to the correct expert
- Managing the MoE gating network
- Streaming token generation
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from sawyer.config import SawyerConfig

logger = logging.getLogger(__name__)


class BackendMode(Enum):
    """Inference backend mode."""

    SUBPROCESS = "subprocess"
    HTTP = "http"


@dataclass
class InferenceResult:
    """Result from an inference request."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_name: str
    expert_ids: list[int] = field(default_factory=list)
    finish_reason: str = "stop"
    token_ids: list[int] = field(default_factory=list)


@dataclass
class BackendStatus:
    """Status of the inference backend."""

    running: bool
    model_loaded: str | None = None
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    uptime_seconds: float = 0.0
    total_inferences: int = 0


class LlamaCppBackend:
    """Inference backend using llama.cpp server.

    Manages a llama.cpp server subprocess for GGUF model inference.
    Supports both local subprocess and remote HTTP modes.
    """

    def __init__(
        self,
        config: SawyerConfig | None = None,
        mode: BackendMode = BackendMode.SUBPROCESS,
        server_url: str | None = None,
        llama_cpp_path: str | None = None,
    ) -> None:
        self.config = config or SawyerConfig()
        self.mode = mode
        self.server_url = server_url or f"http://127.0.0.1:{self.config.inference_port}"
        self.llama_cpp_path = llama_cpp_path or "llama-server"
        self._process: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._model_loaded: str | None = None
        self._start_time: float = 0.0
        self._total_inferences: int = 0

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.server_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    def start_server(
        self,
        model_path: str | Path,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start the llama.cpp server subprocess.

        Args:
            model_path: Path to the GGUF model file
            model_name: Model identifier for tracking
            n_ctx: Context window size
            n_gpu_layers: Number of GPU layers (-1 for all)
            n_threads: Number of threads (None for auto)
            extra_args: Additional llama-server arguments
        """
        if self.mode != BackendMode.SUBPROCESS:
            raise RuntimeError("start_server only valid in subprocess mode")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        cmd = [
            self.llama_cpp_path,
            "-m",
            str(model_path),
            "--host",
            self.server_url.replace("http://", "").rsplit(":", 1)[0] or "127.0.0.1",
            "--port",
            str(self.config.inference_port),
            "-c",
            str(n_ctx),
            "-ngl",
            str(n_gpu_layers),
        ]

        if n_threads:
            cmd.extend(["-t", str(n_threads)])

        if extra_args:
            cmd.extend(extra_args)

        logger.info("Starting llama.cpp server: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._start_time = time.time()
        self._model_loaded = model_name

        # Wait for server to be ready
        self._wait_for_server(timeout=60)

    def _wait_for_server(self, timeout: int = 60) -> None:
        """Wait for the llama.cpp server to become ready."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self.client.get("/health")
                if response.status_code == 200:
                    logger.info("llama.cpp server ready")
                    return
            except httpx.ConnectError:
                pass
            time.sleep(1.0)

        raise TimeoutError(f"llama.cpp server not ready after {timeout}s")

    def stop_server(self) -> None:
        """Stop the llama.cpp server subprocess."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._model_loaded = None
            logger.info("llama.cpp server stopped")

    def load_model(self, model_name: str, model_path: str | Path | None = None) -> None:
        """Load a model for inference.

        In subprocess mode, starts a new server with the model.
        In HTTP mode, just tracks which model is loaded remotely.
        """
        if self.mode == BackendMode.SUBPROCESS:
            if self._process:
                self.stop_server()

            if model_path is None:
                from sawyer.node.weights import WeightLoader

                loader = WeightLoader(self.config)
                if not loader.is_cached(model_name):
                    raise FileNotFoundError(f"Model {model_name} not cached. Download it first.")
                model_path = loader.get_cached_path(model_name)

            self.start_server(model_path, model_name)
        else:
            self._model_loaded = model_name

    def infer(
        self,
        prompt: str,
        model_name: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repeat_penalty: float = 1.1,
        seed: int = -1,
        stream: bool = False,
    ) -> InferenceResult:
        """Run inference on the loaded model.

        Args:
            prompt: Input text prompt
            model_name: Override model name for tracking
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repeat_penalty: Repetition penalty
            seed: Random seed (-1 for random)
            stream: Whether to stream tokens

        Returns:
            InferenceResult with generated text and metrics
        """
        start_time = time.time()

        # Use OpenAI-compatible completion API
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
            "stream": stream,
        }

        response = self.client.post("/completion", json=payload)
        response.raise_for_status()
        result = response.json()

        latency_ms = (time.time() - start_time) * 1000
        self._total_inferences += 1

        return InferenceResult(
            text=result.get("content", ""),
            input_tokens=result.get("prompt_tokens", 0),
            output_tokens=result.get("tokens_evaluated", 0),
            latency_ms=latency_ms,
            model_name=model_name or self._model_loaded or "unknown",
            finish_reason=result.get("stop") and "stop" or "length",
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        model_name: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> InferenceResult:
        """Run chat-completion inference using OpenAI-compatible API.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            model_name: Override model name for tracking
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InferenceResult with generated text and metrics
        """
        start_time = time.time()

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()

        latency_ms = (time.time() - start_time) * 1000
        self._total_inferences += 1

        choices = result.get("choices", [{}])
        content = choices[0].get("message", {}).get("content", "") if choices else ""

        usage = result.get("usage", {})
        return InferenceResult(
            text=content,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            model_name=model_name or self._model_loaded or "unknown",
            finish_reason=choices[0].get("finish_reason", "stop") if choices else "error",
        )

    def get_status(self) -> BackendStatus:
        """Get the current status of the inference backend."""
        if self.mode == BackendMode.SUBPROCESS and self._process:
            running = self._process.poll() is None
        else:
            # HTTP mode — check if server responds
            try:
                resp = self.client.get("/health")
                running = resp.status_code == 200
            except httpx.ConnectError:
                running = False

        uptime = time.time() - self._start_time if self._start_time else 0.0

        return BackendStatus(
            running=running,
            model_loaded=self._model_loaded,
            uptime_seconds=uptime,
            total_inferences=self._total_inferences,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get info about the currently loaded model from llama.cpp."""
        try:
            response = self.client.get("/props")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return {}

    def close(self) -> None:
        """Clean up resources."""
        if self.mode == BackendMode.SUBPROCESS:
            self.stop_server()
        if self._client:
            self._client.close()
            self._client = None
