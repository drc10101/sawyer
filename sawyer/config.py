"""Sawyer configuration management."""

from dataclasses import dataclass


@dataclass
class SawyerConfig:
    """Sawyer node configuration."""

    # Node settings
    node_name: str | None = None
    router_url: str = "https://router.sawyer.dev"
    inference_port: int = 8444

    # Bedrock integration
    bedrock_url: str = "https://localhost:8443"
    bedrock_license_key: str | None = None

    # Inference backend
    inference_backend: str = "auto"  # auto, vllm, llama_cpp

    # Token settings
    api_key: str | None = None

    # Expert hosting
    cache_dir: str = "~/.sawyer/cache"
    expert_cache_dir: str = "~/.sawyer/experts"
    max_experts: int = 2  # Max experts to host concurrently
    max_vram_gb: float | None = None  # Auto-detect if None

    # Performance
    request_timeout: float = 30.0  # seconds
    heartbeat_interval: float = 60.0  # seconds
    max_concurrent_requests: int = 10

    def validate(self) -> None:
        """Validate configuration values."""
        if self.inference_port < 1 or self.inference_port > 65535:
            raise ValueError(f"Invalid port: {self.inference_port}")
        if self.max_experts < 1:
            raise ValueError(f"max_experts must be >= 1, got {self.max_experts}")
        if self.request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0, got {self.request_timeout}")
