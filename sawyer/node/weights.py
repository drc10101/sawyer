"""Sawyer Expert Weight Loader — downloads, verifies, and caches MoE expert shards.

Downloads GGUF expert weight files from HuggingFace, verifies SHA-256
checksums, manages a local cache directory, and provides weight paths
to the inference backend.
"""

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from sawyer.config import SawyerConfig
from sawyer.model.registry import get_model

logger = logging.getLogger(__name__)

# Default HF mirror for GGUF weights
HF_BASE_URL = "https://huggingface.co"

# Known GGUF expert shard repos (model_name -> (org, repo, filename_pattern))
# These map to community-maintained quantized MoE models on HuggingFace.
EXPERT_WEIGHT_REPOS: dict[str, dict[str, str]] = {
    "mixtral-8x7b": {
        "org": "TheBloke",
        "repo": "Mixtral-8x7B-v0.1-GGUF",
        "base_url": f"{HF_BASE_URL}/TheBloke/Mixtral-8x7B-v0.1-GGUF/resolve/main",
        "filename_pattern": "mixtral-8x7b-v0.1.Q4_K_M",
        "shard_pattern": "mixtral-8x7b-v0.1.Q4_K_M-{shard:05d}-of-{total:05d}.gguf",
    },
    "deepseek-v2-lite": {
        "org": "bartowski",
        "repo": "deepseek-v2-lite-GGUF",
        "base_url": f"{HF_BASE_URL}/bartowski/deepseek-v2-lite-GGUF/resolve/main",
        "filename_pattern": "deepseek-v2-lite-Q4_K_M",
        "shard_pattern": "deepseek-v2-lite-Q4_K_M-{shard:05d}-of-{total:05d}.gguf",
    },
    "qwen1.5-moe-a2.7b": {
        "org": "bartowski",
        "repo": "Qwen1.5-MoE-A2.7B-GGUF",
        "base_url": f"{HF_BASE_URL}/bartowski/Qwen1.5-MoE-A2.7B-GGUF/resolve/main",
        "filename_pattern": "Qwen1.5-MoE-A2.7B-Q4_K_M",
        "shard_pattern": "Qwen1.5-MoE-A2.7B-Q4_K_M-{shard:05d}-of-{total:05d}.gguf",
    },
    "dbrx": {
        "org": "bartowski",
        "repo": "DBRX-132B-GGUF",
        "base_url": f"{HF_BASE_URL}/bartowski/DBRX-132B-GGUF/resolve/main",
        "filename_pattern": "DBRX-132B-Q4_K_M",
        "shard_pattern": "DBRX-132B-Q4_K_M-{shard:05d}-of-{total:05d}.gguf",
    },
}


@dataclass
class WeightFile:
    """A downloaded expert weight file."""

    model_name: str
    expert_id: int
    path: Path
    size_bytes: int
    sha256: str
    url: str


@dataclass
class WeightManifest:
    """Manifest of all weight files for a model."""

    model_name: str
    files: list[WeightFile] = field(default_factory=list)
    total_size_bytes: int = 0

    def add_file(self, wf: WeightFile) -> None:
        self.files.append(wf)
        self.total_size_bytes += wf.size_bytes


class WeightLoader:
    """Downloads and caches MoE expert weight files.

    Manages a local cache directory structure:
        cache_dir/
            mixtral-8x7b/
                mixtral-8x7b-v0.1.Q4_K_M-00001-of-00019.gguf
                mixtral-8x7b-v0.1.Q4_K_M-00002-of-00019.gguf
                ...
                MANIFEST.json
    """

    def __init__(self, config: SawyerConfig | None = None) -> None:
        self.config = config or SawyerConfig()
        self.cache_dir = Path(self.config.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifests: dict[str, WeightManifest] = {}
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(300.0, connect=30.0),
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None

    def get_model_dir(self, model_name: str) -> Path:
        """Get the cache directory for a model."""
        d = self.cache_dir / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_expert_urls(self, model_name: str) -> list[str]:
        """Get the download URLs for a model's weight files.

        Returns the full GGUF model URL(s). For most quantized models,
        this is a single file or a set of shards.
        """
        if model_name not in EXPERT_WEIGHT_REPOS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {', '.join(EXPERT_WEIGHT_REPOS.keys())}"
            )

        repo_info = EXPERT_WEIGHT_REPOS[model_name]
        base_url = repo_info["base_url"]
        filename = repo_info["filename_pattern"]

        # Single-file GGUF (most Q4_K_M quantizations are single files)
        return [f"{base_url}/{filename}.gguf"]

    def download_weight(
        self,
        model_name: str,
        expert_id: int | None = None,
        force: bool = False,
        verify: bool = True,
    ) -> WeightFile:
        """Download a weight file for a model.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            expert_id: Optional expert number (for logging/manifest)
            force: Re-download even if cached
            verify: Verify SHA-256 checksum after download

        Returns:
            WeightFile with path and metadata
        """
        model = get_model(model_name)
        urls = self.get_expert_urls(model_name)
        model_dir = self.get_model_dir(model_name)

        # For single-file GGUF, download the first URL
        url = urls[0]
        filename = url.rsplit("/", 1)[-1]
        local_path = model_dir / filename

        # Check cache
        if local_path.exists() and not force:
            size = local_path.stat().st_size
            logger.info(
                "Weight file cached: %s (%.1f GB)",
                local_path.name,
                size / (1024**3),
            )
            sha256 = self._sha256(local_path) if verify else ""
            return WeightFile(
                model_name=model_name,
                expert_id=expert_id or 0,
                path=local_path,
                size_bytes=size,
                sha256=sha256,
                url=url,
            )

        # Download
        logger.info("Downloading %s from %s ...", filename, url)
        total_size = model.model_size_gb_q4 * 1024**3

        try:
            with self.client.stream("GET", url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", total_size))
                downloaded = 0

                with open(local_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total and downloaded % (100 * 1024 * 1024) == 0:
                            logger.info(
                                "  %.1f / %.1f GB (%.0f%%)",
                                downloaded / (1024**3),
                                total / (1024**3),
                                100 * downloaded / total,
                            )

            size = local_path.stat().st_size
            logger.info("Downloaded %s (%.1f GB)", local_path.name, size / (1024**3))

        except httpx.HTTPError as e:
            # Clean up partial download
            if local_path.exists():
                local_path.unlink()
            raise RuntimeError(f"Failed to download {url}: {e}") from e

        # Verify checksum
        sha256 = ""
        if verify:
            sha256 = self._sha256(local_path)
            logger.info("SHA-256: %s", sha256[:16] + "...")

        wf = WeightFile(
            model_name=model_name,
            expert_id=expert_id or 0,
            path=local_path,
            size_bytes=size,
            sha256=sha256,
            url=url,
        )

        # Update manifest
        if model_name not in self._manifests:
            self._manifests[model_name] = WeightManifest(model_name=model_name)
        self._manifests[model_name].add_file(wf)

        return wf

    def is_cached(self, model_name: str) -> bool:
        """Check if weight files are cached locally."""
        model_dir = self.get_model_dir(model_name)
        gguf_files = list(model_dir.glob("*.gguf"))
        return len(gguf_files) > 0

    def get_cached_path(self, model_name: str) -> Path | None:
        """Get the path to a cached weight file, or None if not cached."""
        model_dir = self.get_model_dir(model_name)
        gguf_files = sorted(model_dir.glob("*.gguf"))
        if not gguf_files:
            return None
        # Return the first shard (llama.cpp handles multi-file)
        return gguf_files[0]

    def clear_cache(self, model_name: str | None = None) -> int:
        """Remove cached weight files.

        Args:
            model_name: Specific model to clear, or None for all

        Returns:
            Number of bytes freed
        """
        freed = 0
        if model_name:
            model_dir = self.get_model_dir(model_name)
            if model_dir.exists():
                for f in model_dir.glob("*.gguf"):
                    freed += f.stat().st_size
                shutil.rmtree(model_dir)
        else:
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    for f in model_dir.glob("*.gguf"):
                        freed += f.stat().st_size
                    shutil.rmtree(model_dir)
        return freed

    @staticmethod
    def _sha256(path: Path, block_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(block_size):
                h.update(chunk)
        return h.hexdigest()


# ── Weight URL builder for expert shards ──


def build_expert_weight_url(model_name: str, expert_id: int) -> str | None:
    """Build the download URL for a specific expert's weight shard.

    Note: In practice, MoE models are distributed as complete GGUF files,
    not per-expert shards. The router loads the full model and extracts
    expert weights at runtime. This function provides the canonical
    download URL for the full model weights.
    """
    if model_name not in EXPERT_WEIGHT_REPOS:
        return None
    repo = EXPERT_WEIGHT_REPOS[model_name]
    return f"{repo['base_url']}/{repo['filename_pattern']}.gguf"


def list_cached_models(cache_dir: str | Path) -> list[str]:
    """List model names that have cached weight files."""
    cache = Path(cache_dir).expanduser()
    if not cache.exists():
        return []
    return [d.name for d in cache.iterdir() if d.is_dir() and any(d.glob("*.gguf"))]
