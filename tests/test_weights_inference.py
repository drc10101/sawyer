"""Tests for Sawyer weight loader and inference backend."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from sawyer.config import SawyerConfig
from sawyer.node.inference import BackendMode, BackendStatus, InferenceResult, LlamaCppBackend
from sawyer.node.weights import (
    EXPERT_WEIGHT_REPOS,
    WeightFile,
    WeightLoader,
    WeightManifest,
    build_expert_weight_url,
    list_cached_models,
)

# ── Weight Loader Tests ──


class TestWeightLoader:
    """Test WeightLoader with local cache (no network calls)."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = SawyerConfig(cache_dir=self.tmpdir)
        self.loader = WeightLoader(self.config)

    def test_cache_dir_created(self):
        """WeightLoader creates cache directory on init."""
        assert Path(self.config.cache_dir).exists()

    def test_get_model_dir(self):
        """get_model_dir creates and returns model subdirectory."""
        d = self.loader.get_model_dir("mixtral-8x7b")
        assert d.exists()
        assert d.name == "mixtral-8x7b"

    def test_get_expert_urls_known_model(self):
        """get_expert_urls returns URLs for known models."""
        for model_name in EXPERT_WEIGHT_REPOS:
            urls = self.loader.get_expert_urls(model_name)
            assert len(urls) > 0
            assert urls[0].startswith("https://huggingface.co")
            assert urls[0].endswith(".gguf")

    def test_get_expert_urls_unknown_model(self):
        """get_expert_urls raises ValueError for unknown models."""
        with pytest.raises(ValueError, match="Unknown model"):
            self.loader.get_expert_urls("nonexistent-model")

    def test_is_cached_empty(self):
        """is_cached returns False for uncached models."""
        assert not self.loader.is_cached("mixtral-8x7b")

    def test_is_cached_with_file(self):
        """is_cached returns True when GGUF file exists."""
        model_dir = self.loader.get_model_dir("mixtral-8x7b")
        (model_dir / "model.Q4_K_M.gguf").write_bytes(b"\x00" * 100)
        assert self.loader.is_cached("mixtral-8x7b")

    def test_get_cached_path_none(self):
        """get_cached_path returns None for uncached models."""
        assert self.loader.get_cached_path("mixtral-8x7b") is None

    def test_get_cached_path_exists(self):
        """get_cached_path returns path when GGUF file exists."""
        model_dir = self.loader.get_model_dir("mixtral-8x7b")
        gguf = model_dir / "model.Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 100)
        result = self.loader.get_cached_path("mixtral-8x7b")
        assert result is not None
        assert result.suffix == ".gguf"

    def test_clear_cache_specific_model(self):
        """clear_cache removes files for a specific model."""
        model_dir = self.loader.get_model_dir("mixtral-8x7b")
        gguf = model_dir / "model.Q4_K_M.gguf"
        gguf.write_bytes(b"\x00" * 1024)
        freed = self.loader.clear_cache("mixtral-8x7b")
        assert freed == 1024
        assert not model_dir.exists()

    def test_clear_cache_all(self):
        """clear_cache with no model removes all cached files."""
        for model_name in ["mixtral-8x7b", "deepseek-v2-lite"]:
            model_dir = self.loader.get_model_dir(model_name)
            (model_dir / "model.gguf").write_bytes(b"\x00" * 512)
        freed = self.loader.clear_cache()
        assert freed >= 1024

    def test_sha256(self):
        """SHA-256 hash matches known value."""
        import hashlib

        tmpdir = tempfile.mkdtemp()
        tmpfile = Path(tmpdir) / "sha256_test.bin"
        try:
            tmpfile.write_bytes(b"Sawyer test data for SHA-256")
            expected = hashlib.sha256(b"Sawyer test data for SHA-256").hexdigest()
            result = WeightLoader._sha256(tmpfile)
            assert result == expected
        finally:
            tmpfile.unlink(missing_ok=True)
            os.rmdir(tmpdir)


class TestBuildExpertWeightUrl:
    """Test URL building for expert weights."""

    def test_known_model(self):
        url = build_expert_weight_url("mixtral-8x7b", 0)
        assert url is not None
        assert "TheBloke" in url
        assert url.endswith(".gguf")

    def test_unknown_model(self):
        url = build_expert_weight_url("nonexistent", 0)
        assert url is None


class TestListCachedModels:
    """Test listing cached models."""

    def test_empty_dir(self):
        models = list_cached_models(tempfile.mkdtemp())
        assert models == []

    def test_with_cached_models(self):
        tmpdir = tempfile.mkdtemp()
        (Path(tmpdir) / "mixtral-8x7b").mkdir()
        (Path(tmpdir) / "mixtral-8x7b" / "model.gguf").write_bytes(b"\x00")
        (Path(tmpdir) / "deepseek-v2-lite").mkdir()
        # No GGUF file — should not appear
        models = list_cached_models(tmpdir)
        assert "mixtral-8x7b" in models
        assert "deepseek-v2-lite" not in models


class TestWeightManifest:
    """Test WeightManifest dataclass."""

    def test_add_file(self):
        manifest = WeightManifest(model_name="mixtral-8x7b")
        wf = WeightFile(
            model_name="mixtral-8x7b",
            expert_id=0,
            path=Path("/tmp/model.gguf"),
            size_bytes=1024,
            sha256="abc123",
            url="https://example.com/model.gguf",
        )
        manifest.add_file(wf)
        assert len(manifest.files) == 1
        assert manifest.total_size_bytes == 1024


# ── Inference Backend Tests ──


class TestLlamaCppBackend:
    """Test LlamaCppBackend without starting a real server."""

    def test_init_default(self):
        backend = LlamaCppBackend()
        assert backend.mode == BackendMode.SUBPROCESS
        assert "127.0.0.1" in backend.server_url

    def test_init_http_mode(self):
        backend = LlamaCppBackend(
            mode=BackendMode.HTTP,
            server_url="http://192.168.1.100:8080",
        )
        assert backend.mode == BackendMode.HTTP
        assert backend.server_url == "http://192.168.1.100:8080"

    def test_init_custom_port(self):
        config = SawyerConfig(inference_port=9999)
        backend = LlamaCppBackend(config=config)
        assert "9999" in backend.server_url

    def test_status_no_server(self):
        backend = LlamaCppBackend()
        status = backend.get_status()
        assert isinstance(status, BackendStatus)
        assert not status.running

    def test_inference_result_dataclass(self):
        result = InferenceResult(
            text="Hello",
            input_tokens=5,
            output_tokens=1,
            latency_ms=100.0,
            model_name="mixtral-8x7b",
            expert_ids=[0, 2],
        )
        assert result.text == "Hello"
        assert result.input_tokens == 5
        assert result.expert_ids == [0, 2]

    def test_start_server_wrong_mode(self):
        backend = LlamaCppBackend(mode=BackendMode.HTTP)
        with pytest.raises(RuntimeError, match="subprocess mode"):
            backend.start_server("/fake/model.gguf", "test")

    def test_close_no_server(self):
        backend = LlamaCppBackend()
        backend.close()  # Should not raise

    @patch("subprocess.Popen")
    def test_stop_server_no_process(self, mock_popen):
        backend = LlamaCppBackend()
        backend._process = None
        backend.stop_server()  # Should not raise
