"""Tests for Sawyer Fast Llama download module."""

import json
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sawyer.download import (
    BINARY_DIR,
    RELEASE_TAG,
    _get_binary_name,
    _get_platform_name,
    ensure_llama_bench,
    get_binary_info,
)


class TestPlatformName:
    """Platform name detection."""

    def test_linux_x64(self):
        with patch("sawyer.download.platform") as mock_plat:
            mock_plat.system.return_value = "Linux"
            mock_plat.machine.return_value = "x86_64"
            assert _get_platform_name() == "linux-x64"

    def test_windows_x64(self):
        with patch("sawyer.download.platform") as mock_plat:
            mock_plat.system.return_value = "Windows"
            mock_plat.machine.return_value = "AMD64"
            assert _get_platform_name() == "windows-x64"

    def test_macos_arm64(self):
        with patch("sawyer.download.platform") as mock_plat:
            mock_plat.system.return_value = "Darwin"
            mock_plat.machine.return_value = "arm64"
            assert _get_platform_name() == "macos-arm64"


class TestBinaryName:
    """Binary filename generation."""

    def test_linux_no_extension(self):
        with patch("sawyer.download._get_platform_name", return_value="linux-x64"):
            assert _get_binary_name() == "sawyer-fast-llama-linux-x64"

    def test_windows_has_exe(self):
        with patch("sawyer.download._get_platform_name", return_value="windows-x64"):
            assert _get_binary_name() == "sawyer-fast-llama-windows-x64.exe"


class TestEnsureLlamaBench:
    """Binary download and caching."""

    def test_cached_binary_returned(self, tmp_path):
        """If binary exists in cache, return it without downloading."""
        binary_name = _get_binary_name()
        cached = tmp_path / binary_name
        cached.write_text("# fake binary", encoding="utf-8")

        with patch("sawyer.download.BINARY_DIR", tmp_path):
            with patch("sawyer.download._find_cached_binary", return_value=cached):
                result = ensure_llama_bench(force=False)
                assert result == cached

    def test_force_redownload(self, tmp_path):
        """With force=True, re-download even if cached."""
        binary_name = _get_binary_name()
        cached = tmp_path / binary_name
        cached.write_text("# old binary", encoding="utf-8")

        mock_release = {
            "assets": [
                {
                    "name": binary_name,
                    "browser_download_url": f"https://example.com/{binary_name}",
                    "size": 100,
                }
            ]
        }

        with patch("sawyer.download.BINARY_DIR", tmp_path):
            with patch(
                "sawyer.download._fetch_release_info", return_value=mock_release
            ):
                with patch(
                    "sawyer.download._download_asset"
                ) as mock_dl:
                    mock_dl.side_effect = lambda url, dest: dest.write_text(
                        "# new binary", encoding="utf-8"
                    )
                    # Force should skip cache check
                    with patch(
                        "sawyer.download._find_cached_binary", return_value=None
                    ):
                        result = ensure_llama_bench(force=True)
                        mock_dl.assert_called_once()

    def test_no_matching_platform_raises(self):
        """If no binary for the platform, raise RuntimeError."""
        mock_release = {"assets": [{"name": "sawyer-fast-llama-other-arch"}]}

        with patch(
            "sawyer.download._fetch_release_info", return_value=mock_release
        ):
            with patch("sawyer.download._find_cached_binary", return_value=None):
                with pytest.raises(
                    RuntimeError, match="No binary found for platform"
                ):
                    ensure_llama_bench(force=True)

    def test_github_api_failure_raises(self):
        """If GitHub API unreachable, raise RuntimeError."""
        mock_release = {"assets": [{"name": "sawyer-fast-llama-other-arch"}]}

        with patch(
            "sawyer.download._fetch_release_info", side_effect=Exception("network error")
        ):
            with patch("sawyer.download._find_cached_binary", return_value=None):
                with pytest.raises(
                    RuntimeError, match="Failed to fetch release info"
                ):
                    ensure_llama_bench(force=True)

    def test_github_failure_in_find_raises_file_not_found(self):
        """If auto-download fails in find_llama_bench, raise FileNotFoundError."""
        with patch(
            "sawyer.download._fetch_release_info", side_effect=Exception("network error")
        ):
            with patch("sawyer.download._find_cached_binary", return_value=None):
                with pytest.raises(
                    FileNotFoundError, match="auto-download failed"
                ):
                    from sawyer.bench import find_llama_bench
                    find_llama_bench()


class TestGetBinaryInfo:
    """Binary version info extraction."""

    def test_parse_version_output(self):
        """Parse 'version: 50 (ba09fc5)' format."""
        mock_run = MagicMock(
            return_value=MagicMock(
                stdout="version: 50 (ba09fc5)\nbuilt with GNU 15.2.0 for Linux x86_64"
            )
        )
        with patch("subprocess.run", mock_run):
            result = get_binary_info(Path("/fake/binary"))
            assert result["version"] == "50"
            assert result["commit"] == "ba09fc5"

    def test_empty_output(self):
        """Gracefully handle empty version output — returns empty dict."""
        mock_run = MagicMock(return_value=MagicMock(stdout="", stderr=""))
        with patch("subprocess.run", mock_run):
            result = get_binary_info(Path("/fake/binary"))
            assert result == {}

    def test_subprocess_error(self):
        """Gracefully handle subprocess errors."""
        with patch("subprocess.run", side_effect=OSError):
            result = get_binary_info(Path("/fake/binary"))
            assert result == {}