"""Sawyer Download — fetch Sawyer Fast Llama binaries from GitHub Releases.

Auto-downloads the correct platform binary for `sawyer bench` when no
--binary path is specified. Stores in ~/.sawyer/bin/.

GitHub Release: drc10101/llama.cpp tag sawyer-fast-llama-v0.6.0
Assets: sawyer-fast-llama-linux-x64, sawyer-fast-llama-windows-x64.exe

Usage:
    from sawyer.download import ensure_llama_bench
    binary = ensure_llama_bench()  # downloads if needed, returns Path
"""

import json
import platform
import sys
import urllib.request
from pathlib import Path

RELEASES_API = "https://api.github.com/repos/drc10101/llama.cpp/releases"
BINARY_DIR = Path.home() / ".sawyer" / "bin"
RELEASE_TAG = "sawyer-fast-llama-v0.6.0"


def _get_platform_name() -> str:
    """Return the platform-specific asset name fragment."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine in ("x86_64", "amd64"):
        return "linux-x64"
    elif system == "windows" and machine in ("x86_64", "amd64"):
        return "windows-x64"
    elif system == "darwin" and machine in ("x86_64", "amd64"):
        return "macos-x64"
    elif system == "darwin" and machine == "arm64":
        return "macos-arm64"
    else:
        return f"{system}-{machine}"


def _get_binary_name() -> str:
    """Return the expected binary filename for the current platform."""
    plat = _get_platform_name()
    if plat.startswith("windows"):
        return f"sawyer-fast-llama-{plat}.exe"
    return f"sawyer-fast-llama-{plat}"


def _find_cached_binary() -> Path | None:
    """Check if binary already exists in ~/.sawyer/bin/."""
    binary_name = _get_binary_name()
    candidate = BINARY_DIR / binary_name
    if candidate.is_file():
        return candidate

    # Also check for llama-bench (legacy name) in the same dirs
    # that find_llama_bench() in bench.py searches
    for name in [binary_name, "llama-bench"]:
        for search_dir in [
            Path(sys.executable).parent,
            BINARY_DIR,
        ]:
            p = search_dir / name
            if p.is_file():
                return p

    return None


def _fetch_release_info(tag: str = RELEASE_TAG) -> dict:
    """Fetch release info from GitHub API."""
    url = f"{RELEASES_API}/tags/{tag}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "sawyer-core/0.6.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _download_asset(url: str, dest: Path) -> None:
    """Download a file from URL to dest with progress output."""
    print(f"  Downloading {url.split('/')[-1]}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "sawyer-core/0.6.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        BINARY_DIR.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            downloaded = 0
            chunk_size = 65536
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100 // total
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    print(
                        f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                        end="",
                        flush=True,
                    )
            print()  # newline after progress

    # Make executable on Unix
    if not dest.name.endswith(".exe"):
        dest.chmod(dest.stat().st_mode | 0o755)


def ensure_llama_bench(force: bool = False) -> Path:
    """Download Sawyer Fast Llama binary if not cached.

    Returns the path to the binary. Downloads from GitHub Releases
    if not found locally.

    Args:
        force: Re-download even if cached binary exists.

    Returns:
        Path to the llama-bench binary.

    Raises:
        RuntimeError: If download fails or platform not supported.
    """
    if not force:
        cached = _find_cached_binary()
        if cached:
            return cached

    binary_name = _get_binary_name()
    plat = _get_platform_name()

    print(f"Sawyer Fast Llama — downloading for {plat}...")
    print(f"  Release: {RELEASE_TAG}")

    try:
        release = _fetch_release_info()
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch release info from GitHub: {e}\n"
            f"Check your internet connection or download manually from:\n"
            f"  https://github.com/drc10101/llama.cpp/releases/tag/{RELEASE_TAG}"
        ) from e

    # Find the matching asset
    assets = release.get("assets", [])
    matching = [a for a in assets if a["name"] == binary_name]

    if not matching:
        available = [a["name"] for a in assets]
        raise RuntimeError(
            f"No binary found for platform '{plat}' "
            f"(expected: {binary_name}).\n"
            f"Available assets: {available}\n"
            f"Download manually from:\n"
            f"  https://github.com/drc10101/llama.cpp/releases/tag/{RELEASE_TAG}"
        )

    asset = matching[0]
    dest = BINARY_DIR / binary_name
    _download_asset(asset["browser_download_url"], dest)

    # Also download llama-bench symlink/alias for backwards compat
    # (On Linux, the same binary serves as both llama-bench and llama-server
    #  via argv[0] — but we'll just link it)
    bench_dest = BINARY_DIR / "llama-bench"
    if not dest.name.endswith(".exe") and not bench_dest.exists():
        try:
            bench_dest.symlink_to(dest)
        except OSError:
            # Symlink might fail on some systems; just copy
            import shutil

            shutil.copy2(dest, bench_dest)

    return dest


def get_binary_info(binary: Path) -> dict:
    """Get version info from a Sawyer Fast Llama binary.

    Tries --version first (llama-cli), falls back to running the binary
    with no args and parsing stderr for version info (llama-bench).

    Returns dict with 'version' and 'commit' keys, or empty dict on failure.
    """
    import subprocess
    import re

    # Try --version first (works for llama-cli)
    for args in [["--version"], ["-v"]]:
        try:
            result = subprocess.run(
                [str(binary)] + args,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip() or result.stderr.strip()
            if output and "version" in output.lower():
                version_match = re.search(r"version:\s*(\d+)", output)
                commit_match = re.search(r"\(([a-f0-9]{7,})\)", output)
                return {
                    "version": version_match.group(1) if version_match else "unknown",
                    "commit": commit_match.group(1) if commit_match else "unknown",
                    "raw": output,
                }
        except Exception:
            pass

    return {}
