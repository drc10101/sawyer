"""Sawyer Bench — before/after benchmarking for MoE prefill optimizations.

Runs llama-bench twice:
  1. Baseline — no optimization env vars
  2. Optimized — GGML_SCHED_PREFETCH_EXPERTS=1 (pinned mmap is always-on with CUDA)

Parses the markdown output and prints a comparison table focused on prefill speed.

Usage:
    sawyer bench -m /path/to/model.gguf
    sawyer bench -m /path/to/model.gguf --binary /path/to/llama-bench
    sawyer bench -m /path/to/model.gguf -p 512,1024,2048 -ngl 99
"""

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BINARY_NAME = "llama-bench"

# Default benchmark prompts for prefill testing
DEFAULT_PROMPT_SIZES = "512,1024,2048"
DEFAULT_REPETITIONS = 3
DEFAULT_THREADS = -1  # auto
DEFAULT_NGL = 99  # offload all layers to GPU


@dataclass
class BenchResult:
    """Single benchmark row from llama-bench output."""

    model: str
    size: str
    params: str
    backend: str
    threads: int
    test: str  # e.g. "pp512", "tg128"
    tok_per_sec: float
    std_dev: float


def find_llama_bench(binary_path: str | None = None) -> Path:
    """Find the llama-bench binary.

    Search order:
    1. Explicit path provided by user
    2. Next to the current Python executable (same venv/bin dir)
    3. On PATH
    4. In ~/.sawyer/bin/
    """
    if binary_path:
        p = Path(binary_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified binary not found: {binary_path}")

    # Check next to python executable (venv/bin or Scripts dir)
    python_dir = Path(sys.executable).parent
    candidate = python_dir / BINARY_NAME
    if candidate.is_file():
        return candidate

    # Check PATH
    found = shutil.which(BINARY_NAME)
    if found:
        return Path(found)

    # Check ~/.sawyer/bin/
    sawyer_bin = Path.home() / ".sawyer" / "bin" / BINARY_NAME
    if sawyer_bin.is_file():
        return sawyer_bin

    raise FileNotFoundError(
        f"Cannot find {BINARY_NAME}. Install Sawyer's optimized llama.cpp "
        f"binary or specify path with --binary.\n"
        f"Searched: {python_dir}, PATH, ~/.sawyer/bin/"
    )


def parse_llama_bench_output(output: str) -> list[BenchResult]:
    """Parse markdown table output from llama-bench.

    Expected format:
    | model | size | params | backend | threads | test | t/s |
    | ... | ... | ... | ... | ... | ... | ... |
    """
    results = []
    # Match rows like: | llama 1B Q4_0 | 606.53 MiB | 1.10 B | CPU | 4 | pp512 | 69.02 ± 1.01 |
    pattern = re.compile(
        r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*\|\s*(\S+)\s*\|"  # noqa: E501
        r"\s*([\d.]+)\s*(?:±\s*([\d.]+))?\s*\|"
    )

    for line in output.strip().split("\n"):
        m = pattern.match(line.strip())
        if m:
            results.append(
                BenchResult(
                    model=m.group(1).strip(),
                    size=m.group(2).strip(),
                    params=m.group(3).strip(),
                    backend=m.group(4).strip(),
                    threads=int(m.group(5)),
                    test=m.group(6).strip(),
                    tok_per_sec=float(m.group(7)),
                    std_dev=float(m.group(8)) if m.group(8) else 0.0,
                )
            )
    return results


def run_benchmark(
    binary: Path,
    model: Path,
    prompt_sizes: str = DEFAULT_PROMPT_SIZES,
    repetitions: int = DEFAULT_REPETITIONS,
    threads: int = DEFAULT_THREADS,
    ngl: int = DEFAULT_NGL,
    extra_env: dict[str, str] | None = None,
) -> list[BenchResult]:
    """Run llama-bench with given parameters and return parsed results."""
    cmd = [
        str(binary),
        "-m",
        str(model),
        "-p",
        prompt_sizes,
        "-r",
        str(repetitions),
        "-o",
        "md",
    ]
    if threads > 0:
        cmd.extend(["-t", str(threads)])
    if ngl >= 0:
        cmd.extend(["-ngl", str(ngl)])

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,  # 10 min max per benchmark run
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"llama-bench failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )

    return parse_llama_bench_output(result.stdout)


def compare_results(
    baseline: list[BenchResult],
    optimized: list[BenchResult],
) -> list[dict]:
    """Compare baseline vs optimized results, computing speedup ratios.

    Returns list of dicts with test, baseline_t/s, optimized_t/s,
    speedup, improvement%.
    """
    # Index by test name for matching
    baseline_by_test = {r.test: r for r in baseline}
    optimized_by_test = {r.test: r for r in optimized}

    all_tests = sorted(set(baseline_by_test.keys()) | set(optimized_by_test.keys()))

    comparison = []
    for test in all_tests:
        b = baseline_by_test.get(test)
        o = optimized_by_test.get(test)
        if not b or not o:
            continue

        speedup = o.tok_per_sec / b.tok_per_sec if b.tok_per_sec > 0 else 0
        improvement_pct = (speedup - 1.0) * 100

        comparison.append(
            {
                "test": test,
                "baseline_tps": b.tok_per_sec,
                "baseline_std": b.std_dev,
                "optimized_tps": o.tok_per_sec,
                "optimized_std": o.std_dev,
                "speedup": speedup,
                "improvement_pct": improvement_pct,
            }
        )
    return comparison


def format_comparison(comparison: list[dict], model_name: str) -> str:
    """Format comparison results as a readable table."""
    lines = []
    lines.append("")
    lines.append("=" * 72)
    lines.append(f"  Sawyer Bench Results — {model_name}")
    lines.append("=" * 72)
    lines.append("")

    # Prefill rows first, then decode
    prefill = [c for c in comparison if c["test"].startswith("pp")]
    decode = [c for c in comparison if c["test"].startswith("tg")]

    if prefill:
        lines.append("  Prefill (prompt processing):")
        lines.append(
            f"  {'Test':<10} {'Baseline':>12} {'Optimized':>12} " f"{'Speedup':>10} {'Change':>10}"
        )
        lines.append("  " + "-" * 56)
        for c in prefill:
            change = f"{c['improvement_pct']:+.1f}%"
            lines.append(
                f"  {c['test']:<10} "
                f"{c['baseline_tps']:>8.1f} t/s "
                f"{c['optimized_tps']:>8.1f} t/s "
                f"{c['speedup']:>8.2f}x  "
                f"{change:>10}"
            )
        lines.append("")

    if decode:
        lines.append("  Decode (token generation):")
        lines.append(
            f"  {'Test':<10} {'Baseline':>12} {'Optimized':>12} " f"{'Speedup':>10} {'Change':>10}"
        )
        lines.append("  " + "-" * 56)
        for c in decode:
            change = f"{c['improvement_pct']:+.1f}%"
            lines.append(
                f"  {c['test']:<10} "
                f"{c['baseline_tps']:>8.1f} t/s "
                f"{c['optimized_tps']:>8.1f} t/s "
                f"{c['speedup']:>8.2f}x  "
                f"{change:>10}"
            )
        lines.append("")

    # Summary
    if prefill:
        avg_speedup = sum(c["speedup"] for c in prefill) / len(prefill)
        avg_improvement = sum(c["improvement_pct"] for c in prefill) / len(prefill)
        lines.append(
            f"  Average prefill speedup: {avg_speedup:.2f}x " f"({avg_improvement:+.1f}%)"
        )

    if decode:
        avg_speedup_d = sum(c["speedup"] for c in decode) / len(decode)
        avg_improvement_d = sum(c["improvement_pct"] for c in decode) / len(decode)
        lines.append(
            f"  Average decode speedup:  {avg_speedup_d:.2f}x " f"({avg_improvement_d:+.1f}%)"
        )

    lines.append("")
    lines.append("  Optimization env vars:")
    lines.append("    GGML_SCHED_PREFETCH_EXPERTS=1  " "(pipeline expert uploads with compute)")
    lines.append("    Pinned mmap: automatic with CUDA backend")
    lines.append("")

    return "\n".join(lines)


def cmd_bench(args) -> int:
    """Run before/after MoE benchmark comparison."""
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1

    # Find binary
    try:
        binary = find_llama_bench(args.binary)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Sawyer Bench — MoE prefill optimization comparison")
    print(f"  Model:   {model_path}")
    print(f"  Binary:  {binary}")
    print(f"  Prompts: {args.prompt_sizes}")
    print(f"  GPU offload layers: {args.ngl}")
    print(f"  Repetitions: {args.repetitions}")
    print()

    # Run baseline (no optimization env vars)
    print("Running baseline (no optimizations)...")
    try:
        baseline = run_benchmark(
            binary=binary,
            model=model_path,
            prompt_sizes=args.prompt_sizes,
            repetitions=args.repetitions,
            threads=args.threads,
            ngl=args.ngl,
        )
    except RuntimeError as e:
        print(f"Baseline benchmark failed: {e}", file=sys.stderr)
        return 1

    if not baseline:
        print(
            "Error: baseline benchmark produced no results.",
            file=sys.stderr,
        )
        return 1

    print(f"  Got {len(baseline)} results: {[r.test for r in baseline]}")

    # Run optimized (with env vars)
    print("Running optimized (GGML_SCHED_PREFETCH_EXPERTS=1)...")
    optimized_env = {"GGML_SCHED_PREFETCH_EXPERTS": "1"}
    try:
        optimized = run_benchmark(
            binary=binary,
            model=model_path,
            prompt_sizes=args.prompt_sizes,
            repetitions=args.repetitions,
            threads=args.threads,
            ngl=args.ngl,
            extra_env=optimized_env,
        )
    except RuntimeError as e:
        print(f"Optimized benchmark failed: {e}", file=sys.stderr)
        return 1

    if not optimized:
        print(
            "Error: optimized benchmark produced no results.",
            file=sys.stderr,
        )
        return 1

    print(f"  Got {len(optimized)} results: {[r.test for r in optimized]}")

    # Compare
    comparison = compare_results(baseline, optimized)
    if not comparison:
        print(
            "Error: no matching test results to compare.",
            file=sys.stderr,
        )
        return 1

    output = format_comparison(comparison, model_path.name)
    print(output)

    # Optionally save JSON
    if args.json:
        json_path = Path(args.json)
        json_path.write_text(
            json.dumps(
                {
                    "model": str(model_path),
                    "binary": str(binary),
                    "baseline": [
                        {
                            "test": r.test,
                            "tok_per_sec": r.tok_per_sec,
                            "std_dev": r.std_dev,
                        }
                        for r in baseline
                    ],
                    "optimized": [
                        {
                            "test": r.test,
                            "tok_per_sec": r.tok_per_sec,
                            "std_dev": r.std_dev,
                        }
                        for r in optimized
                    ],
                    "comparison": comparison,
                },
                indent=2,
            )
        )
        print(f"Results saved to {json_path}")

    return 0
