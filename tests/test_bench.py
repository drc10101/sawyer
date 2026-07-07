"""Tests for sawyer bench command."""

import tempfile
from pathlib import Path

import pytest

from sawyer.bench import (
    compare_results,
    find_llama_bench,
    format_comparison,
    parse_llama_bench_output,
)

# Sample llama-bench markdown output (baseline, CPU, dense model)
BASELINE_MD = (
    "| model | size | params | backend | threads | test | t/s |\n"
    "| ----- | --- | ------ | ------- | ------: | ---: | ---: |\n"
    "| llama 1B Q4_0 | 606.53 MiB | 1.10 B | CPU | 4 | pp512 | 69.02 ± 1.01 |\n"
    "| llama 1B Q4_0 | 606.53 MiB | 1.10 B | CPU | 4 | pp1024 | 67.73 ± 1.09 |\n"
    "| llama 1B Q4_0 | 606.53 MiB | 1.10 B | CPU | 4 | tg128 | 28.17 ± 0.19 |"
)

# Sample llama-bench markdown output (optimized — GPU MoE scenario)
OPTIMIZED_MD = (
    "| model | size | params | backend | threads | test | t/s |\n"
    "| ----- | --- | ------ | ------- | ------: | ---: | ---: |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | pp512 | 1880.50 ± 5.20 |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | pp1024 | 1750.30 ± 3.80 |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | tg128 | 42.15 ± 0.30 |"
)

# Baseline GPU MoE (before optimization)
BASELINE_GPU_MD = (
    "| model | size | params | backend | threads | test | t/s |\n"
    "| ----- | --- | ------ | ------- | ------: | ---: | ---: |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | pp512 | 1143.20 ± 2.50 |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | pp1024 | 1080.40 ± 1.90 |\n"
    "| mixtral 8x7B Q4_0 | 24.00 GiB | 46.70 B | CUDA | 8 | tg128 | 41.80 ± 0.25 |"
)


class TestParseLlamaBenchOutput:
    def test_parse_basic_table(self):
        results = parse_llama_bench_output(BASELINE_MD)
        assert len(results) == 3

        assert results[0].model == "llama 1B Q4_0"
        assert results[0].size == "606.53 MiB"
        assert results[0].params == "1.10 B"
        assert results[0].backend == "CPU"
        assert results[0].threads == 4
        assert results[0].test == "pp512"
        assert results[0].tok_per_sec == 69.02
        assert results[0].std_dev == 1.01

    def test_parse_all_rows(self):
        results = parse_llama_bench_output(BASELINE_MD)
        tests = [r.test for r in results]
        assert tests == ["pp512", "pp1024", "tg128"]

    def test_parse_without_std_dev(self):
        md = (
            "| model | size | params | backend | threads | test | t/s |\n"
            "| ----- | --- | ------ | ------- | ------: | ---: | ---: |\n"
            "| test_model | 100 MiB | 1.00 B | CPU | 4 | pp512 | 100.00 |"
        )
        results = parse_llama_bench_output(md)
        assert len(results) == 1
        assert results[0].std_dev == 0.0

    def test_parse_empty(self):
        results = parse_llama_bench_output("")
        assert len(results) == 0

    def test_parse_header_only(self):
        md = (
            "| model | size | params | backend | threads | test | t/s |\n"
            "| ----- | --- | ------ | ------- | ------: | ---: | ---: |"
        )
        results = parse_llama_bench_output(md)
        assert len(results) == 0

    def test_parse_gpu_results(self):
        results = parse_llama_bench_output(OPTIMIZED_MD)
        assert len(results) == 3
        assert results[0].backend == "CUDA"
        assert results[0].tok_per_sec == 1880.50


class TestCompareResults:
    def test_speedup_calculation(self):
        baseline = parse_llama_bench_output(BASELINE_GPU_MD)
        optimized = parse_llama_bench_output(OPTIMIZED_MD)
        comparison = compare_results(baseline, optimized)

        assert len(comparison) == 3

        # pp512: 1880.5 / 1143.2 = ~1.645x, ~64.5% improvement
        pp512 = next(c for c in comparison if c["test"] == "pp512")
        assert pp512["speedup"] == pytest.approx(1880.5 / 1143.2, rel=0.01)
        assert pp512["improvement_pct"] == pytest.approx((1880.5 / 1143.2 - 1) * 100, rel=0.01)

        # tg128: similar speed (decode not affected by prefill optimization)
        tg128 = next(c for c in comparison if c["test"] == "tg128")
        assert tg128["speedup"] == pytest.approx(42.15 / 41.80, rel=0.01)

    def test_no_overlap_returns_empty(self):
        baseline = parse_llama_bench_output(BASELINE_MD)
        optimized = parse_llama_bench_output(OPTIMIZED_MD)
        # Different models, same test names — should still match on test name
        comparison = compare_results(baseline, optimized)
        assert len(comparison) == 3

    def test_identical_results_show_no_change(self):
        results = parse_llama_bench_output(BASELINE_MD)
        comparison = compare_results(results, results)
        for c in comparison:
            assert c["speedup"] == pytest.approx(1.0)
            assert c["improvement_pct"] == pytest.approx(0.0)


class TestFormatComparison:
    def test_output_contains_key_sections(self):
        baseline = parse_llama_bench_output(BASELINE_GPU_MD)
        optimized = parse_llama_bench_output(OPTIMIZED_MD)
        comparison = compare_results(baseline, optimized)
        output = format_comparison(comparison, "mixtral-8x7b-q4_0.gguf")

        assert "Sawyer Bench Results" in output
        assert "mixtral-8x7b-q4_0.gguf" in output
        assert "Prefill" in output
        assert "pp512" in output
        assert "pp1024" in output
        assert "GGML_SCHED_PREFETCH_EXPERTS" in output
        assert "Average prefill speedup" in output

    def test_output_has_decode_section(self):
        baseline = parse_llama_bench_output(BASELINE_GPU_MD)
        optimized = parse_llama_bench_output(OPTIMIZED_MD)
        comparison = compare_results(baseline, optimized)
        output = format_comparison(comparison, "test.gguf")

        assert "Decode" in output
        assert "tg128" in output

    def test_output_shows_positive_speedup(self):
        baseline = parse_llama_bench_output(BASELINE_GPU_MD)
        optimized = parse_llama_bench_output(OPTIMIZED_MD)
        comparison = compare_results(baseline, optimized)
        output = format_comparison(comparison, "test.gguf")

        # Should show ~64.5% improvement for pp512
        assert "+64" in output or "+65" in output


class TestFindLlamaBench:
    def test_explicit_path_not_found(self):
        with pytest.raises(FileNotFoundError, match="Specified binary not found"):
            find_llama_bench("/nonexistent/path/llama-bench")

    def test_explicit_path_found(self):
        with tempfile.NamedTemporaryFile(suffix="-llama-bench", delete=False) as f:
            path = Path(f.name)
        try:
            result = find_llama_bench(str(path))
            assert result == path
        finally:
            path.unlink()
