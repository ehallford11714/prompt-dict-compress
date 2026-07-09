"""Scale / 100M→1M simulation tests (fast, no huge RAM)."""

from __future__ import annotations

from pathlib import Path

from promptdict.scale import (
    MillionTokenBudgetCompressor,
    generate_repetitive_pages,
    run_scale_demo,
)


def test_streaming_lossless_pages(tmp_path: Path) -> None:
    pages = list(generate_repetitive_pages(n_pages=20, page_lines=30, n_templates=8))
    original = "".join(t for _, t in pages)
    comp = MillionTokenBudgetCompressor(
        input_token_budget=100_000_000,
        output_token_budget=1_000_000,
        page_chars=10_000,
        hot_page_fraction=0.2,
    )
    result = comp.compress_stream(iter(pages), tmp_path / "out")
    decoded = comp.decompress_all(result.out_dir)
    assert decoded == original
    # Sample page
    assert comp.decompress_page(result.out_dir, 0) == pages[0][1]
    assert result.prompt_fits_budget


def test_scale_demo_simulate_fast(tmp_path: Path) -> None:
    result = run_scale_demo(
        target_in=100_000_000,
        target_out=1_000_000,
        simulate=True,
        out_dir=tmp_path / "demo",
        max_materialized_pages=40,
    )
    assert result.input_tokens_est == 100_000_000
    assert result.prompt_tokens_est <= 1_000_000
    assert result.prompt_fits_budget
    assert result.metrics.get("simulation") is True
    # Lossless sample
    comp = MillionTokenBudgetCompressor()
    p0 = comp.decompress_page(result.out_dir, 0)
    assert len(p0) > 0
    # Honesty: prompt-only 100x may or may not hold; factor uses simulated target
    assert result.compression_factor_vs_prompt > 1.0


def test_prompt_pack_under_budget_on_repetitive(tmp_path: Path) -> None:
    pages = generate_repetitive_pages(n_pages=60, page_lines=40, n_templates=12)
    comp = MillionTokenBudgetCompressor(output_token_budget=50_000, hot_page_fraction=0.05)
    result = comp.compress_stream(pages, tmp_path / "budget")
    assert result.prompt_tokens_est <= 50_000
    assert result.prompt_fits_budget
