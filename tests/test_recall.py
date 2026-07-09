"""Cold-store recall tests."""

from __future__ import annotations

from pathlib import Path

from promptdict.recall import recall
from promptdict.scale import BudgetedContextCompressor, generate_repetitive_pages


def test_recall_by_page_id_after_compress_stream(tmp_path: Path) -> None:
    marker = "UNIQUE_TOKEN_ZEBRA_42"
    pages = list(generate_repetitive_pages(n_pages=12, page_lines=25, n_templates=6))
    # Inject a known token into page 3
    pid, text = pages[3]
    pages[3] = (pid, text + f"\nmarker={marker}\n")

    comp = BudgetedContextCompressor(
        input_token_budget=10_000_000,
        output_token_budget=50_000,
        page_chars=8_000,
        hot_page_fraction=0.2,
    )
    result = comp.compress_stream(iter(pages), tmp_path / "store")
    out = Path(result.out_dir)

    by_id = recall(store=out, page_ids=[3])
    assert by_id.hits
    assert marker in by_id.hits[0].text
    assert marker in by_id.packed_fragment


def test_keyword_recall_finds_known_token(tmp_path: Path) -> None:
    marker = "UNIQUE_TOKEN_ZEBRA_42"
    pages = list(generate_repetitive_pages(n_pages=10, page_lines=20, n_templates=5))
    pid, text = pages[2]
    pages[2] = (pid, text + f"\nfound {marker} here\n")

    comp = BudgetedContextCompressor(
        input_token_budget=10_000_000,
        output_token_budget=40_000,
        hot_page_fraction=0.15,
    )
    result = comp.compress_stream(iter(pages), tmp_path / "kw")
    kw = recall(store=result.out_dir, query=marker, top_k=3)
    assert kw.hits
    assert any(marker in h.text for h in kw.hits)
