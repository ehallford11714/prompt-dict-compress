"""Hierarchical PageIndex compressor tests."""

from __future__ import annotations

from pathlib import Path

from promptdict.hierarchical import HierarchicalPageIndexCompress

FIXTURES = Path(__file__).parent / "fixtures"


def test_hierarchical_lossless_logs() -> None:
    text = ((FIXTURES / "sample.log").read_text(encoding="utf-8") + "\n") * 30
    h = HierarchicalPageIndexCompress(page_size=800, min_freq=2)
    result = h.compress(text)
    decoded = h.decompress(result)
    assert decoded == text
    assert result.metrics["n_pages"] >= 1


def test_hierarchical_json_roundtrip() -> None:
    text = ((FIXTURES / "sample.jsonl").read_text(encoding="utf-8") + "\n") * 40
    h = HierarchicalPageIndexCompress(page_size=500, min_freq=2)
    result = h.compress(text)
    assert h.decompress(result) == text
