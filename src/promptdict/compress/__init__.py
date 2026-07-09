"""Compression pillar: dictionary-encoding + hierarchical PageIndex packing.

Compression *encodes* text into a shorter, API-portable prompt form.
Contrast with :mod:`promptdict.compact` (working-set management over time)
and :mod:`promptdict.recall` (restore from cold_store / page index).
"""

from __future__ import annotations

from ..compressor import CompressResult, DictCompressor
from ..hierarchical import HierarchicalPageIndexCompress, HierarchicalResult
from ..scale import (
    BudgetedContextCompressor,
    MillionTokenBudgetCompressor,
    ScaleCompressResult,
    StreamingHierarchicalCompressor,
    run_scale_demo,
)

__all__ = [
    "DictCompressor",
    "CompressResult",
    "HierarchicalPageIndexCompress",
    "HierarchicalResult",
    "StreamingHierarchicalCompressor",
    "BudgetedContextCompressor",
    "MillionTokenBudgetCompressor",
    "ScaleCompressResult",
    "run_scale_demo",
    "compress_text",
    "compress_hierarchical",
]


def compress_text(
    text: str,
    *,
    min_freq: int = 3,
    max_dict_size: int = 256,
    use_line_patterns: bool = True,
) -> CompressResult:
    """Flat lossless dictionary-encode ``text`` into a packed prompt."""
    return DictCompressor(
        min_freq=min_freq,
        max_dict_size=max_dict_size,
        use_line_patterns=use_line_patterns,
    ).compress(text)


def compress_hierarchical(
    text: str,
    *,
    page_size: int = 4000,
) -> HierarchicalResult:
    """In-memory PageIndex hierarchical compress."""
    return HierarchicalPageIndexCompress(page_size=page_size).compress(
        text, page_size=page_size
    )
