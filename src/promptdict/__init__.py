"""promptdict — lossless dictionary-encoding prompt compression."""

from .compressor import DictCompressor, CompressResult
from .hierarchical import HierarchicalPageIndexCompress, HierarchicalResult
from .scale import (
    MillionTokenBudgetCompressor,
    ScaleCompressResult,
    StreamingHierarchicalCompressor,
)
from .metrics import estimate_tokens, compression_metrics

__version__ = "0.2.0"

__all__ = [
    "DictCompressor",
    "CompressResult",
    "HierarchicalPageIndexCompress",
    "HierarchicalResult",
    "MillionTokenBudgetCompressor",
    "StreamingHierarchicalCompressor",
    "ScaleCompressResult",
    "estimate_tokens",
    "compression_metrics",
    "__version__",
]
