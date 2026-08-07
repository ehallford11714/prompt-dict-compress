"""promptdict — compression, compaction, and recall for LLM prompt memory."""



from .compact import ColdRef, CompactResult, compact_messages, expand_cold_ref

from .compress import compress_hierarchical, compress_text

from .compressor import CompressResult, DictCompressor

from .hierarchical import HierarchicalPageIndexCompress, HierarchicalResult

from .metrics import compression_metrics, estimate_tokens

from .recall import ColdStore, RecallHit, RecallResult, recall

from .scale import (

    BudgetedContextCompressor,

    MillionTokenBudgetCompressor,

    ScaleCompressResult,

    StreamingHierarchicalCompressor,

)

from .suite import PromptMemorySuite



__version__ = "0.3.0"



__all__ = [

    "PromptMemorySuite",

    "DictCompressor",

    "CompressResult",

    "compress_text",

    "compress_hierarchical",

    "HierarchicalPageIndexCompress",

    "HierarchicalResult",

    "BudgetedContextCompressor",

    "StreamingHierarchicalCompressor",

    "MillionTokenBudgetCompressor",

    "ScaleCompressResult",

    "compact_messages",

    "CompactResult",

    "ColdRef",

    "expand_cold_ref",

    "recall",

    "ColdStore",

    "RecallHit",

    "RecallResult",

    "estimate_tokens",

    "compression_metrics",

    "__version__",

]


