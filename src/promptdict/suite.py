"""Suite facade: compress + compact + recall under one budgeted API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from .compact import CompactMode, CompactResult, compact_messages
from .compress import compress_hierarchical, compress_text
from .compressor import CompressResult
from .hierarchical import HierarchicalResult
from .recall import RecallResult, recall
from .scale import BudgetedContextCompressor, ScaleCompressResult

PathLike = Union[str, Path]
Message = Mapping[str, Any]


class PromptMemorySuite:
    """Unified compression / compaction / recall for LLM prompt memory.

    Example::

        suite = PromptMemorySuite(output_budget=1_000_000)
        packed = suite.compress(text)
        compacted = suite.compact(messages)
        restored = suite.recall(page_id="0", store=".scale_demo")
    """

    def __init__(
        self,
        *,
        output_budget: int = 1_000_000,
        input_budget: int = 10_000_000,
        page_chars: int = 16_000,
        compact_budget: Optional[int] = None,
        cold_store: Optional[PathLike] = None,
    ) -> None:
        self.output_budget = output_budget
        self.input_budget = input_budget
        self.page_chars = page_chars
        self.compact_budget = compact_budget or min(8_000, output_budget)
        self.cold_store = Path(cold_store) if cold_store else None
        self._scale = BudgetedContextCompressor(
            input_token_budget=input_budget,
            output_token_budget=output_budget,
            page_chars=page_chars,
        )

    def compress(
        self,
        text: str,
        *,
        mode: str = "flat",
        out_dir: Optional[PathLike] = None,
    ) -> Union[CompressResult, HierarchicalResult, ScaleCompressResult]:
        """Compress text (encode).

        ``mode``:
          - ``flat`` — :func:`compress_text` dictionary pack
          - ``hierarchical`` — in-memory PageIndex
          - ``budgeted`` / ``scale`` — streaming two-tier pack + cold_store
        """
        if mode == "flat":
            return compress_text(text)
        if mode == "hierarchical":
            return compress_hierarchical(text)
        if mode in ("budgeted", "scale", "streaming"):
            dest = Path(out_dir or self.cold_store or ".promptdict_store")
            result = self._scale.compress_stream(text, dest)
            self.cold_store = Path(result.out_dir)
            return result
        raise ValueError(
            f"unknown compress mode {mode!r}; use flat|hierarchical|budgeted"
        )

    def compact(
        self,
        messages: Sequence[Message],
        *,
        budget: Optional[int] = None,
        mode: CompactMode = "auto",
        **kwargs: Any,
    ) -> CompactResult:
        """Compact a message working set under a token budget."""
        return compact_messages(
            messages,
            budget=budget if budget is not None else self.compact_budget,
            mode=mode,
            **kwargs,
        )

    def recall(
        self,
        *,
        page_id: Optional[Union[int, str]] = None,
        page_ids: Optional[Sequence[Union[int, str]]] = None,
        query: Optional[str] = None,
        store: Optional[PathLike] = None,
        top_k: int = 5,
        method: str = "keyword",
    ) -> RecallResult:
        """Recall from cold_store by page id(s) and/or keyword query."""
        path = store or self.cold_store
        if path is None:
            raise ValueError(
                "recall requires store=... or suite cold_store from a prior "
                "compress(mode='budgeted')"
            )
        ids: list[Union[int, str]] = []
        if page_id is not None:
            ids.append(page_id)
        if page_ids:
            ids.extend(page_ids)
        return recall(
            store=path,
            page_ids=ids or None,
            query=query,
            top_k=top_k,
            method=method,
        )
