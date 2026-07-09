"""Recall pillar: restore text from cold_store / page index.

Retrieve by ``page_id`` or simple keyword query. Optional embedding recall
is stubbed for future backends (no vector DB required for the MVP).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .compressor import DictCompressor
from .metrics import estimate_tokens

PathLike = Union[str, Path]


@dataclass
class RecallHit:
    page_id: Union[int, str]
    score: float
    text: str
    fingerprint: str = ""
    template_id: str = ""
    source: str = "cold_store"

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "score": self.score,
            "text": self.text,
            "fingerprint": self.fingerprint,
            "template_id": self.template_id,
            "source": self.source,
            "tokens_est": estimate_tokens(self.text),
        }


@dataclass
class RecallResult:
    hits: list[RecallHit]
    packed_fragment: str
    query: str = ""
    store_path: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": [h.to_dict() for h in self.hits],
            "packed_fragment": self.packed_fragment,
            "query": self.query,
            "store_path": self.store_path,
            "metrics": self.metrics,
            "version": "0.3.0",
        }

    @property
    def text(self) -> str:
        """Concatenated hit texts (page_id order when applicable)."""
        return "".join(h.text for h in self.hits)


class ColdStore:
    """Load and query a scale ``cold_store.jsonl`` (+ optional page_index)."""

    def __init__(self, store: PathLike) -> None:
        self.path = Path(store)
        if self.path.is_dir():
            self.out_dir = self.path
            self.cold_path = self.path / "cold_store.jsonl"
            self.index_path = self.path / "page_index.json"
        else:
            self.cold_path = self.path
            self.out_dir = self.path.parent
            self.index_path = self.out_dir / "page_index.json"
        self._rows: Optional[list[dict[str, Any]]] = None
        self._index: Optional[dict[str, Any]] = None
        self._decoder = DictCompressor()

    def _load_rows(self) -> list[dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        if not self.cold_path.exists():
            raise FileNotFoundError(f"cold_store not found: {self.cold_path}")
        rows: list[dict[str, Any]] = []
        with self.cold_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        self._rows = rows
        return rows

    def _load_index(self) -> dict[str, Any]:
        if self._index is not None:
            return self._index
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self._index = {}
        return self._index

    def decode_row(self, row: dict[str, Any]) -> str:
        encoded = str(row.get("encoded", ""))
        dictionary = {
            str(k): str(v) for k, v in dict(row.get("local_dictionary", {})).items()
        }
        return self._decoder.decode(encoded, dictionary)

    def get_page(self, page_id: Union[int, str]) -> str:
        pid = int(page_id) if str(page_id).lstrip("-").isdigit() else page_id
        for row in self._load_rows():
            rid = row.get("page_id")
            if rid == pid or str(rid) == str(page_id):
                return self.decode_row(row)
        raise KeyError(f"page_id {page_id!r} not found in {self.cold_path}")

    def iter_pages(self) -> Sequence[tuple[Any, str, dict[str, Any]]]:
        out: list[tuple[Any, str, dict[str, Any]]] = []
        for row in self._load_rows():
            text = self.decode_row(row)
            out.append((row.get("page_id"), text, row))
        return out

    def keyword_search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RecallHit]:
        """Simple case-insensitive keyword / token overlap recall."""
        tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) >= 2]
        if not tokens:
            return []
        hits: list[RecallHit] = []
        for page_id, text, row in self.iter_pages():
            lower = text.lower()
            score = 0.0
            for t in tokens:
                score += lower.count(t)
            # Boost if fingerprint / template mentions query
            fp = str(row.get("fingerprint", ""))
            tid = str(row.get("template_id", ""))
            blob = f"{fp} {tid}".lower()
            for t in tokens:
                if t in blob:
                    score += 2.0
            if score > 0:
                hits.append(
                    RecallHit(
                        page_id=page_id if page_id is not None else -1,
                        score=score,
                        text=text,
                        fingerprint=fp,
                        template_id=tid,
                        source=str(self.cold_path),
                    )
                )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


def pack_fragment(hits: Sequence[RecallHit]) -> str:
    """Pack recalled pages into a prompt fragment."""
    if not hits:
        return ""
    parts = [
        "RECALLED_PAGES (lossless from cold_store):",
        f"n_hits={len(hits)}",
        "",
    ]
    for h in hits:
        parts.append(f"<<<PAGE {h.page_id} score={h.score:.1f}>>>")
        parts.append(h.text.rstrip())
        parts.append("")
    return "\n".join(parts)


def recall(
    *,
    store: PathLike,
    page_ids: Optional[Sequence[Union[int, str]]] = None,
    query: Optional[str] = None,
    top_k: int = 5,
    method: str = "keyword",
) -> RecallResult:
    """Recall text from a cold_store by page_id(s) and/or query.

    Parameters
    ----------
    store:
        Path to ``cold_store.jsonl`` or a scale ``out_dir`` containing it.
    page_ids:
        Exact page ids to restore (lossless).
    query:
        Keyword query (MVP). Ignored when ``method='embedding'`` (stub).
    top_k:
        Max keyword hits when ``query`` is set.
    method:
        ``keyword`` (default) or ``embedding`` (raises NotImplementedError stub).
    """
    if method == "embedding":
        return embedding_recall_stub(query=query or "", store=store, top_k=top_k)

    cs = ColdStore(store)
    hits: list[RecallHit] = []

    if page_ids:
        for pid in page_ids:
            text = cs.get_page(pid)
            hits.append(
                RecallHit(
                    page_id=pid,
                    score=1.0,
                    text=text,
                    source=str(cs.cold_path),
                )
            )

    if query:
        kw_hits = cs.keyword_search(query, top_k=top_k)
        seen = {str(h.page_id) for h in hits}
        for h in kw_hits:
            if str(h.page_id) not in seen:
                hits.append(h)
                seen.add(str(h.page_id))

    if not page_ids and not query:
        raise ValueError("recall() requires page_ids and/or query")

    fragment = pack_fragment(hits)
    return RecallResult(
        hits=hits,
        packed_fragment=fragment,
        query=query or "",
        store_path=str(cs.cold_path.resolve()),
        metrics={
            "n_hits": len(hits),
            "method": method,
            "tokens_est": estimate_tokens(fragment),
        },
    )


def embedding_recall_stub(
    *,
    query: str,
    store: PathLike,
    top_k: int = 5,
) -> RecallResult:
    """Optional embedding recall — not implemented in MVP (no vector DB).

    Raise so callers know to use ``method='keyword'`` or plug in their own backend.
    """
    raise NotImplementedError(
        "embedding recall is a stub in promptdict 0.3; use method='keyword' "
        f"(query={query!r}, store={store!r}, top_k={top_k}) or provide an "
        "external vector index over ColdStore.iter_pages()."
    )
