"""PageIndex-style hierarchical dictionary compression (v1)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .compressor import DictCompressor
from .metrics import estimate_tokens
from .mining import iter_pages


@dataclass
class PageRecord:
    page_id: int
    encoded: str
    local_dictionary: dict[str, str]
    original_chars: int
    fingerprint: str  # short summary / hash of content class

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "encoded": self.encoded,
            "local_dictionary": self.local_dictionary,
            "original_chars": self.original_chars,
            "fingerprint": self.fingerprint,
        }


@dataclass
class HierarchicalResult:
    pages: list[PageRecord]
    global_dictionary: dict[str, str]
    page_index: list[dict[str, Any]]
    packed_prompt: str
    encoded_corpus: str
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "0.2.0",
            "kind": "hierarchical_pageindex",
            "global_dictionary": self.global_dictionary,
            "page_index": self.page_index,
            "pages": [p.to_dict() for p in self.pages],
            "encoded_corpus": self.encoded_corpus,
            "packed_prompt": self.packed_prompt,
            "metrics": self.metrics,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class HierarchicalPageIndexCompress:
    """
    Split into pages → per-page local dictionaries → cross-page global dictionary
    of repeated page fingerprints / shared chunks → nested encode.

    This is the in-memory v1 path. For 100M-scale streaming, see
    ``promptdict.scale.MillionTokenBudgetCompressor``.
    """

    def __init__(
        self,
        *,
        page_size: int = 4000,
        min_freq: int = 2,
        max_local_dict: int = 64,
        max_global_dict: int = 128,
    ) -> None:
        self.page_size = page_size
        self.min_freq = min_freq
        self.max_local_dict = max_local_dict
        self.max_global_dict = max_global_dict
        self._local = DictCompressor(
            min_freq=min_freq,
            max_dict_size=max_local_dict,
            use_line_patterns=True,
        )
        self._global = DictCompressor(
            min_freq=max(2, min_freq),
            max_dict_size=max_global_dict,
            use_line_patterns=True,
            use_char_ngrams=True,
        )

    @staticmethod
    def _fingerprint(text: str) -> str:
        # Cheap content-class fingerprint: first non-empty line + length bucket
        line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        head = line[:80]
        return f"L{len(text)}|{head}"

    def compress(self, text: str, page_size: Optional[int] = None) -> HierarchicalResult:
        ps = page_size or self.page_size
        raw_pages = list(iter_pages(text, ps))

        # Pass 1: local encode each page
        pages: list[PageRecord] = []
        local_bodies: list[str] = []
        for pid, page_text in raw_pages:
            local_res = self._local.compress(page_text)
            fp = self._fingerprint(page_text)
            pages.append(
                PageRecord(
                    page_id=pid,
                    encoded=local_res.encoded,
                    local_dictionary=local_res.dictionary,
                    original_chars=len(page_text),
                    fingerprint=fp,
                )
            )
            local_bodies.append(local_res.encoded)

        # Pass 2: global dictionary over concatenated locally-encoded bodies
        joined = "\n\n".join(f"<<<PAGE {p.page_id}>>>\n{p.encoded}" for p in pages)
        global_res = self._global.compress(joined)
        # Re-encode page bodies with global dict only (joined already encoded)
        # For lossless: store global dict + globally-encoded corpus; decode global then local.
        encoded_corpus = global_res.encoded
        global_dictionary = global_res.dictionary

        page_index = [
            {
                "page_id": p.page_id,
                "fingerprint": p.fingerprint,
                "original_chars": p.original_chars,
                "local_dict_size": len(p.local_dictionary),
            }
            for p in pages
        ]

        # Pack prompt: global dict + page index + encoded corpus
        nl = "\\n"
        gdict_lines = [f"  {k} = {v.replace(chr(10), nl)}" for k, v in global_dictionary.items()]
        ldict_blocks = []
        for p in pages:
            if not p.local_dictionary:
                continue
            ldict_blocks.append(f"PAGE_{p.page_id}_DICT:")
            for k, v in p.local_dictionary.items():
                ldict_blocks.append(f"  {k} = {v.replace(chr(10), nl)}")

        packed_parts = [
            "HIERARCHICAL PAGEINDEX DICTIONARY-ENCODED CORPUS",
            "Decode order: expand GLOBAL_DICT, then each PAGE_n_DICT on that page body.",
            "",
            "GLOBAL_DICT:",
            *gdict_lines,
            "",
            "PAGE_INDEX:",
            json.dumps(page_index, ensure_ascii=False),
            "",
            *ldict_blocks,
            "",
            "ENCODED_CORPUS:",
            encoded_corpus,
        ]
        packed_prompt = "\n".join(packed_parts)

        orig_tok = estimate_tokens(text)
        packed_tok = estimate_tokens(packed_prompt)
        metrics = {
            "original_chars": len(text),
            "original_tokens_est": orig_tok,
            "packed_tokens_est": packed_tok,
            "compression_factor": (orig_tok / packed_tok) if packed_tok else 0.0,
            "ratio": 1.0 - (packed_tok / orig_tok) if orig_tok else 0.0,
            "n_pages": len(pages),
            "page_size": ps,
            "global_dict_size": len(global_dictionary),
            "local_dict_total": sum(len(p.local_dictionary) for p in pages),
        }
        return HierarchicalResult(
            pages=pages,
            global_dictionary=global_dictionary,
            page_index=page_index,
            packed_prompt=packed_prompt,
            encoded_corpus=encoded_corpus,
            metrics=metrics,
        )

    def decompress(self, result: HierarchicalResult) -> str:
        """Lossless round-trip from per-page local dictionaries (authoritative).

        The globally-encoded corpus in ``packed_prompt`` is for LLM ICL size
        reduction; mechanical lossless decode uses ``pages[].encoded`` + local dicts
        so page boundaries cannot be corrupted by global substitutions.
        """
        out_pages: list[str] = []
        for p in sorted(result.pages, key=lambda x: x.page_id):
            out_pages.append(self._local.decode(p.encoded, p.local_dictionary))
        return "".join(out_pages)

    def decompress_dict(self, data: dict[str, Any]) -> str:
        pages = [
            PageRecord(
                page_id=int(p["page_id"]),
                encoded=str(p["encoded"]),
                local_dictionary={str(k): str(v) for k, v in p.get("local_dictionary", {}).items()},
                original_chars=int(p.get("original_chars", 0)),
                fingerprint=str(p.get("fingerprint", "")),
            )
            for p in data.get("pages", [])
        ]
        result = HierarchicalResult(
            pages=pages,
            global_dictionary={str(k): str(v) for k, v in data.get("global_dictionary", {}).items()},
            page_index=list(data.get("page_index", [])),
            packed_prompt=str(data.get("packed_prompt", "")),
            encoded_corpus=str(data.get("encoded_corpus", "")),
            metrics=dict(data.get("metrics", {})),
        )
        return self.decompress(result)
