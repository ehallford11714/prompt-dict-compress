"""Core lossless dictionary-encoding compressor."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .metrics import Metrics, compression_metrics, estimate_tokens
from .mining import (
    merge_candidates,
    mine_char_ngrams,
    mine_line_patterns,
    mine_ngram_patterns,
)


META_PREFIX = "⟦"
META_SUFFIX = "⟧"
# Fallback ASCII form if source already contains ⟦ ⟧
ALT_PREFIX = "<<PD"
ALT_SUFFIX = ">>"


@dataclass
class CompressResult:
    original: str
    encoded: str
    dictionary: dict[str, str]
    system_dictionary: str
    packed_prompt: str
    metrics: Metrics
    meta_style: str = "unicode"

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded": self.encoded,
            "dictionary": self.dictionary,
            "system_dictionary": self.system_dictionary,
            "packed_prompt": self.packed_prompt,
            "metrics": self.metrics.to_dict(),
            "meta_style": self.meta_style,
            "version": "0.2.0",
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class DictCompressor:
    """Mine patterns → build dictionary → longest-first encode / decode."""

    def __init__(
        self,
        *,
        min_freq: int = 3,
        max_ngram: int = 10,
        max_dict_size: int = 256,
        use_line_patterns: bool = True,
        use_char_ngrams: bool = False,
        meta_style: str = "auto",
    ) -> None:
        self.min_freq = min_freq
        self.max_ngram = max_ngram
        self.max_dict_size = max_dict_size
        self.use_line_patterns = use_line_patterns
        self.use_char_ngrams = use_char_ngrams
        self.meta_style = meta_style

    def _choose_style(self, text: str) -> tuple[str, str, str]:
        style = self.meta_style
        if style == "auto":
            if META_PREFIX in text or META_SUFFIX in text:
                style = "ascii"
            else:
                style = "unicode"
        if style == "unicode":
            return "unicode", META_PREFIX, META_SUFFIX
        return "ascii", ALT_PREFIX, ALT_SUFFIX

    def _meta_token(self, idx: int, prefix: str, suffix: str) -> str:
        # Base-26 letters: A, B, ... Z, AA, ...
        n = idx
        letters = []
        while True:
            letters.append(chr(ord("A") + (n % 26)))
            n = n // 26 - 1
            if n < 0:
                break
        name = "".join(reversed(letters))
        return f"{prefix}{name}{suffix}"

    def build_dictionary(self, text: str) -> tuple[dict[str, str], str]:
        style, prefix, suffix = self._choose_style(text)
        groups = [mine_ngram_patterns(text, max_len=self.max_ngram, min_freq=self.min_freq)]
        if self.use_line_patterns:
            groups.append(mine_line_patterns(text, min_freq=max(2, self.min_freq - 1)))
        if self.use_char_ngrams:
            groups.append(mine_char_ngrams(text, min_freq=self.min_freq))
        cands = merge_candidates(*groups)

        dictionary: dict[str, str] = {}
        # Avoid selecting patterns that are substrings of already-selected longer ones
        # when they would not add savings after longer replacement — greedy longest-first.
        selected_texts: list[str] = []
        for pc in cands:
            if len(dictionary) >= self.max_dict_size:
                break
            # Skip if pattern already contains a meta-looking token
            if prefix in pc.text or suffix in pc.text:
                continue
            # Skip if this pattern is fully covered as exact duplicate of selected
            if any(pc.text == s for s in selected_texts):
                continue
            # Token-savings gate using char proxy (paper uses token savings)
            meta = self._meta_token(len(dictionary), prefix, suffix)
            # Estimate: each occurrence saves (len(pattern) - len(meta)), pay once for dict entry
            save = pc.count * (pc.length - len(meta)) - (len(meta) + pc.length + 8)
            if save <= 0:
                continue
            dictionary[meta] = pc.text
            selected_texts.append(pc.text)
        return dictionary, style

    def encode(self, text: str, dictionary: Mapping[str, str]) -> str:
        if not dictionary:
            return text
        # Longest pattern first to avoid partial overlaps
        items = sorted(dictionary.items(), key=lambda kv: len(kv[1]), reverse=True)
        out = text
        for meta, pattern in items:
            if not pattern:
                continue
            out = out.replace(pattern, meta)
        return out

    def decode(self, encoded: str, dictionary: Mapping[str, str]) -> str:
        if not dictionary:
            return encoded
        # Longer meta tokens first (AA before A) — sort by meta length desc then name
        items = sorted(dictionary.items(), key=lambda kv: len(kv[0]), reverse=True)
        out = encoded
        for meta, pattern in items:
            out = out.replace(meta, pattern)
        return out

    def pack_prompt(self, encoded: str, dictionary: Mapping[str, str]) -> tuple[str, str]:
        lines = [
            "You are analyzing dictionary-encoded text. Meta-tokens expand via this dictionary.",
            "Treat each meta-token as exactly its expansion. Do not invent expansions.",
            "",
            "DICTIONARY:",
        ]
        nl = "\\n"
        for meta, pattern in dictionary.items():
            # Escape newlines in pattern for readability
            safe = pattern.replace("\n", nl)
            lines.append(f"  {meta} = {safe}")
        system_dictionary = "\n".join(lines)
        packed = system_dictionary + "\n\nENCODED_BODY:\n" + encoded
        return system_dictionary, packed

    def compress(self, text: str) -> CompressResult:
        dictionary, style = self.build_dictionary(text)
        encoded = self.encode(text, dictionary)
        # Verify lossless before returning
        roundtrip = self.decode(encoded, dictionary)
        if roundtrip != text:
            # Fall back: empty dict (identity) rather than corrupt
            dictionary = {}
            encoded = text
            style = style
        system_dictionary, packed = self.pack_prompt(encoded, dictionary)
        metrics = compression_metrics(text, encoded, dictionary, packed_prompt=packed)
        return CompressResult(
            original=text,
            encoded=encoded,
            dictionary=dict(dictionary),
            system_dictionary=system_dictionary,
            packed_prompt=packed,
            metrics=metrics,
            meta_style=style,
        )

    def decompress(self, encoded: str, dictionary: Mapping[str, str]) -> str:
        return self.decode(encoded, dictionary)

    @staticmethod
    def from_packed_dict(data: Mapping[str, Any]) -> tuple[str, dict[str, str]]:
        encoded = str(data.get("encoded", ""))
        dictionary = {str(k): str(v) for k, v in dict(data.get("dictionary", {})).items()}
        return encoded, dictionary
