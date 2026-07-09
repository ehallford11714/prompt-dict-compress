"""Token estimation and compression metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


def estimate_tokens(text: str, *, prefer_tiktoken: bool = True) -> int:
    """Estimate token count. Uses tiktoken if available, else ~chars/4."""
    if not text:
        return 0
    if prefer_tiktoken:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)


@dataclass
class Metrics:
    original_tokens: int
    compressed_tokens: int
    dictionary_tokens: int
    packed_tokens: int
    ratio: float  # 1 - packed/original (token savings fraction)
    compression_factor: float  # original / packed
    dictionary_entries: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "dictionary_tokens": self.dictionary_tokens,
            "packed_tokens": self.packed_tokens,
            "ratio": round(self.ratio, 6),
            "compression_factor": round(self.compression_factor, 6),
            "dictionary_entries": self.dictionary_entries,
            "notes": self.notes,
        }


def compression_metrics(
    original: str,
    encoded: str,
    dictionary: Mapping[str, str],
    *,
    packed_prompt: Optional[str] = None,
) -> Metrics:
    """Compute token metrics including dictionary overhead."""
    dict_text = "\n".join(f"{k} = {v}" for k, v in dictionary.items())
    packed = packed_prompt
    if packed is None:
        packed = f"DICTIONARY:\n{dict_text}\n\nENCODED:\n{encoded}"

    ot = estimate_tokens(original)
    et = estimate_tokens(encoded)
    dt = estimate_tokens(dict_text)
    pt = estimate_tokens(packed)
    factor = (ot / pt) if pt else 0.0
    ratio = 1.0 - (pt / ot) if ot else 0.0
    return Metrics(
        original_tokens=ot,
        compressed_tokens=et,
        dictionary_tokens=dt,
        packed_tokens=pt,
        ratio=ratio,
        compression_factor=factor,
        dictionary_entries=len(dictionary),
    )
