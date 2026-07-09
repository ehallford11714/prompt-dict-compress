"""Multi-scale pattern mining for dictionary-encoding compression."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class PatternCandidate:
    text: str
    count: int
    length: int

    @property
    def savings_score(self) -> int:
        """Rough char savings if replaced by a short meta-token (~6 chars)."""
        meta_len = 6
        return (self.count - 1) * (self.length - meta_len) - meta_len


def _whitespace_units(text: str) -> list[str]:
    return [u for u in text.split() if u]


def mine_ngram_patterns(
    text: str,
    *,
    min_len: int = 2,
    max_len: int = 12,
    min_freq: int = 3,
    max_candidates: int = 500,
) -> list[PatternCandidate]:
    """Mine frequent whitespace n-grams (multi-scale, longest-first friendly)."""
    units = _whitespace_units(text)
    if len(units) < min_len:
        return []

    counts: Counter[str] = Counter()
    upper = min(max_len, len(units))
    for n in range(upper, min_len - 1, -1):
        for i in range(len(units) - n + 1):
            gram = " ".join(units[i : i + n])
            counts[gram] += 1

    cands: list[PatternCandidate] = []
    for gram, c in counts.items():
        if c < min_freq:
            continue
        length = len(gram)
        if length < 8:  # too short to be worth a meta-token usually
            continue
        pc = PatternCandidate(text=gram, count=c, length=length)
        if pc.savings_score > 0:
            cands.append(pc)

    cands.sort(key=lambda p: (p.savings_score, p.length, p.count), reverse=True)
    return cands[:max_candidates]


def mine_line_patterns(
    text: str,
    *,
    min_freq: int = 2,
    min_line_len: int = 16,
    max_candidates: int = 300,
) -> list[PatternCandidate]:
    """Mine repeated full lines (strong for logs / JSONL)."""
    lines = [ln for ln in text.splitlines() if len(ln) >= min_line_len]
    counts = Counter(lines)
    cands: list[PatternCandidate] = []
    for line, c in counts.items():
        if c < min_freq:
            continue
        pc = PatternCandidate(text=line, count=c, length=len(line))
        if pc.savings_score > 0:
            cands.append(pc)
    cands.sort(key=lambda p: (p.savings_score, p.length, p.count), reverse=True)
    return cands[:max_candidates]


def mine_char_ngrams(
    text: str,
    *,
    lengths: Sequence[int] = (32, 24, 16, 12),
    min_freq: int = 3,
    max_candidates: int = 200,
) -> list[PatternCandidate]:
    """Mine repeated character substrings (for dense repetitive blobs)."""
    counts: Counter[str] = Counter()
    for n in lengths:
        if n > len(text):
            continue
        step = max(1, n // 4)
        for i in range(0, len(text) - n + 1, step):
            counts[text[i : i + n]] += 1
    cands: list[PatternCandidate] = []
    for s, c in counts.items():
        if c < min_freq:
            continue
        pc = PatternCandidate(text=s, count=c, length=len(s))
        if pc.savings_score > 0:
            cands.append(pc)
    cands.sort(key=lambda p: (p.savings_score, p.length, p.count), reverse=True)
    return cands[:max_candidates]


def merge_candidates(*groups: Iterable[PatternCandidate]) -> list[PatternCandidate]:
    """Deduplicate by text, keep highest count, sort by savings."""
    best: dict[str, PatternCandidate] = {}
    for group in groups:
        for pc in group:
            prev = best.get(pc.text)
            if prev is None or pc.count > prev.count:
                best[pc.text] = pc
    out = list(best.values())
    out.sort(key=lambda p: (p.savings_score, p.length, p.count), reverse=True)
    return out


def iter_pages(text: str, page_size: int) -> Iterator[tuple[int, str]]:
    """Split text into fixed-size character pages (approx token pages via chars)."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if not text:
        yield 0, ""
        return
    idx = 0
    for i in range(0, len(text), page_size):
        yield idx, text[i : i + page_size]
        idx += 1
