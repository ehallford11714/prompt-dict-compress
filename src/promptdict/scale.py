"""Streaming hierarchical compression into a fixed context budget.

Honest design:
- Full lossless reconstruction of the *corpus* uses prompt_pack + cold_store on disk.
- The prompt-resident pack holds: global/nested dictionaries, PageIndex
  directory, and a budgeted set of hot encoded pages (or page refs).
- Extreme prompt-resident ratios are only plausible for highly repetitive
  logs/JSON/code with shared templates; published dict+ICL is typically ~2–5×.
  High-entropy prose will not match those gains.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TextIO, Union

from .compressor import DictCompressor
from .metrics import estimate_tokens


PathLike = Union[str, Path]
PageSource = Iterator[tuple[int, str]]  # (page_id, text)


@dataclass
class ScaleCompressResult:
    out_dir: str
    input_tokens_est: int
    prompt_tokens_est: int
    cold_store_chars: int
    compression_factor_vs_prompt: float
    prompt_fits_budget: bool
    output_token_budget: int
    input_token_budget: int
    n_pages: int
    n_templates: int
    global_dict_size: int
    hot_pages: int
    cold_pages: int
    metrics: dict[str, Any]
    prompt_pack_path: str
    cold_store_path: str
    index_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": self.out_dir,
            "input_tokens_est": self.input_tokens_est,
            "prompt_tokens_est": self.prompt_tokens_est,
            "cold_store_chars": self.cold_store_chars,
            "compression_factor_vs_prompt": round(self.compression_factor_vs_prompt, 6),
            "prompt_fits_budget": self.prompt_fits_budget,
            "output_token_budget": self.output_token_budget,
            "input_token_budget": self.input_token_budget,
            "n_pages": self.n_pages,
            "n_templates": self.n_templates,
            "global_dict_size": self.global_dict_size,
            "hot_pages": self.hot_pages,
            "cold_pages": self.cold_pages,
            "metrics": self.metrics,
            "prompt_pack_path": self.prompt_pack_path,
            "cold_store_path": self.cold_store_path,
            "index_path": self.index_path,
            "semantics": {
                "lossless_scope": "prompt_pack + cold_store",
                "prompt_pack_alone": "addressable compressed view; may omit cold page bodies",
                "note": (
                    "Published dict+ICL is typically ~2–5×; extreme prompt-resident "
                    "ratios need highly repetitive data. Use two-tier "
                    "(prompt_pack + cold_store) for full lossless decode."
                ),
            },
        }


def _page_fingerprint(text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    head = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")[:60]
    return f"{h}|{head}"


def _template_id(text: str) -> str:
    """Normalize volatile fields lightly for template clustering (logs)."""
    # Collapse long digit runs → #
    import re

    norm = re.sub(r"\d+", "#", text)
    norm = re.sub(r"[0-9a-f]{8,}", "#hex#", norm, flags=re.I)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


class StreamingHierarchicalCompressor:
    """
    Stream pages → Level-0 local dict → Level-1 template/global dict →
    Level-2 PageIndex directory. Emit prompt_pack (≤ budget) + cold_store.
    """

    def __init__(
        self,
        *,
        input_token_budget: int = 10_000_000,
        output_token_budget: int = 1_000_000,
        page_chars: int = 16_000,  # ~4k tokens at chars/4
        max_global_dict: int = 512,
        max_local_dict: int = 48,
        hot_page_fraction: float = 0.05,
        min_freq: int = 2,
    ) -> None:
        self.input_token_budget = input_token_budget
        self.output_token_budget = output_token_budget
        self.page_chars = page_chars
        self.max_global_dict = max_global_dict
        self.max_local_dict = max_local_dict
        self.hot_page_fraction = hot_page_fraction
        self.min_freq = min_freq
        self._local = DictCompressor(
            min_freq=min_freq,
            max_dict_size=max_local_dict,
            use_line_patterns=True,
        )

    def iter_file_pages(self, path: PathLike) -> PageSource:
        path = Path(path)
        buf: list[str] = []
        size = 0
        pid = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buf.append(line)
                size += len(line)
                if size >= self.page_chars:
                    yield pid, "".join(buf)
                    pid += 1
                    buf = []
                    size = 0
        if buf:
            yield pid, "".join(buf)

    def iter_text_pages(self, text: str) -> PageSource:
        if not text:
            yield 0, ""
            return
        pid = 0
        for i in range(0, len(text), self.page_chars):
            yield pid, text[i : i + self.page_chars]
            pid += 1

    def compress_stream(
        self,
        source: Union[PathLike, PageSource, str],
        out_dir: PathLike,
        *,
        simulated_input_tokens: Optional[int] = None,
    ) -> ScaleCompressResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cold_path = out / "cold_store.jsonl"
        index_path = out / "page_index.json"
        prompt_path = out / "prompt_pack.txt"
        meta_path = out / "scale_meta.json"

        if isinstance(source, (str, Path)) and Path(source).exists() and Path(source).is_file():
            pages_iter: PageSource = self.iter_file_pages(source)
        elif isinstance(source, str):
            pages_iter = self.iter_text_pages(source)
        else:
            pages_iter = source  # type: ignore[assignment]

        # Level-1: accumulate template → expansion candidates across stream
        template_counts: dict[str, int] = {}
        template_example: dict[str, str] = {}
        page_rows: list[dict[str, Any]] = []
        input_chars = 0
        input_tokens_est = 0

        # First pass streaming write of locally-encoded pages to cold store
        with cold_path.open("w", encoding="utf-8") as cold:
            for page_id, page_text in pages_iter:
                input_chars += len(page_text)
                tok = estimate_tokens(page_text)
                input_tokens_est += tok
                tid = _template_id(page_text)
                template_counts[tid] = template_counts.get(tid, 0) + 1
                if tid not in template_example:
                    template_example[tid] = page_text[:2000]

                local = self._local.compress(page_text)
                fp = _page_fingerprint(page_text)
                row = {
                    "page_id": page_id,
                    "template_id": tid,
                    "fingerprint": fp,
                    "original_chars": len(page_text),
                    "original_tokens_est": tok,
                    "encoded": local.encoded,
                    "local_dictionary": local.dictionary,
                }
                cold.write(json.dumps(row, ensure_ascii=False) + "\n")
                page_rows.append(
                    {
                        "page_id": page_id,
                        "template_id": tid,
                        "fingerprint": fp,
                        "original_chars": len(page_text),
                        "original_tokens_est": tok,
                        "local_dict_size": len(local.dictionary),
                        "encoded_chars": len(local.encoded),
                    }
                )

                # Soft cap: if simulating huge input, allow early stop when
                # simulated budget accounting is handled by caller via generator.
                if simulated_input_tokens is None and input_tokens_est >= self.input_token_budget:
                    break

        if simulated_input_tokens is not None:
            # Scale demo: pages are representative; report target as input size
            reported_input = simulated_input_tokens
        else:
            reported_input = input_tokens_est

        # Build Level-1 / Level-2 global dictionary from frequent templates
        # Map frequent template bodies → meta tokens (shared codebook)
        global_comp = DictCompressor(
            min_freq=1,
            max_dict_size=self.max_global_dict,
            use_line_patterns=True,
            use_char_ngrams=False,
        )
        # Concatenate frequent template exemplars for mining
        frequent = sorted(template_counts.items(), key=lambda kv: kv[1], reverse=True)
        exemplar_blob = "\n".join(
            template_example[tid] for tid, c in frequent[: self.max_global_dict * 2] if c >= 1
        )
        g_res = global_comp.compress(exemplar_blob)
        global_dictionary = g_res.dictionary

        # Also map template_id → short page meta for index
        template_meta: dict[str, str] = {}
        for i, (tid, c) in enumerate(frequent[: self.max_global_dict]):
            if c < 2 and len(frequent) > 8:
                continue
            template_meta[tid] = f"⟦T{i}⟧"

        # PageIndex directory (Level-2)
        page_index = {
            "version": "0.3.0",
            "kind": "streaming_pageindex",
            "n_pages": len(page_rows),
            "n_templates": len(template_counts),
            "template_meta": template_meta,
            "global_dictionary": global_dictionary,
            "pages": page_rows,
            "decode_order": [
                "1. Expand GLOBAL_DICT meta-tokens",
                "2. Resolve page via PAGE_INDEX / template_meta",
                "3. Expand page local_dictionary from cold_store",
                "4. Concatenate pages in page_id order for full corpus",
            ],
        }
        index_path.write_text(json.dumps(page_index, ensure_ascii=False, indent=2), encoding="utf-8")

        # Two-tier prompt pack: dicts + index + hot pages only
        n_hot = max(1, int(len(page_rows) * self.hot_page_fraction)) if page_rows else 0
        # Prefer diversity of templates in hot set
        seen_t: set[str] = set()
        hot_ids: list[int] = []
        for row in page_rows:
            if len(hot_ids) >= n_hot:
                break
            if row["template_id"] not in seen_t or len(seen_t) >= len(template_counts):
                hot_ids.append(row["page_id"])
                seen_t.add(row["template_id"])
        while len(hot_ids) < n_hot and len(hot_ids) < len(page_rows):
            for row in page_rows:
                if row["page_id"] not in hot_ids:
                    hot_ids.append(row["page_id"])
                if len(hot_ids) >= n_hot:
                    break

        hot_set = set(hot_ids)
        hot_bodies: list[str] = []
        # Re-read cold store for hot pages
        if hot_set:
            with cold_path.open("r", encoding="utf-8") as cold:
                for line in cold:
                    row = json.loads(line)
                    if row["page_id"] in hot_set:
                        # Apply global dict encode on local encoded body for extra squeeze
                        body = global_comp.encode(row["encoded"], global_dictionary)
                        hot_bodies.append(
                            f"<<<PAGE {row['page_id']} T={row['template_id']}>>>\n{body}"
                        )

        nl = "\\n"
        gdict_lines = [f"  {k} = {v.replace(chr(10), nl)}" for k, v in global_dictionary.items()]
        tmeta_lines = [
            f"  {v} → template {k} (count={template_counts.get(k, 0)})"
            for k, v in template_meta.items()
        ]
        slim_index = [
            {
                "page_id": r["page_id"],
                "template_id": r["template_id"],
                "fingerprint": r["fingerprint"],
                "tokens_est": r["original_tokens_est"],
                "in_prompt": r["page_id"] in hot_set,
            }
            for r in page_rows
        ]

        def _build_pack(bodies: list[str]) -> str:
            return "\n".join(
                [
                    "BUDGETED PAGEINDEX PACK",
                    f"input_tokens_est={reported_input} "
                    f"input_budget={self.input_token_budget} "
                    f"output_budget={self.output_token_budget}",
                    "Lossless full decode requires cold_store.jsonl + this pack.",
                    "Prompt holds dictionaries, directory, and HOT pages only.",
                    "",
                    "GLOBAL_DICT:",
                    *gdict_lines,
                    "",
                    "TEMPLATE_CODEBOOK:",
                    *tmeta_lines,
                    "",
                    "PAGE_INDEX:",
                    json.dumps(slim_index, ensure_ascii=False),
                    "",
                    "HOT_ENCODED_PAGES:",
                    *bodies,
                    "",
                    "COLD_STORE_REF: cold_store.jsonl",
                ]
            )

        packed = _build_pack(hot_bodies)

        # If packed exceeds budget, drop hot pages until under budget (keep dicts+index)
        prompt_tok = estimate_tokens(packed)
        while prompt_tok > self.output_token_budget and hot_bodies:
            hot_bodies.pop()
            packed = _build_pack(hot_bodies)
            prompt_tok = estimate_tokens(packed)

        prompt_path.write_text(packed, encoding="utf-8")
        prompt_tok = estimate_tokens(packed)
        fits = prompt_tok <= self.output_token_budget
        factor = (reported_input / prompt_tok) if prompt_tok else 0.0

        result = ScaleCompressResult(
            out_dir=str(out.resolve()),
            input_tokens_est=reported_input,
            prompt_tokens_est=prompt_tok,
            cold_store_chars=cold_path.stat().st_size,
            compression_factor_vs_prompt=factor,
            prompt_fits_budget=fits,
            output_token_budget=self.output_token_budget,
            input_token_budget=self.input_token_budget,
            n_pages=len(page_rows),
            n_templates=len(template_counts),
            global_dict_size=len(global_dictionary),
            hot_pages=len(hot_bodies),
            cold_pages=max(0, len(page_rows) - len(hot_bodies)),
            metrics={
                "input_chars_streamed": input_chars,
                "streamed_tokens_est": input_tokens_est,
                "simulated_input_tokens": simulated_input_tokens,
                "unique_templates": len(template_counts),
                "target_ratio": (
                    reported_input / self.output_token_budget if self.output_token_budget else 0
                ),
                "achieved_factor_prompt_only": factor,
                "extreme_prompt_resident_ratio": factor >= 50.0,
                "lossless_requires_cold_store": True,
            },
            prompt_pack_path=str(prompt_path.resolve()),
            cold_store_path=str(cold_path.resolve()),
            index_path=str(index_path.resolve()),
        )
        meta_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    def decompress_page(self, out_dir: PathLike, page_id: int) -> str:
        """Lossless decode of a single page from cold_store (+ global dict unused for local)."""
        out = Path(out_dir)
        cold_path = out / "cold_store.jsonl"
        with cold_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row["page_id"]) == page_id:
                    return self._local.decode(row["encoded"], row["local_dictionary"])
        raise KeyError(f"page_id {page_id} not found in cold store")

    def decompress_all(self, out_dir: PathLike) -> str:
        out = Path(out_dir)
        cold_path = out / "cold_store.jsonl"
        pages: list[tuple[int, str]] = []
        with cold_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = self._local.decode(row["encoded"], row["local_dictionary"])
                pages.append((int(row["page_id"]), text))
        pages.sort(key=lambda x: x[0])
        return "".join(t for _, t in pages)


# Preferred name: budgeted ultra-long context path
BudgetedContextCompressor = StreamingHierarchicalCompressor
# Back-compat alias
MillionTokenBudgetCompressor = StreamingHierarchicalCompressor


# ---------------------------------------------------------------------------
# Synthetic scale generator (emulates ultra-long redundancy without huge RAM)
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATES = [
    "2026-07-09T12:00:00Z INFO svc=api request_id={rid} path=/v1/users method=GET status=200 latency_ms={lat} msg=ok",
    "2026-07-09T12:00:00Z WARN svc=api request_id={rid} path=/v1/orders method=POST status=429 latency_ms={lat} msg=rate_limited",
    "2026-07-09T12:00:00Z ERROR svc=worker job={rid} attempt={lat} error=timeout retry=true queue=default",
    '{{"event":"metric","host":"node-{rid}","cpu":{lat},"mem":{lat},"tags":["prod","us-east"]}}',
    "SELECT id, name, created_at FROM users WHERE tenant_id={rid} AND active=1 LIMIT {lat};",
    "def process_batch_{rid}(items):\n    for item in items:\n        validate(item)\n        emit(item, region={lat})\n    return len(items)\n",
]


def generate_repetitive_pages(
    *,
    n_pages: int,
    page_lines: int = 40,
    n_templates: int = 32,
    seed: int = 42,
) -> Iterator[tuple[int, str]]:
    """Yield pages built from a small template set (high redundancy)."""
    import random

    rng = random.Random(seed)
    templates = []
    for i in range(max(1, n_templates)):
        base = DEFAULT_TEMPLATES[i % len(DEFAULT_TEMPLATES)]
        templates.append(base.replace("svc=api", f"svc=api{i % 7}"))

    for pid in range(n_pages):
        lines = []
        for _ in range(page_lines):
            t = templates[rng.randint(0, len(templates) - 1)]
            lines.append(t.format(rid=rng.randint(1, 50), lat=rng.randint(1, 200)))
        yield pid, "\n".join(lines) + "\n"


def estimate_generator_tokens(n_pages: int, page_lines: int = 40, n_templates: int = 32) -> int:
    """Estimate tokens produced by generate_repetitive_pages without materializing all."""
    sample = list(generate_repetitive_pages(n_pages=3, page_lines=page_lines, n_templates=n_templates))
    avg = sum(estimate_tokens(t) for _, t in sample) / max(1, len(sample))
    return int(avg * n_pages)


def run_scale_demo(
    *,
    target_in: int = 10_000_000,
    target_out: int = 1_000_000,
    simulate: bool = True,
    out_dir: PathLike = ".scale_demo",
    max_materialized_pages: int = 400,
) -> ScaleCompressResult:
    """
    Demonstrate packing an ultra-long (simulated) corpus into ``target_out``.

    In --simulate mode we materialize a modest number of highly repetitive pages
    that stand in for the redundancy structure of a large corpus, then
    report metrics against the *target* input size (honest labeling in metrics).
    """
    out = Path(out_dir)
    if simulate:
        # Choose page count for a manageable demo disk footprint
        n_pages = min(max_materialized_pages, 400)
        pages = generate_repetitive_pages(n_pages=n_pages, page_lines=50, n_templates=24)
        # Project: if each page ~tok, how many pages would target_in need?
        sample_tok = estimate_generator_tokens(5, page_lines=50, n_templates=24) / 5
        projected_pages = int(target_in / max(1, sample_tok))
        comp = BudgetedContextCompressor(
            input_token_budget=target_in,
            output_token_budget=target_out,
            page_chars=50_000,
            hot_page_fraction=0.08,
            max_global_dict=256,
        )
        result = comp.compress_stream(pages, out, simulated_input_tokens=target_in)
        result.metrics["simulation"] = True
        result.metrics["materialized_pages"] = n_pages
        result.metrics["projected_pages_for_target"] = projected_pages
        result.metrics["avg_tokens_per_page_est"] = sample_tok
        result.metrics["honesty"] = (
            "Input token count is the TARGET (simulated). Disk holds a redundancy-equivalent "
            "sample, not a full materialization of target_in tokens. "
            "Compression factor uses target_in / prompt_tokens."
        )
        # Re-write meta with updated metrics
        (out / "scale_meta.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result

    # Non-simulate: write a large-ish file capped then compress
    n_pages = min(max_materialized_pages, 2000)
    path = out / "synthetic_corpus.txt"
    out.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _, page in generate_repetitive_pages(n_pages=n_pages, page_lines=80, n_templates=24):
            f.write(page)
            f.write("\n")
    comp = BudgetedContextCompressor(
        input_token_budget=target_in,
        output_token_budget=target_out,
    )
    return comp.compress_stream(path, out)
