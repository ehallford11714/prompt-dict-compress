"""Compaction pillar: manage an agent working set over time.

Compaction *manages* conversation / tool-trace context under a token budget.
It can optionally route repetitive bodies through lossless dictionary encoding,
or emit lossy summary stubs for recoverable redundancy.

Contrast with :mod:`promptdict.compress` (encode a corpus once) and
:mod:`promptdict.recall` (pull cold pages back into context).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from .compressor import CompressResult, DictCompressor
from .metrics import estimate_tokens

Message = Mapping[str, Any]
CompactMode = Literal["lossless_dict", "lossy_stub", "auto"]


@dataclass
class ColdRef:
    """Reference to content moved out of the active working set."""

    ref_id: str
    role: str
    kind: str  # "dict_encoded" | "summary_stub"
    original_tokens_est: int
    preview: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "role": self.role,
            "kind": self.kind,
            "original_tokens_est": self.original_tokens_est,
            "preview": self.preview,
            "payload": self.payload,
        }


@dataclass
class CompactResult:
    """Result of :func:`compact_messages`."""

    messages: list[dict[str, Any]]
    packed_prompt: str
    cold_refs: list[ColdRef]
    mode: str
    original_tokens_est: int
    compacted_tokens_est: int
    budget: int
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "packed_prompt": self.packed_prompt,
            "cold_refs": [c.to_dict() for c in self.cold_refs],
            "mode": self.mode,
            "original_tokens_est": self.original_tokens_est,
            "compacted_tokens_est": self.compacted_tokens_est,
            "budget": self.budget,
            "metrics": self.metrics,
            "version": "0.3.0",
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def _msg_content(msg: Mapping[str, Any]) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(c)


def _normalize_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        out.append({"role": role, "content": _msg_content(m)})
    return out


def _merge_adjacent_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []
    merged: list[dict[str, Any]] = [dict(messages[0])]
    for m in messages[1:]:
        if m["role"] == merged[-1]["role"]:
            merged[-1]["content"] = (
                merged[-1]["content"].rstrip() + "\n\n" + m["content"].lstrip()
            )
        else:
            merged.append(dict(m))
    return merged


_RECOVERABLE_RE = re.compile(
    r"(?i)(tool[_ ]?result|function[_ ]?response|observation|"
    r"```json|status=\d{3}|request_id=|latency_ms=)"
)


def _looks_repetitive(text: str) -> bool:
    if len(text) < 200:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 8:
        uniq = len(set(lines))
        if uniq / max(1, len(lines)) < 0.55:
            return True
    return bool(_RECOVERABLE_RE.search(text)) and estimate_tokens(text) >= 80


def _lossy_stub(text: str, *, max_preview: int = 240) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    head = lines[:2]
    tail = lines[-1:] if n > 3 else []
    preview = " | ".join(head + tail)
    if len(preview) > max_preview:
        preview = preview[: max_preview - 1] + "…"
    return (
        f"[compacted stub: ~{estimate_tokens(text)} tokens, {n} lines; "
        f"recoverable detail omitted] {preview}"
    )


def _pack_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        parts.append(f"<<<{role}>>>\n{m.get('content', '')}")
    return "\n\n".join(parts)


def compact_messages(
    messages: Sequence[Message],
    *,
    budget: int = 8_000,
    mode: CompactMode = "auto",
    keep_system: bool = True,
    keep_last_n: int = 4,
    min_freq: int = 2,
    max_dict_size: int = 128,
) -> CompactResult:
    """Compact a chat/tool message list into a budgeted working set.

    Parameters
    ----------
    messages:
        OpenAI-style ``{"role", "content"}`` mappings.
    budget:
        Approximate token budget for the compacted packed prompt.
    mode:
        ``lossless_dict`` — encode bulky turns with :class:`DictCompressor`
        and stash full dict payloads in ``cold_refs``.
        ``lossy_stub`` — replace bulky turns with short stubs (not invertible).
        ``auto`` — prefer lossless_dict when patterns mine well; else stub.
    keep_system:
        Always retain ``system`` messages verbatim in the working set.
    keep_last_n:
        Always retain the last N non-system messages verbatim.
    """
    normalized = _normalize_messages(messages)
    original_tok = estimate_tokens(_pack_messages(normalized))
    merged = _merge_adjacent_same_role(normalized)

    system_msgs = [m for m in merged if m["role"] == "system"] if keep_system else []
    non_system = [m for m in merged if m["role"] != "system"] if keep_system else list(merged)

    protect_from = max(0, len(non_system) - keep_last_n)
    protected = non_system[protect_from:]
    candidates = non_system[:protect_from]

    cold_refs: list[ColdRef] = []
    compacted_mid: list[dict[str, Any]] = []
    compressor = DictCompressor(min_freq=min_freq, max_dict_size=max_dict_size)
    resolved_mode = mode

    for i, msg in enumerate(candidates):
        content = msg["content"]
        tok = estimate_tokens(content)
        # Small or non-repetitive turns stay as-is
        if tok < 120 or not _looks_repetitive(content):
            compacted_mid.append(dict(msg))
            continue

        use_dict = mode in ("lossless_dict", "auto")
        result: Optional[CompressResult] = None
        if use_dict:
            result = compressor.compress(content)
            # Only keep dict path if we actually saved tokens in the packed form
            if result.dictionary and result.metrics.packed_tokens < tok * 0.92:
                ref_id = f"c{len(cold_refs)}"
                cold_refs.append(
                    ColdRef(
                        ref_id=ref_id,
                        role=msg["role"],
                        kind="dict_encoded",
                        original_tokens_est=tok,
                        preview=content[:160].replace("\n", " "),
                        payload={
                            "encoded": result.encoded,
                            "dictionary": result.dictionary,
                            "packed_prompt": result.packed_prompt,
                        },
                    )
                )
                compacted_mid.append(
                    {
                        "role": msg["role"],
                        "content": (
                            f"[cold_ref:{ref_id} dict-encoded "
                            f"~{result.metrics.packed_tokens} tok; "
                            f"full lossless body in cold_refs]\n"
                            f"{result.packed_prompt}"
                        ),
                    }
                )
                if mode == "auto":
                    resolved_mode = "lossless_dict"
                continue
            if mode == "lossless_dict":
                # Forced lossless but no savings — keep original
                compacted_mid.append(dict(msg))
                continue

        # Lossy stub path
        stub = _lossy_stub(content)
        ref_id = f"c{len(cold_refs)}"
        cold_refs.append(
            ColdRef(
                ref_id=ref_id,
                role=msg["role"],
                kind="summary_stub",
                original_tokens_est=tok,
                preview=content[:160].replace("\n", " "),
                payload={"original": content},
            )
        )
        compacted_mid.append(
            {
                "role": msg["role"],
                "content": f"[cold_ref:{ref_id}] {stub}",
            }
        )
        if mode == "auto":
            resolved_mode = "lossy_stub"

    working = system_msgs + compacted_mid + protected

    # If still over budget, drop oldest mid messages into stubs (lossy)
    packed = _pack_messages(working)
    while estimate_tokens(packed) > budget and compacted_mid:
        victim = compacted_mid.pop(0)
        # Find corresponding entry in working and stub it further
        for j, w in enumerate(working):
            if w is victim or (
                w.get("role") == victim.get("role")
                and w.get("content") == victim.get("content")
            ):
                tok = estimate_tokens(str(victim.get("content", "")))
                ref_id = f"c{len(cold_refs)}"
                cold_refs.append(
                    ColdRef(
                        ref_id=ref_id,
                        role=str(victim.get("role", "user")),
                        kind="summary_stub",
                        original_tokens_est=tok,
                        preview=str(victim.get("content", ""))[:120],
                        payload={"original": victim.get("content", "")},
                    )
                )
                working[j] = {
                    "role": victim["role"],
                    "content": f"[cold_ref:{ref_id}] {_lossy_stub(str(victim.get('content', '')))}",
                }
                break
        packed = _pack_messages(working)

    compacted_tok = estimate_tokens(packed)
    return CompactResult(
        messages=working,
        packed_prompt=packed,
        cold_refs=cold_refs,
        mode=resolved_mode if mode == "auto" else mode,
        original_tokens_est=original_tok,
        compacted_tokens_est=compacted_tok,
        budget=budget,
        metrics={
            "n_input_messages": len(normalized),
            "n_output_messages": len(working),
            "n_cold_refs": len(cold_refs),
            "n_merged": len(merged),
            "fits_budget": compacted_tok <= budget,
            "compression_factor": (
                round(original_tok / compacted_tok, 4) if compacted_tok else 0.0
            ),
        },
    )


def expand_cold_ref(ref: Union[ColdRef, Mapping[str, Any]]) -> str:
    """Restore original text from a cold_ref when available."""
    if isinstance(ref, ColdRef):
        data = ref.to_dict()
    else:
        data = dict(ref)
    kind = data.get("kind")
    payload = dict(data.get("payload") or {})
    if kind == "dict_encoded":
        enc = str(payload.get("encoded", ""))
        dictionary = {str(k): str(v) for k, v in dict(payload.get("dictionary", {})).items()}
        return DictCompressor().decompress(enc, dictionary)
    if "original" in payload:
        return str(payload["original"])
    return str(data.get("preview", ""))
