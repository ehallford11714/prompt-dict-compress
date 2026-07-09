#!/usr/bin/env python3
"""Offline experiment: does compaction preserve / improve reasoning-trace structure?

Compares raw vs lossless dict-compress vs promptdict compact vs naive truncate
on synthetic reasoning traces with repetitive filler + critical goal/constraint
isolates.

Usage (from PromptDictCompress repo root)::

    python experiments/reasoning_trace_compaction.py
    python experiments/reasoning_trace_compaction.py --budget 2500 --out-dir experiments/results

No paid APIs required. Soft-imports intentisolates / llmintent.isolates.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Path bootstrap: run without install
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Sibling research packages (optional)
_RESEARCH = _ROOT.parent
for _sib in (
    _RESEARCH / "IntentIsolates" / "src",
    _RESEARCH / "LLMIntent" / "src",
):
    if _sib.is_dir() and str(_sib) not in sys.path:
        sys.path.insert(0, str(_sib))

from promptdict.compact import (  # noqa: E402
    ColdRef,
    compact_messages,
    expand_cold_ref,
)
from promptdict.compressor import DictCompressor  # noqa: E402
from promptdict.metrics import estimate_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Soft import: isolates / motifs / trajectories
# ---------------------------------------------------------------------------
_ISOLATES_NOTE = "unavailable"
_identify_isolates: Optional[Callable[..., Any]] = None
_form_motifs: Optional[Callable[..., Any]] = None
_trajectory_from_motifs: Optional[Callable[..., Any]] = None


def _try_load_isolates() -> None:
    global _ISOLATES_NOTE, _identify_isolates, _form_motifs, _trajectory_from_motifs
    try:
        from intentisolates import (  # type: ignore
            form_motifs,
            identify_isolates,
            trajectory_from_motifs,
        )

        _identify_isolates = identify_isolates
        _form_motifs = form_motifs
        _trajectory_from_motifs = trajectory_from_motifs
        _ISOLATES_NOTE = "intentisolates"
        return
    except ImportError:
        pass
    try:
        from llmintent.isolates import (  # type: ignore
            form_motifs,
            identify_isolates,
            trajectory_from_motifs,
        )

        _identify_isolates = identify_isolates
        _form_motifs = form_motifs
        _trajectory_from_motifs = trajectory_from_motifs
        _ISOLATES_NOTE = "llmintent.isolates"
        return
    except ImportError:
        _ISOLATES_NOTE = (
            "missing — using keyword typology proxy "
            "(install IntentIsolates or LLMIntent for full pipeline)"
        )


_try_load_isolates()

# ---------------------------------------------------------------------------
# Fixtures: synthetic reasoning traces
# ---------------------------------------------------------------------------

_LOG_LINE = (
    "tool_result status=200 request_id=req-{i:04d} latency_ms={lat} "
    "payload={{\"ok\": true, \"batch\": {i}, \"msg\": \"heartbeat ok\"}}"
)


def _repetitive_tool_block(n: int = 24, start: int = 0) -> str:
    lines = []
    for i in range(start, start + n):
        lines.append(_LOG_LINE.format(i=i, lat=40 + (i % 17)))
    return "\n".join(lines)


def _fixture_messages(
    *,
    goal: str,
    constraints: Sequence[str],
    mid_constraints: Sequence[str],
    plan: Sequence[str],
    outcome: str,
    filler_n: int = 28,
) -> list[dict[str, str]]:
    """Build a multi-turn trace with critical constraints buried mid-history.

    Early turns carry the goal + soft constraints; *mid_constraints* sit between
    large tool dumps so naive head/tail truncate is likely to drop them while
    dict-compact / protect-compact can keep or restore them.
    """
    soft = constraints[:1] if constraints else []
    msgs: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a careful planning assistant. Preserve all goals and constraints. "
                "Do not invent requirements."
            ),
        },
        {
            "role": "user",
            "content": (
                f"GOAL: {goal}\n"
                + "\n".join(f"CONSTRAINT: {c}" for c in soft)
                + "\nPlease produce a step plan and report the outcome."
            ),
        },
        {
            "role": "assistant",
            "content": "PLAN:\n" + "\n".join(f"- {p}" for p in plan),
        },
        {
            "role": "tool",
            "content": (
                "BEGIN_TOOL_DUMP\n"
                + _repetitive_tool_block(filler_n, 0)
                + "\nEND_TOOL_DUMP"
            ),
        },
        {
            "role": "user",
            "content": (
                "CRITICAL MID-TRACE CONSTRAINTS (must survive compaction):\n"
                + "\n".join(f"CONSTRAINT: {c}" for c in mid_constraints)
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Acknowledged mid-trace constraints. Continuing analysis on the tool stream…\n"
                + _repetitive_tool_block(max(12, filler_n // 2), 100)
            ),
        },
        {
            "role": "tool",
            "content": (
                "BEGIN_TOOL_DUMP\n"
                + _repetitive_tool_block(filler_n, 200)
                + "\nEND_TOOL_DUMP"
            ),
        },
        {
            "role": "user",
            "content": (
                "Reminder: mid-trace constraints still apply. "
                "Summarize whether the goal was met."
            ),
        },
        {
            "role": "assistant",
            "content": f"OUTCOME: {outcome}",
        },
    ]
    return msgs


def build_fixtures() -> list[dict[str, Any]]:
    """3–5 synthetic traces with gold key isolates (incl. mid-trace constraints)."""
    specs = [
        {
            "id": "deploy_budget",
            "goal": "I want to deploy the billing API to staging by Friday.",
            "constraints": [
                "Deadline is Friday 17:00 UTC unless the on-call approves.",
            ],
            "mid_constraints": [
                "Cannot exceed a $200 cloud budget.",
                "Must not change the public schema without a migration review.",
                "Require blue-green cutover with instant rollback.",
            ],
            "plan": [
                "Run canary using the deploy tool via the staging pipeline.",
                "Check latency_ms and status codes in the tool stream.",
                "If budget constraint binds, roll back.",
            ],
            "outcome": (
                "Result: canary passed; therefore staging deploy is ready. "
                "Budget remaining within the $200 limit."
            ),
            "gold_keys": {
                "goal": ["deploy the billing API to staging"],
                "constraint": [
                    "$200 cloud budget",
                    "public schema",
                    "blue-green cutover",
                ],
                "outcome": ["canary passed", "Budget remaining"],
            },
        },
        {
            "id": "privacy_export",
            "goal": "My goal is to export user analytics for the Q2 report.",
            "constraints": [
                "Only if legal review has signed off.",
            ],
            "mid_constraints": [
                "Cannot include PII fields (email, phone, name).",
                "Require differential privacy epsilon <= 1.0.",
                "Must not ship row-level identifiers in the CSV.",
            ],
            "plan": [
                "Build the export job using the warehouse tool.",
                "Apply DP noise via the privacy library.",
                "Submit the artifact to the report folder.",
            ],
            "outcome": (
                "Outcome: export completed without PII; consequently the Q2 pack is ready."
            ),
            "gold_keys": {
                "goal": ["export user analytics"],
                "constraint": ["PII", "epsilon", "row-level identifiers"],
                "outcome": ["export completed", "Q2 pack"],
            },
        },
        {
            "id": "incident_mitigate",
            "goal": "Aim to mitigate the rate-limit incident on checkout.",
            "constraints": [
                "Provided that error budget remains above 5%.",
            ],
            "mid_constraints": [
                "Must not disable fraud checks.",
                "Cannot raise global QPS above 120 without approval.",
                "Require canary on 5% of checkout traffic first.",
            ],
            "plan": [
                "Implement a local cache using Redis via the ops tool.",
                "Execute a gradual QPS ramp.",
                "Write a postmortem outline.",
            ],
            "outcome": (
                "Result: latency recovered; therefore the incident is mitigated "
                "without disabling fraud checks."
            ),
            "gold_keys": {
                "goal": ["mitigate the rate-limit incident"],
                "constraint": ["fraud checks", "QPS above 120", "canary on 5%"],
                "outcome": ["latency recovered", "incident is mitigated"],
            },
        },
        {
            "id": "research_summary",
            "goal": "I need to produce a literature summary on prompt compression.",
            "constraints": [
                "Limit the draft to 800 words.",
            ],
            "mid_constraints": [
                "Must cite at least LLMLingua and dictionary-encoding work.",
                "Cannot claim LLM task equivalence without a citation.",
                "Require a limitations section on lossy vs lossless methods.",
            ],
            "plan": [
                "Gather papers using the search tool.",
                "Create an outline through thematic clustering.",
                "Write the summary and attach citations.",
            ],
            "outcome": (
                "Outcome: draft produced under the word limit; yields a citable summary."
            ),
            "gold_keys": {
                "goal": ["literature summary on prompt compression"],
                "constraint": ["LLMLingua", "task equivalence", "limitations section"],
                "outcome": ["draft produced", "citable summary"],
            },
        },
        {
            "id": "sql_migration",
            "goal": "Objective: migrate the orders table to partitioned storage.",
            "constraints": [
                "Budget for the migration window is 90 minutes.",
            ],
            "mid_constraints": [
                "Require a zero-downtime cutover.",
                "Must not drop the legacy table until checksums match.",
                "Cannot lock the orders table for more than 30 seconds.",
            ],
            "plan": [
                "Create the partitioned table using the migration tool.",
                "Run dual-write and checksum jobs.",
                "Cut over and monitor.",
            ],
            "outcome": (
                "Result: checksums matched; consequently cutover succeeded within budget."
            ),
            "gold_keys": {
                "goal": ["migrate the orders table"],
                "constraint": ["zero-downtime", "checksums", "30 seconds"],
                "outcome": ["checksums matched", "cutover succeeded"],
            },
        },
    ]
    out = []
    for s in specs:
        msgs = _fixture_messages(
            goal=s["goal"],
            constraints=s["constraints"],
            mid_constraints=s["mid_constraints"],
            plan=s["plan"],
            outcome=s["outcome"],
            filler_n=36,
        )
        packed = _pack(msgs)
        out.append(
            {
                "id": s["id"],
                "messages": msgs,
                "packed": packed,
                "gold_keys": s["gold_keys"],
                "gold_phrases": {
                    "goal": [s["goal"]],
                    "constraint": list(s["constraints"]) + list(s["mid_constraints"]),
                    "outcome": [s["outcome"]],
                },
            }
        )
    return out


def _pack(messages: Sequence[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<<<{str(m.get('role', 'user')).upper()}>>>\n{m.get('content', '')}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def condition_raw(fixture: dict[str, Any], **_: Any) -> dict[str, Any]:
    text = fixture["packed"]
    return {
        "condition": "raw",
        "visible_text": text,
        "restored_text": text,
        "tokens_before": estimate_tokens(text),
        "tokens_after": estimate_tokens(text),
        "lossless_exact": True,
        "meta": {"n_messages": len(fixture["messages"])},
    }


def condition_compress(fixture: dict[str, Any], **_: Any) -> dict[str, Any]:
    text = fixture["packed"]
    before = estimate_tokens(text)
    # Lower min_freq so repetitive tool lines form a dict on these fixtures
    comp = DictCompressor(min_freq=2, max_dict_size=128, use_line_patterns=True)
    result = comp.compress(text)
    restored = comp.decompress(result.encoded, result.dictionary)
    after = estimate_tokens(result.packed_prompt)
    return {
        "condition": "compress",
        "visible_text": result.packed_prompt,
        "restored_text": restored,
        "tokens_before": before,
        "tokens_after": after,
        "lossless_exact": restored == text,
        "meta": {
            "dict_size": len(result.dictionary),
            "compression_factor": result.metrics.compression_factor,
            "packed_tokens": result.metrics.packed_tokens,
        },
    }


def condition_compact(
    fixture: dict[str, Any],
    *,
    budget: int,
    mode: str = "lossless_dict",
    **_: Any,
) -> dict[str, Any]:
    text = fixture["packed"]
    before = estimate_tokens(text)
    result = compact_messages(
        fixture["messages"],
        budget=budget,
        mode=mode,  # type: ignore[arg-type]
        keep_system=True,
        keep_last_n=3,
        min_freq=2,
        max_dict_size=128,
    )
    # Reconstruct best-effort restored text from cold_refs + working messages
    restored_parts: list[str] = []
    ref_by_id = {c.ref_id: c for c in result.cold_refs}
    for m in result.messages:
        content = str(m.get("content", ""))
        m_ref = re.search(r"\[cold_ref:(c\d+)", content)
        if m_ref and m_ref.group(1) in ref_by_id:
            restored_parts.append(expand_cold_ref(ref_by_id[m_ref.group(1)]))
        else:
            restored_parts.append(content)
    restored = "\n\n".join(restored_parts)
    # Visible working set (what the model would see without expanding cold originals
    # that are only in payload — packed_prompt already embeds dict packs for lossless path)
    visible = result.packed_prompt
    # Exact lossless vs original packed is strict; for compact we measure
    # whether all gold key phrases survive in restored (via cold_refs) and/or visible.
    return {
        "condition": "compact",
        "visible_text": visible,
        "restored_text": restored,
        "tokens_before": before,
        "tokens_after": result.compacted_tokens_est,
        "lossless_exact": False,  # message merge/order differs from packed; use phrase restore
        "meta": {
            "mode": result.mode,
            "n_cold_refs": len(result.cold_refs),
            "cold_kinds": [c.kind for c in result.cold_refs],
            "fits_budget": result.metrics.get("fits_budget"),
            "compression_factor": result.metrics.get("compression_factor"),
            "budget": budget,
        },
    }


def condition_lossy_truncate(
    fixture: dict[str, Any],
    *,
    budget: int,
    **_: Any,
) -> dict[str, Any]:
    """Naive memory baseline: keep first/last messages + head/tail char trim to budget.

    Drops the middle of the conversation (where mid-trace constraints live), then
    trims characters if still over budget. This is intentionally lossy.
    """
    msgs = list(fixture["messages"])
    text = fixture["packed"]
    before = estimate_tokens(text)

    # Keep system + first user + last two turns; drop middle (tool dumps + mid constraints)
    if len(msgs) <= 4:
        kept = msgs
    else:
        kept = [msgs[0], msgs[1]] + msgs[-2:]
    truncated_msgs = kept
    truncated = _pack(truncated_msgs)

    # Further char trim if still over budget
    if estimate_tokens(truncated) > budget:
        target_chars = max(200, budget * 4)
        head = int(target_chars * 0.5)
        tail = int(target_chars * 0.4)
        if head + tail < len(truncated):
            truncated = (
                truncated[:head]
                + f"\n\n…[TRUNCATED {len(truncated) - head - tail} chars]…\n\n"
                + truncated[-tail:]
            )

    after = estimate_tokens(truncated)
    return {
        "condition": "lossy_truncate",
        "visible_text": truncated,
        "restored_text": truncated,
        "tokens_before": before,
        "tokens_after": after,
        "lossless_exact": truncated == text,
        "meta": {
            "budget": budget,
            "truncated": True,
            "policy": "keep_first2_last2_then_char_trim",
            "n_messages_kept": len(truncated_msgs),
        },
    }


# ---------------------------------------------------------------------------
# Minimal experiment-local compact (documented fallback / ablation)
# ---------------------------------------------------------------------------

_HIGH_VALUE_RE = re.compile(
    r"(?i)\b(goal|constraint|must not|cannot|deadline|budget|require|"
    r"outcome|result|objective|aim to|i want|i need)\b"
)


def condition_protect_compact(
    fixture: dict[str, Any],
    *,
    budget: int,
    **_: Any,
) -> dict[str, Any]:
    """Experiment-local compact: keep high-value sentences; dict-compress the rest.

    Documented clearly as experiment-local (not the library default). Used when
    we want an isolate-aware protection policy for comparison.
    """
    text = fixture["packed"]
    before = estimate_tokens(text)
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    keep: list[str] = []
    rest: list[str] = []
    for s in sentences:
        if _HIGH_VALUE_RE.search(s) or s.startswith("<<<SYSTEM") or s.startswith("<<<USER"):
            keep.append(s)
        else:
            rest.append(s)
    rest_blob = "\n".join(rest)
    keep_blob = "\n".join(keep)
    comp = DictCompressor(min_freq=2, max_dict_size=128)
    enc = comp.compress(rest_blob) if rest_blob.strip() else None
    if enc and enc.dictionary:
        visible = (
            keep_blob
            + "\n\n<<<COMPACTED_FILLER dict-encoded>>>\n"
            + enc.packed_prompt
        )
        restored = keep_blob + "\n" + comp.decompress(enc.encoded, enc.dictionary)
        lossless = (keep_blob + "\n" + rest_blob).replace("\n\n", "\n")  # soft
        exact = restored.replace("\n\n", "\n") == (keep_blob + "\n" + rest_blob).replace(
            "\n\n", "\n"
        )
    else:
        visible = keep_blob + ("\n" + rest_blob if rest_blob else "")
        restored = visible
        exact = True
    # If still over budget, trim rest from visible only (keep protected)
    if estimate_tokens(visible) > budget and enc:
        visible = keep_blob + f"\n\n[filler dict cold ~{enc.metrics.packed_tokens} tok omitted from hot set]"
        exact = False
    after = estimate_tokens(visible)
    return {
        "condition": "protect_compact",
        "visible_text": visible,
        "restored_text": restored,
        "tokens_before": before,
        "tokens_after": after,
        "lossless_exact": exact,
        "meta": {
            "budget": budget,
            "n_keep_sentences": len(keep),
            "n_rest_sentences": len(rest),
            "note": "experiment-local: keyword-protect + dict-compress filler",
        },
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_KEY_TYPES = ("goal", "constraint", "outcome")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def phrase_coverage(text: str, gold_keys: dict[str, list[str]]) -> dict[str, Any]:
    """Precision/recall-style coverage of gold key substrings in text."""
    text_n = _norm(text)
    per_type: dict[str, dict[str, float]] = {}
    hits_all = 0
    total_all = 0
    for t in _KEY_TYPES:
        keys = gold_keys.get(t, [])
        hit = sum(1 for k in keys if _norm(k) in text_n)
        total = len(keys)
        hits_all += hit
        total_all += total
        recall = hit / total if total else 1.0
        # Precision proxy: fraction of type keys found (no false-positive model)
        per_type[t] = {"hit": hit, "total": total, "recall": round(recall, 4)}
    return {
        "per_type": per_type,
        "micro_recall": round(hits_all / total_all, 4) if total_all else 1.0,
        "hits": hits_all,
        "total": total_all,
    }


def keyword_typology_proxy(text: str) -> dict[str, set[str]]:
    """Fallback isolate tags when intentisolates is missing."""
    cues = {
        "goal": re.compile(
            r"(?i)\b(goal|i want|i need|aim to|objective|intend|purpose)\b.{0,80}"
        ),
        "constraint": re.compile(
            r"(?i)\b(cannot|can't|must not|constraint|budget|deadline|require|only if|unless)\b.{0,80}"
        ),
        "outcome": re.compile(
            r"(?i)\b(outcome|result|therefore|consequently|yields|leads to)\b.{0,80}"
        ),
    }
    found: dict[str, set[str]] = {k: set() for k in cues}
    for typ, rx in cues.items():
        for m in rx.finditer(text):
            found[typ].add(_norm(m.group(0)[:100]))
    return found


def isolate_pipeline_metrics(
    raw_text: str,
    cond_text: str,
) -> dict[str, Any]:
    """Run identify → motifs → trajectory on raw vs condition text."""
    if _identify_isolates is None or _form_motifs is None or _trajectory_from_motifs is None:
        raw_tags = keyword_typology_proxy(raw_text)
        cond_tags = keyword_typology_proxy(cond_text)
        jaccards = {}
        for t in _KEY_TYPES:
            a, b = raw_tags[t], cond_tags[t]
            if not a and not b:
                jaccards[t] = 1.0
            else:
                jaccards[t] = round(len(a & b) / len(a | b), 4) if (a | b) else 0.0
        return {
            "backend": "keyword_proxy",
            "typology_jaccard": jaccards,
            "mean_typology_jaccard": round(
                sum(jaccards.values()) / max(1, len(jaccards)), 4
            ),
            "motif_jaccard": None,
            "layer_path_match": None,
            "layer_path_raw": None,
            "layer_path_cond": None,
            "n_isolates_raw": sum(len(v) for v in raw_tags.values()),
            "n_isolates_cond": sum(len(v) for v in cond_tags.values()),
            "n_motifs_raw": None,
            "n_motifs_cond": None,
        }

    raw_iso = _identify_isolates(text=raw_text, backend="rule")
    cond_iso = _identify_isolates(text=cond_text, backend="rule")
    raw_mot = _form_motifs(raw_iso)
    cond_mot = _form_motifs(cond_iso)
    raw_traj = _trajectory_from_motifs(raw_mot, raw_iso)
    cond_traj = _trajectory_from_motifs(cond_mot, cond_iso)

    def _typ_set(isos: Sequence[Any], label: str) -> set[str]:
        out: set[str] = set()
        for iso in isos:
            typ = getattr(iso, "typology", None)
            typ_v = typ.value if hasattr(typ, "value") else str(typ)
            if typ_v == label:
                out.add(_norm(str(getattr(iso, "label", ""))[:120]))
        return out

    jaccards = {}
    for t in _KEY_TYPES:
        a, b = _typ_set(raw_iso, t), _typ_set(cond_iso, t)
        if not a and not b:
            jaccards[t] = 1.0
        else:
            jaccards[t] = round(len(a & b) / len(a | b), 4) if (a | b) else 0.0

    def _motif_sig(m: Any) -> str:
        mid = getattr(m, "id", None) or ""
        pat = getattr(m, "pattern", "") or ""
        members = tuple(sorted(getattr(m, "member_ids", []) or []))
        return f"{mid}|{pat}|{members}"

    raw_mset = {_motif_sig(m) for m in raw_mot}
    cond_mset = {_motif_sig(m) for m in cond_mot}
    if not raw_mset and not cond_mset:
        motif_j = 1.0
    else:
        motif_j = (
            round(len(raw_mset & cond_mset) / len(raw_mset | cond_mset), 4)
            if (raw_mset | cond_mset)
            else 0.0
        )

    raw_path = [str(x) for x in (getattr(raw_traj, "layer_path", None) or [])]
    cond_path = [str(x) for x in (getattr(cond_traj, "layer_path", None) or [])]
    if not raw_path and not cond_path:
        path_match = 1.0
    elif not raw_path:
        path_match = 0.0
    else:
        # Completeness: fraction of raw layers still present (order-insensitive soft match)
        path_match = round(len(set(raw_path) & set(cond_path)) / len(set(raw_path)), 4)

    return {
        "backend": _ISOLATES_NOTE,
        "typology_jaccard": jaccards,
        "mean_typology_jaccard": round(
            sum(jaccards.values()) / max(1, len(jaccards)), 4
        ),
        "motif_jaccard": motif_j,
        "layer_path_match": path_match,
        "layer_path_raw": raw_path,
        "layer_path_cond": cond_path,
        "n_isolates_raw": len(raw_iso),
        "n_isolates_cond": len(cond_iso),
        "n_motifs_raw": len(raw_mot),
        "n_motifs_cond": len(cond_mot),
    }


def typology_pr_vs_gold(
    text: str,
    gold_keys: dict[str, list[str]],
) -> dict[str, Any]:
    """If isolates available, P/R of identified typology labels against gold key presence.

    Gold is phrase-based: a predicted isolate of type T is a TP if any gold key for T
    overlaps the isolate label; FN = gold keys with no overlapping isolate of that type.
    """
    if _identify_isolates is None:
        # Fall back to phrase coverage as recall; precision = same under proxy
        cov = phrase_coverage(text, gold_keys)
        return {
            "mode": "phrase_proxy",
            "micro_recall": cov["micro_recall"],
            "micro_precision": cov["micro_recall"],
            "per_type": cov["per_type"],
        }

    isos = _identify_isolates(text=text, backend="rule")
    by_type: dict[str, list[str]] = {t: [] for t in _KEY_TYPES}
    for iso in isos:
        typ = getattr(iso, "typology", None)
        typ_v = typ.value if hasattr(typ, "value") else str(typ)
        if typ_v in by_type:
            by_type[typ_v].append(_norm(str(getattr(iso, "label", ""))))

    per_type: dict[str, Any] = {}
    tp_all = fp_all = fn_all = 0
    for t in _KEY_TYPES:
        gold = [_norm(g) for g in gold_keys.get(t, [])]
        preds = by_type[t]
        tp = 0
        matched_gold = set()
        matched_pred = set()
        for i, p in enumerate(preds):
            for j, g in enumerate(gold):
                if g in p or p in g or (len(g) > 12 and g[:20] in p):
                    tp += 1
                    matched_gold.add(j)
                    matched_pred.add(i)
                    break
        # Count unique TPs more carefully
        tp_u = len(matched_gold)
        fp = max(0, len(preds) - len(matched_pred))
        fn = max(0, len(gold) - len(matched_gold))
        prec = tp_u / (tp_u + fp) if (tp_u + fp) else (1.0 if not preds else 0.0)
        rec = tp_u / (tp_u + fn) if (tp_u + fn) else 1.0
        per_type[t] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "tp": tp_u,
            "fp": fp,
            "fn": fn,
        }
        tp_all += tp_u
        fp_all += fp
        fn_all += fn
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 1.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 1.0
    return {
        "mode": "isolates",
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "per_type": per_type,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class Row:
    fixture_id: str
    condition: str
    tokens_before: int
    tokens_after: int
    token_ratio: float
    lossless_exact: bool
    gold_phrase_recall_visible: float
    gold_phrase_recall_restored: float
    mid_constraint_recall_visible: float
    isolate_micro_recall_visible: float
    isolate_micro_precision_visible: float
    mean_typology_jaccard: Optional[float]
    motif_jaccard: Optional[float]
    layer_path_match: Optional[float]
    meta: dict[str, Any] = field(default_factory=dict)


def run_experiment(*, budget: int, out_dir: Path) -> dict[str, Any]:
    fixtures = build_fixtures()
    conditions: list[tuple[str, Callable[..., dict[str, Any]]]] = [
        ("raw", condition_raw),
        ("compress", condition_compress),
        ("compact", condition_compact),
        ("lossy_truncate", condition_lossy_truncate),
        ("protect_compact", condition_protect_compact),
    ]

    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []

    for fix in fixtures:
        raw_text = fix["packed"]
        # Run compact first so truncate can match its realized token budget
        compact_first = condition_compact(fix, budget=budget)
        truncate_budget = max(200, int(compact_first["tokens_after"]))

        for name, fn in conditions:
            if name == "compact":
                cond = compact_first
            elif name == "lossy_truncate":
                cond = fn(fix, budget=truncate_budget)
            elif name == "protect_compact":
                cond = fn(fix, budget=truncate_budget)
            else:
                cond = fn(fix, budget=budget)
            visible = cond["visible_text"]
            restored = cond["restored_text"]
            gold = fix["gold_keys"]

            phrase_vis = phrase_coverage(visible, gold)
            phrase_res = phrase_coverage(restored, gold)
            # Structure metrics:
            # - For lossless compress, score *restored* (encoded visible is not human/isolate text).
            # - For compact/truncate/protect, score *visible* hot working set (what the model sees).
            structure_text = restored if name == "compress" else visible
            pipe_struct = isolate_pipeline_metrics(raw_text, structure_text)
            pipe_vis = isolate_pipeline_metrics(raw_text, visible)
            pipe_res = isolate_pipeline_metrics(raw_text, restored)
            iso_pr = typology_pr_vs_gold(structure_text, gold)

            # Mid-constraint stress: gold constraint keys only (buried mid-trace)
            mid_gold = {"goal": [], "constraint": gold.get("constraint", []), "outcome": []}
            mid_vis = phrase_coverage(visible, mid_gold)
            mid_res = phrase_coverage(restored, mid_gold)

            tb, ta = cond["tokens_before"], cond["tokens_after"]
            row = Row(
                fixture_id=fix["id"],
                condition=name,
                tokens_before=tb,
                tokens_after=ta,
                token_ratio=round(tb / ta, 4) if ta else 0.0,
                lossless_exact=bool(cond["lossless_exact"]),
                gold_phrase_recall_visible=phrase_vis["micro_recall"],
                gold_phrase_recall_restored=phrase_res["micro_recall"],
                mid_constraint_recall_visible=mid_vis["micro_recall"],
                isolate_micro_recall_visible=float(iso_pr["micro_recall"]),
                isolate_micro_precision_visible=float(iso_pr["micro_precision"]),
                mean_typology_jaccard=pipe_struct.get("mean_typology_jaccard"),
                motif_jaccard=pipe_struct.get("motif_jaccard"),
                layer_path_match=pipe_struct.get("layer_path_match"),
                meta={
                    **cond.get("meta", {}),
                    "pipeline_structure": pipe_struct,
                    "pipeline_visible": pipe_vis,
                    "pipeline_restored": pipe_res,
                    "phrase_visible": phrase_vis,
                    "phrase_restored": phrase_res,
                    "mid_constraint_recall_visible": mid_vis["micro_recall"],
                    "mid_constraint_recall_restored": mid_res["micro_recall"],
                    "isolate_pr_structure": iso_pr,
                    "matched_compact_tokens": truncate_budget,
                    "structure_text_source": "restored" if name == "compress" else "visible",
                },
            )
            rows.append(asdict(row))
            detailed.append(
                {
                    "fixture_id": fix["id"],
                    "condition": name,
                    "tokens_before": tb,
                    "tokens_after": ta,
                    "lossless_exact": cond["lossless_exact"],
                    "meta": cond.get("meta", {}),
                }
            )

    # Aggregate by condition
    by_cond: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    def _avg(vals: list[Optional[float]]) -> Optional[float]:
        nums = [v for v in vals if v is not None]
        if not nums:
            return None
        return round(sum(nums) / len(nums), 4)

    summary_table: list[dict[str, Any]] = []
    for cond, rs in by_cond.items():
        summary_table.append(
            {
                "condition": cond,
                "n": len(rs),
                "avg_tokens_before": _avg([r["tokens_before"] for r in rs]),
                "avg_tokens_after": _avg([r["tokens_after"] for r in rs]),
                "avg_token_ratio": _avg([r["token_ratio"] for r in rs]),
                "lossless_exact_rate": _avg(
                    [1.0 if r["lossless_exact"] else 0.0 for r in rs]
                ),
                "avg_gold_phrase_recall_visible": _avg(
                    [r["gold_phrase_recall_visible"] for r in rs]
                ),
                "avg_gold_phrase_recall_restored": _avg(
                    [r["gold_phrase_recall_restored"] for r in rs]
                ),
                "avg_mid_constraint_recall_visible": _avg(
                    [r["mid_constraint_recall_visible"] for r in rs]
                ),
                "avg_isolate_micro_recall_visible": _avg(
                    [r["isolate_micro_recall_visible"] for r in rs]
                ),
                "avg_isolate_micro_precision_visible": _avg(
                    [r["isolate_micro_precision_visible"] for r in rs]
                ),
                "avg_typology_jaccard": _avg(
                    [r["mean_typology_jaccard"] for r in rs]
                ),
                "avg_motif_jaccard": _avg([r["motif_jaccard"] for r in rs]),
                "avg_layer_path_match": _avg([r["layer_path_match"] for r in rs]),
            }
        )

    # Verdict heuristics
    sm = {s["condition"]: s for s in summary_table}
    compact_s = sm.get("compact", {})
    trunc_s = sm.get("lossy_truncate", {})
    compress_s = sm.get("compress", {})
    protect_s = sm.get("protect_compact", {})

    def _g(d: dict, k: str, default: float = 0.0) -> float:
        v = d.get(k)
        return float(v) if v is not None else default

    verdict_bits = []
    if _g(compact_s, "avg_mid_constraint_recall_visible") > _g(
        trunc_s, "avg_mid_constraint_recall_visible"
    ) + 0.05:
        verdict_bits.append(
            "Compaction preserved mid-trace constraints better than naive truncate "
            "at matched budgets (H1/H3 supported)."
        )
    elif _g(trunc_s, "avg_mid_constraint_recall_visible") > _g(
        compact_s, "avg_mid_constraint_recall_visible"
    ) + 0.05:
        verdict_bits.append(
            "Naive truncate retained more mid-trace constraints than library compact — "
            "check last-N protection vs stub eviction of mid user turns."
        )
    else:
        verdict_bits.append(
            "Compact vs truncate mid-constraint recall was similar; inspect motif/layer metrics."
        )

    if _g(compress_s, "lossless_exact_rate") >= 0.99:
        verdict_bits.append(
            "Lossless dict compress achieved exact round-trip (H2 supported for restore)."
        )
    else:
        verdict_bits.append(
            "Lossless compress round-trip was imperfect on some fixtures — investigate mining."
        )

    if _g(compress_s, "avg_motif_jaccard") is not None and _g(
        compress_s, "avg_motif_jaccard"
    ) >= 0.9:
        verdict_bits.append(
            "After decompress, motif Jaccard vs raw is high — lossless path preserves "
            "reasoning-structure metrics (H2)."
        )
    elif _g(compress_s, "avg_typology_jaccard") >= 0.9:
        verdict_bits.append(
            "Compress restore preserves typology Jaccard even if motif ids churn."
        )

    if _g(compress_s, "avg_gold_phrase_recall_restored") >= 0.99 and _g(
        compress_s, "avg_token_ratio"
    ) > 1.05:
        verdict_bits.append(
            "Compress beats compact for archival fidelity: full restore + token savings "
            "when patterns mine well."
        )

    if protect_s and _g(protect_s, "avg_mid_constraint_recall_visible") >= _g(
        compact_s, "avg_mid_constraint_recall_visible"
    ):
        verdict_bits.append(
            "Isolate-aware protect_compact matched or beat library compact on mid-constraint "
            "recall — prefer isolate-then-compact for reasoning traces."
        )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "isolates_backend": _ISOLATES_NOTE,
        "promptdict_src": str(_SRC),
        "n_fixtures": len(fixtures),
        "conditions": [c[0] for c in conditions],
        "summary_table": summary_table,
        "rows": rows,
        "verdict": verdict_bits,
        "hypotheses": {
            "H1_distractors": "Compaction reduces tokens while keeping goal/constraint/outcome.",
            "H2_lossless_motifs": "Dict compress exact restore preserves structure.",
            "H3_lossy_hurts": "Truncate/stub without protection drops constraints.",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"reasoning_compaction_{stamp}.json"
    md_path = out_dir / f"reasoning_compaction_{stamp}.md"
    latest_json = out_dir / "reasoning_compaction_latest.json"
    latest_md = out_dir / "reasoning_compaction_latest.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = render_markdown(payload)
    md_path.write_text(md, encoding="utf-8")
    latest_md.write_text(md, encoding="utf-8")

    # Also append conclusions into research doc if present
    research_doc = _RESEARCH / "docs" / "ISOLATES_COMPACTION_REASONING.md"
    if research_doc.is_file():
        update_research_doc(research_doc, payload)

    payload["paths"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
    }
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reasoning-trace compaction experiment results",
        "",
        f"- Created: `{payload['created_at']}`",
        f"- Budget (compact/truncate): `{payload['budget']}` tokens",
        f"- Isolates backend: `{payload['isolates_backend']}`",
        f"- Fixtures: `{payload['n_fixtures']}`",
        "",
        "## Summary table (averages)",
        "",
        "| condition | tok_before | tok_after | ratio | lossless | gold_R_vis | mid_R_vis | gold_R_rest | iso_R | typ_J | motif_J | layer_match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in payload["summary_table"]:

        def fmt(x: Any) -> str:
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)

        lines.append(
            "| {condition} | {avg_tokens_before} | {avg_tokens_after} | {avg_token_ratio} | "
            "{lossless_exact_rate} | {avg_gold_phrase_recall_visible} | "
            "{avg_mid_constraint_recall_visible} | {avg_gold_phrase_recall_restored} | "
            "{avg_isolate_micro_recall_visible} | {avg_typology_jaccard} | "
            "{avg_motif_jaccard} | {avg_layer_path_match} |".format(
                condition=s["condition"],
                avg_tokens_before=fmt(s["avg_tokens_before"]),
                avg_tokens_after=fmt(s["avg_tokens_after"]),
                avg_token_ratio=fmt(s["avg_token_ratio"]),
                lossless_exact_rate=fmt(s["lossless_exact_rate"]),
                avg_gold_phrase_recall_visible=fmt(s["avg_gold_phrase_recall_visible"]),
                avg_mid_constraint_recall_visible=fmt(
                    s.get("avg_mid_constraint_recall_visible")
                ),
                avg_gold_phrase_recall_restored=fmt(s["avg_gold_phrase_recall_restored"]),
                avg_isolate_micro_recall_visible=fmt(
                    s["avg_isolate_micro_recall_visible"]
                ),
                avg_typology_jaccard=fmt(s["avg_typology_jaccard"]),
                avg_motif_jaccard=fmt(s["avg_motif_jaccard"]),
                avg_layer_path_match=fmt(s["avg_layer_path_match"]),
            )
        )

    lines += ["", "## Verdict", ""]
    for v in payload["verdict"]:
        lines.append(f"- {v}")

    lines += [
        "",
        "## Per-fixture rows",
        "",
        "| fixture | condition | tok_after | ratio | gold_R_vis | mid_R_vis | gold_R_rest | typ_J | motif_J |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload["rows"]:

        def fmt(x: Any) -> str:
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.3f}"
            return str(x)

        lines.append(
            f"| {r['fixture_id']} | {r['condition']} | {r['tokens_after']} | "
            f"{fmt(r['token_ratio'])} | {fmt(r['gold_phrase_recall_visible'])} | "
            f"{fmt(r['mid_constraint_recall_visible'])} | "
            f"{fmt(r['gold_phrase_recall_restored'])} | {fmt(r['mean_typology_jaccard'])} | "
            f"{fmt(r['motif_jaccard'])} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `compress` = lossless `DictCompressor` on full packed trace.",
        "- `compact` = `promptdict.compact_messages` (lossless_dict preferred).",
        "- `lossy_truncate` = head/tail truncate to the same budget.",
        "- `protect_compact` = **experiment-local** keyword-protect + dict-compress filler.",
        "- Core metrics are automatic (no LLM judge).",
        "",
    ]
    return "\n".join(lines) + "\n"


def update_research_doc(path: Path, payload: dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "## 7. Conclusions (filled after run)"
    if marker not in text:
        return
    pre, _sep, post = text.partition(marker)
    # Keep everything after next ## if present in post — replace section body
    rest = post
    # Drop old section until next ## at start of line (## 8.)
    m = re.search(r"\n## 8\.", rest)
    if m:
        tail = rest[m.start() :]
    else:
        tail = ""

    sm = {s["condition"]: s for s in payload["summary_table"]}

    def cell(cond: str, key: str) -> str:
        v = sm.get(cond, {}).get(key)
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    body = f"""
## 7. Conclusions (filled after run)

**Run:** `{payload['created_at']}` · budget=`{payload['budget']}` · backend=`{payload['isolates_backend']}`

### Did compaction improve / preserve structure vs truncate?

| condition | tok_ratio | gold_R_visible | mid_R_visible | gold_R_restored | typ_J | motif_J | layer_match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | {cell('raw','avg_token_ratio')} | {cell('raw','avg_gold_phrase_recall_visible')} | {cell('raw','avg_mid_constraint_recall_visible')} | {cell('raw','avg_gold_phrase_recall_restored')} | {cell('raw','avg_typology_jaccard')} | {cell('raw','avg_motif_jaccard')} | {cell('raw','avg_layer_path_match')} |
| compress | {cell('compress','avg_token_ratio')} | {cell('compress','avg_gold_phrase_recall_visible')} | {cell('compress','avg_mid_constraint_recall_visible')} | {cell('compress','avg_gold_phrase_recall_restored')} | {cell('compress','avg_typology_jaccard')} | {cell('compress','avg_motif_jaccard')} | {cell('compress','avg_layer_path_match')} |
| compact | {cell('compact','avg_token_ratio')} | {cell('compact','avg_gold_phrase_recall_visible')} | {cell('compact','avg_mid_constraint_recall_visible')} | {cell('compact','avg_gold_phrase_recall_restored')} | {cell('compact','avg_typology_jaccard')} | {cell('compact','avg_motif_jaccard')} | {cell('compact','avg_layer_path_match')} |
| lossy_truncate | {cell('lossy_truncate','avg_token_ratio')} | {cell('lossy_truncate','avg_gold_phrase_recall_visible')} | {cell('lossy_truncate','avg_mid_constraint_recall_visible')} | {cell('lossy_truncate','avg_gold_phrase_recall_restored')} | {cell('lossy_truncate','avg_typology_jaccard')} | {cell('lossy_truncate','avg_motif_jaccard')} | {cell('lossy_truncate','avg_layer_path_match')} |
| protect_compact | {cell('protect_compact','avg_token_ratio')} | {cell('protect_compact','avg_gold_phrase_recall_visible')} | {cell('protect_compact','avg_mid_constraint_recall_visible')} | {cell('protect_compact','avg_gold_phrase_recall_restored')} | {cell('protect_compact','avg_typology_jaccard')} | {cell('protect_compact','avg_motif_jaccard')} | {cell('protect_compact','avg_layer_path_match')} |

### Verdict bullets

"""
    for v in payload["verdict"]:
        body += f"- {v}\n"

    body += """
### When compress (lossless) beats compact

- Use **compress** when you need archival / ICL-packed fidelity and the trace is repetitive enough to mine: exact restore + motif recovery.
- Use **compact** for live agent loops under a hard working-set budget; pair with cold_ref expansion on demand.
- Use **protect_compact / isolate-then-compact** when constraints sit in mid-history that default last-N protection might not cover after aggressive eviction.

### Recommendations (LLMIntent + PromptDict)

1. **Isolate then compact:** run `identify_isolates` (goal/constraint/outcome), mark spans as protected, then `compact_messages` / dict-encode only low-value repetitive regions.
2. **Compact then isolate** only for cheap triage on the hot set — do not treat stubbed text as ground-truth reasoning structure.
3. Prefer **lossless_dict** mode over **lossy_stub** whenever tool dumps are recoverable and patterns mine well.
4. Never rely on naive head/tail truncate as reasoning memory.

"""
    path.write_text(pre + body + tail, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--budget",
        type=int,
        default=1200,
        help="Token budget for compact / truncate conditions (default 1200)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "experiments" / "results",
        help="Output directory for JSON + markdown",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    print(f"Isolates backend: {_ISOLATES_NOTE}")
    print(f"Budget: {args.budget}")
    print(f"Out dir: {args.out_dir}")
    payload = run_experiment(budget=args.budget, out_dir=args.out_dir)
    print("\n=== Summary ===")
    for s in payload["summary_table"]:
        print(
            f"{s['condition']:16s}  tok {s['avg_tokens_after']:>6}  "
            f"ratio {s['avg_token_ratio']:>6}  "
            f"gold_R_vis {s['avg_gold_phrase_recall_visible']}  "
            f"mid_R {s.get('avg_mid_constraint_recall_visible')}  "
            f"motif_J {s['avg_motif_jaccard']}"
        )
    print("\nVerdict:")
    for v in payload["verdict"]:
        print(f"  - {v}")
    print(f"\nWrote {payload['paths']['json']}")
    print(f"Wrote {payload['paths']['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
