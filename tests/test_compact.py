"""Working-set compaction tests."""

from __future__ import annotations

from promptdict.compact import compact_messages, expand_cold_ref
from promptdict.metrics import estimate_tokens


def _repetitive_tool_body(n: int = 80) -> str:
    # Identical lines so dict+ICL clearly beats raw (seq variation hurts mining).
    line = (
        "tool_result status=200 request_id=abc123 latency_ms=12 "
        '{"ok": true, "items": [1, 2, 3]}'
    )
    return "\n".join(line for _ in range(n))


def test_compact_messages_reduces_tokens_and_cold_refs() -> None:
    body = _repetitive_tool_body(100)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "run the batch"},
        {"role": "assistant", "content": "calling tool"},
        {"role": "tool", "content": body},
        {"role": "user", "content": "thanks, continue"},
        {"role": "assistant", "content": "done"},
    ]
    result = compact_messages(
        messages,
        budget=2_000,
        mode="lossless_dict",
        keep_last_n=2,
        min_freq=2,
    )
    assert result.compacted_tokens_est < result.original_tokens_est
    assert result.cold_refs, "expected cold_refs for repetitive tool-like body"
    ref = result.cold_refs[0]
    assert ref.kind == "dict_encoded"
    restored = expand_cold_ref(ref)
    assert "tool_result" in restored
    assert "request_id=abc123" in restored
    assert estimate_tokens(restored) >= 80


def test_expand_cold_ref_dict_path_roundtrip() -> None:
    body = _repetitive_tool_body(100)
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "tool", "content": body},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    result = compact_messages(messages, budget=1_500, mode="lossless_dict", keep_last_n=2)
    assert result.cold_refs
    for ref in result.cold_refs:
        if ref.kind == "dict_encoded":
            text = expand_cold_ref(ref)
            assert "status=200" in text
            assert text.count("tool_result") >= 1
            break
    else:
        raise AssertionError("no dict_encoded cold_ref")
