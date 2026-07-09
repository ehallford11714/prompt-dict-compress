"""Lossless round-trip tests for DictCompressor."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptdict.compressor import DictCompressor

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("name", ["sample.log", "sample.jsonl"])
def test_lossless_roundtrip_fixtures(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    # Amplify repetition so dictionary mining fires
    amplified = (text + "\n") * 5
    comp = DictCompressor(min_freq=2, max_dict_size=128)
    result = comp.compress(amplified)
    decoded = comp.decompress(result.encoded, result.dictionary)
    assert decoded == amplified


def test_identity_when_no_patterns() -> None:
    text = "xyz unique once only"
    comp = DictCompressor(min_freq=3)
    result = comp.compress(text)
    assert comp.decompress(result.encoded, result.dictionary) == text


def test_packed_json_roundtrip(tmp_path: Path) -> None:
    text = ("ERROR timeout retry=true queue=default\n" * 20) + ("INFO status=200 msg=ok\n" * 20)
    comp = DictCompressor(min_freq=2)
    result = comp.compress(text)
    path = tmp_path / "packed.json"
    path.write_text(result.to_json(), encoding="utf-8")
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    enc, d = DictCompressor.from_packed_dict(data)
    assert DictCompressor().decompress(enc, d) == text
