# PromptDictCompress

PromptDictCompress (`promptdict`) is a small LLM prompt-memory suite: lossless dictionary-encoding compression, hierarchical PageIndex packing, working-set compaction for agent loops, and cold_store recall — packing ultra-long repetitive corpora into a fixed context budget via `prompt_pack` + `cold_store`. Published dict+ICL ~2–5×; extreme ratios only on repetitive data; full lossless needs cold_store.

Inspired by [Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning](https://arxiv.org/abs/2604.13066) (arXiv:2604.13066), extended with hierarchical / streaming scale paths and a compress / compact / recall suite.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```bash
# Flat dictionary compress
promptdict compress path/to/logs.txt -o packed.json

# Hierarchical PageIndex-style pack (in-memory)
promptdict hierarchical path/to/corpus.txt -o hierarchical_packed.json

# Budgeted ultra-long demo (simulated redundancy sample by default)
promptdict scale-demo --out .scale_demo

# Compact agent messages / recall from cold_store
promptdict compact -f messages.json -o compacted.json --budget 8000
promptdict recall -s .scale_demo --page-id 0
promptdict suite --help
```

Python:

```python
from promptdict import PromptMemorySuite

suite = PromptMemorySuite(output_budget=1_000_000)
packed = suite.compress(open("logs.txt", encoding="utf-8").read())
compacted = suite.compact(messages)
restored = suite.recall(page_id=0, store=".scale_demo")
```

## Ultra-long context → fixed budget

**Verdict:** Strictly lossless packing of *arbitrary* ultra-long inputs into a *single* fixed API context is **not** information-theoretically plausible. For repetitive corpora, dict+ICL can give strong reductions (published ~2–5× / up to ~80% token reduction — not extreme universal ratios). The **SOTA-practical** design is:

1. **Hierarchical PageIndex** (navigate without loading all bodies)
2. **Nested dictionary encoding** (local + global/template codebooks)
3. **`prompt_pack` (≤ `output_budget`) + `cold_store` (lossless on disk)** — full corpus invertibility; agent/tools fetch cold pages

| Doc | Contents |
| --- | --- |
| [docs/SUITE.md](docs/SUITE.md) | Compress / compact / recall pillars |
| [docs/SOTA_ULTRA_LONG_CONTEXT.md](docs/SOTA_ULTRA_LONG_CONTEXT.md) | Full research brief, citations, ranked methods |
| [docs/SCALE_ULTRA_LONG.md](docs/SCALE_ULTRA_LONG.md) | Architecture recommendation + code mapping |

**Code path:** `src/promptdict/scale.py` (`BudgetedContextCompressor`; alias `MillionTokenBudgetCompressor` still exists). Scale demos that pass `simulated_input_tokens` are **labeled simulations** of redundancy structure, not literal ultra-long files on disk.

## Layout

```
src/promptdict/
  compress/          # Compression pillar exports
  compact.py         # Working-set compaction for agent loops
  recall.py          # cold_store / page_id / keyword recall
  suite.py           # PromptMemorySuite facade
  compressor.py      # DictCompressor — flat lossless dict+ICL pack
  hierarchical.py    # PageIndex-style nested dicts (in-memory)
  scale.py           # Streaming budgeted path + cold_store
  mining.py          # Pattern mining
  metrics.py         # Token estimates / ratios
  cli.py             # promptdict CLI
docs/
  SUITE.md
  SOTA_ULTRA_LONG_CONTEXT.md
  SCALE_ULTRA_LONG.md
  SOTA_LOSSLESS_PROMPT_COMPRESSION.md
```

## License

MIT
