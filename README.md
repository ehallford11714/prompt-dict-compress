# PromptDictCompress

PromptDictCompress (promptdict) is a small LLM prompt-memory suite: lossless dictionary-encoding compression, hierarchical PageIndex packing, working-set compaction for agent loops, and cold_store recall — packing ultra-long repetitive corpora into a fixed context budget via prompt_pack + cold_store. Published dict+ICL ~2–5×; extreme ratios only on repetitive data; full lossless needs cold_store.

Inspired by [Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning](https://arxiv.org/abs/2604.13066) (arXiv:2604.13066), extended with hierarchical / streaming scale paths and a compress / compact / recall suite.

## Install

`ash
pip install -e ".[dev]"
`

## Quick start

`ash
# Flat dictionary compress
promptdict compress path/to/logs.txt -o packed.json

# Hierarchical PageIndex-style pack (in-memory)
promptdict hierarchical path/to/corpus.txt -o hierarchical_packed.json

# Budgeted ultra-long demo (simulated redundancy sample by default)
promptdict scale-demo --out .scale_demo
`

Python:

`python
from promptdict import DictCompressor, PromptMemorySuite, BudgetedContextCompressor
from promptdict.hierarchical import HierarchicalPageIndexCompress

r = DictCompressor().compress(open("logs.txt", encoding="utf-8").read())
print(r.metrics)  # compression_factor, packed_tokens, ...

suite = PromptMemorySuite(output_budget=1_000_000)
`

## Ultra-long context → fixed budget

**Verdict:** Strictly lossless packing of *arbitrary* ultra-long inputs into a *single* fixed API context is **not** information-theoretically plausible. For repetitive corpora, dict+ICL can give strong reductions (published ~2–5× / up to ~80% token reduction — not extreme universal ratios). The **SOTA-practical** design is:

1. **Hierarchical PageIndex** (navigate without loading all bodies)
2. **Nested dictionary encoding** (local + global/template codebooks)
3. **prompt_pack (≤ output_budget) + cold_store (lossless on disk)** — full corpus invertibility; agent/tools fetch cold pages

| Doc | Contents |
| --- | --- |
| [docs/SOTA_ULTRA_LONG_CONTEXT.md](docs/SOTA_ULTRA_LONG_CONTEXT.md) | Full research brief, citations, ranked methods |
| [docs/SCALE_ULTRA_LONG.md](docs/SCALE_ULTRA_LONG.md) | Architecture recommendation + code mapping |
| [docs/SUITE.md](docs/SUITE.md) | Compress / compact / recall pillars |

**Code path:** src/promptdict/scale.py (BudgetedContextCompressor; alias MillionTokenBudgetCompressor still exists). Scale demos that pass simulated_input_tokens are **labeled simulations** of redundancy structure, not literal ultra-long files on disk.

## Layout

`
src/promptdict/
  compressor.py      # DictCompressor — flat lossless dict+ICL pack
  hierarchical.py    # PageIndex-style nested dicts (in-memory)
  scale.py           # Streaming budgeted path + cold_store
  compact.py         # Working-set compaction for agent loops
  recall.py          # cold_store / page_id / keyword recall
  suite.py           # PromptMemorySuite facade
  mining.py          # Pattern mining
  metrics.py         # Token estimates / ratios
  cli.py             # promptdict CLI
docs/
  SOTA_ULTRA_LONG_CONTEXT.md
  SCALE_ULTRA_LONG.md
  SUITE.md
  SOTA_LOSSLESS_PROMPT_COMPRESSION.md
`

## License

MIT
