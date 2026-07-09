# PromptDictCompress

Lossless (and two-tier scalable) **prompt compression** via dictionary-encoding + in-context learning, with PageIndex-style hierarchical packing toward large corpora.

Inspired by [Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning](https://arxiv.org/abs/2604.13066) (arXiv:2604.13066), extended with hierarchical / streaming scale paths.

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

# Scale demo toward 100M→1M budgets (simulated redundancy sample by default)
promptdict scale-demo --out .scale_demo
```

Python:

```python
from promptdict import DictCompressor
from promptdict.hierarchical import HierarchicalPageIndexCompress
from promptdict.scale import MillionTokenBudgetCompressor

r = DictCompressor().compress(open("logs.txt", encoding="utf-8").read())
print(r.metrics)  # compression_factor, packed_tokens, ...
```

## 100M → 1M

**Verdict:** Strictly lossless packing of *arbitrary* 100M tokens into a *single* 1M API context is **not** information-theoretically plausible. For repetitive corpora, dict+ICL can give strong reductions (published ~2–5× / up to ~80% token reduction — not 100×). The **SOTA-practical** design is:

1. **Hierarchical PageIndex** (navigate without loading all bodies)
2. **Nested dictionary encoding** (local + global/template codebooks)
3. **`prompt_pack` (≤1M) + `cold_store` (lossless on disk)** — full corpus invertibility; agent/tools fetch cold pages

| Doc | Contents |
| --- | --- |
| [docs/SOTA_100M_TO_1M.md](docs/SOTA_100M_TO_1M.md) | Full research brief, citations, ranked methods |
| [docs/SCALE_100M_TO_1M.md](docs/SCALE_100M_TO_1M.md) | Architecture recommendation + code mapping |

**Code path:** `src/promptdict/scale.py` (`MillionTokenBudgetCompressor`). Scale demos that pass `simulated_input_tokens` are **labeled simulations** of redundancy structure, not literal 100M-token files on disk.

## Layout

```
src/promptdict/
  compressor.py      # DictCompressor — flat lossless dict+ICL pack
  hierarchical.py    # PageIndex-style nested dicts (in-memory)
  scale.py           # Streaming 100M-budget path + cold_store
  mining.py          # Pattern mining
  metrics.py         # Token estimates / ratios
  cli.py             # promptdict CLI
docs/
  SOTA_100M_TO_1M.md
  SCALE_100M_TO_1M.md
```

## License

MIT
