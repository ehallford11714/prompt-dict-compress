# PromptDict suite: Compression vs Compaction vs Recall

**Package:** `promptdict`  
**Facade:** `PromptMemorySuite`

This library is a small **prompt-memory suite** with three pillars. They solve different problems and should not be conflated.

| Pillar | Verb | What it does | Lossless? |
| --- | --- | --- | --- |
| **Compression** | *encode* | Dictionary-encode text / hierarchical PageIndex pack for API prompts | Yes (round-trip) |
| **Compaction** | *manage* | Shrink an agent message working set over time under a budget | Optional lossless dict path **or** lossy stubs |
| **Recall** | *restore* | Pull pages back from `cold_store` / page index into context | Yes (from store) |

---

## Compression (`promptdict.compress`)

**Job:** Turn a corpus (or blob) into a shorter, API-portable packed prompt.

- Flat: `DictCompressor` / `compress_text` — mine patterns → meta-tokens → `DICTIONARY` + `ENCODED_BODY`.
- Hierarchical: `HierarchicalPageIndexCompress` — pages + local/global dicts in memory.
- Budgeted / ultra-long: `BudgetedContextCompressor` — stream pages → `prompt_pack` + `cold_store`.

Published dict+ICL results are typically **~2–5×** on repetitive logs/JSON. Extreme ratios are data-dependent; full corpus invertibility for ultra-long inputs uses the two-tier store.

```python
from promptdict import compress_text, PromptMemorySuite

r = compress_text(open("logs.txt", encoding="utf-8").read())
# or
suite = PromptMemorySuite(output_budget=1_000_000)
packed = suite.compress(text, mode="flat")
```

CLI: `python -m promptdict compress -f logs.txt -o packed.json`

---

## Compaction (`promptdict.compact`)

**Job:** Keep an **agent loop** under a token budget as turns accumulate.

- Merge adjacent same-role turns.
- Drop / stub recoverable redundancy (tool dumps, repetitive logs).
- Optional **lossless dict path**: bulky turns become dict-encoded packs with `cold_refs` for exact restore.
- **Lossy stub path**: short previews when dict savings are weak or budget forces eviction.

```python
from promptdict import compact_messages

result = compact_messages(messages, budget=8_000, mode="auto")
# result.messages, result.packed_prompt, result.cold_refs
```

CLI: `python -m promptdict compact -f messages.json -o compacted.json --budget 8000`

**Difference from compression:** compression encodes a corpus once; compaction continuously manages a *working set* across turns.

---

## Recall (`promptdict.recall`)

**Job:** Losslessly restore content that left the active window into `cold_store` (or was never hot).

- By `page_id` / `page_ids`
- By simple **keyword** query over decoded pages (MVP; no vector DB)
- Embedding recall is an explicit **stub** (`method="embedding"` → `NotImplementedError`)

```python
from promptdict import recall

restored = recall(store=".scale_demo", page_ids=[0, 2])
# or
hits = recall(store=".scale_demo", query="rate_limited", top_k=3)
print(hits.packed_fragment)
```

CLI: `python -m promptdict recall -s .scale_demo --page-id 0`  
CLI: `python -m promptdict recall -s .scale_demo -q timeout --top-k 3`

---

## Suite facade

```python
from promptdict import PromptMemorySuite

suite = PromptMemorySuite(output_budget=1_000_000)
packed = suite.compress(text)                    # encode
compacted = suite.compact(messages)              # working set
restored = suite.recall(page_id=0, store=".scale_demo")  # restore
```

CLI:

```bash
python -m promptdict compress ...
python -m promptdict compact ...
python -m promptdict recall ...
python -m promptdict suite --help
python -m promptdict suite --demo
```

---

## Related docs

- [SCALE_ULTRA_LONG.md](./SCALE_ULTRA_LONG.md) — two-tier budgeted architecture
- [SOTA_ULTRA_LONG_CONTEXT.md](./SOTA_ULTRA_LONG_CONTEXT.md) — research brief
- [SOTA_LOSSLESS_PROMPT_COMPRESSION.md](./SOTA_LOSSLESS_PROMPT_COMPRESSION.md) — dict+ICL paper mapping
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — module map
