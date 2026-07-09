# Architecture: Ultra-Long Corpora into a Fixed Context Budget

**Companion research:** [SOTA_ULTRA_LONG_CONTEXT.md](./SOTA_ULTRA_LONG_CONTEXT.md)  
**Suite overview:** [SUITE.md](./SUITE.md)  
**Code:** `src/promptdict/scale.py`, `hierarchical.py`, `compressor.py`, `recall.py`

---

## Goal semantics (read carefully)

| Goal wording | Meaning we adopt |
| --- | --- |
| “Ultra-long → fixed budget” | **Invertible** representation of the corpus whose *active API prompt* is ≤ `output_budget` tokens |
| Strict interpretation A | Entire corpus body lives **only** in the prompt | **Rejected** for general text (entropy) |
| Practical interpretation B | `prompt_pack` ≤ `output_budget` **plus** `cold_store` on disk; together lossless | **Adopted SOTA-practical** |
| Stretch interpretation C | Highly repetitive corpus; dictionaries + encoded bodies fit in the prompt alone | **Possible for logs/templates; extreme ratios unproven in papers** |

Published dict+ICL work shows ~**2–5×** (60–80% reduction). Extreme *prompt-resident* ratios require extreme redundancy **or** counting only the hot working set (cold pages live on disk).

---

## Recommended architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              RAW CORPUS (ultra-long / input_budget)              │
└───────────────────────────────┬─────────────────────────────────┘
                                │ stream by page
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L0  Local dict-encode (per page)     lossless string coding      │
│     DictCompressor / page local_dictionary                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L1  Global / template codebook       cross-page repetition       │
│     template_id clustering + GLOBAL_DICT                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2  PageIndex directory              navigation without bodies   │
│     page_id, fingerprint, template_id, in_prompt flag            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
┌──────────────────────────┐        ┌──────────────────────────────┐
│ prompt_pack (≤ budget)   │        │ cold_store.jsonl (lossless)  │
│ • GLOBAL_DICT            │        │ • all pages encoded          │
│ • TEMPLATE_CODEBOOK      │        │ • local dictionaries         │
│ • PAGE_INDEX (slim)      │        │ • exact decode per page_id   │
│ • HOT_ENCODED_PAGES      │        └──────────────────────────────┘
│ • COLD_STORE_REF         │                        ▲
└────────────┬─────────────┘                        │
             │                                      │
             ▼                                      │
┌──────────────────────────┐                        │
│ API LLM (ICL over dict)  │── recall / tool fetch ─┘
│ analyze / QA / aggregate │
└──────────────────────────┘
```

### Why this is the right default

1. **API-portable** — only text tokens; no custom KV stack.
2. **Lossless where it matters** — cold_store + dictionaries invert to original strings (`decompress_page` / `recall`).
3. **Matches SOTA practice** — PageIndex-style navigation + Anthropic “filesystem as memory” + dict+ICL for repetitive bodies.
4. **Honest about ratios** — compression factor vs *prompt* can look huge when most pages are cold; true prompt-resident factors track published ~2–5× unless data is extremely repetitive.

---

## Component mapping (code)

| Layer | Module | Notes |
| --- | --- | --- |
| L0 local encode | `compressor.DictCompressor` / `promptdict.compress` | Meta-tokens `⟦A⟧…`; token-savings gate; round-trip check |
| In-memory hierarchy | `hierarchical.HierarchicalPageIndexCompress` | Full packed prompt for smaller corpora |
| Budgeted streaming | `scale.BudgetedContextCompressor` | Streaming; writes `prompt_pack.txt`, `cold_store.jsonl`, `page_index.json` |
| Recall | `recall.recall` / `ColdStore` | Restore by `page_id` or keyword query |
| Demo | `scale.run_scale_demo` | `--simulate` uses redundancy sample + **labeled** `input_budget` |

### Outputs of a scale run

```
out_dir/
  prompt_pack.txt      # ≤ output_token_budget
  cold_store.jsonl     # one JSON object per page (encoded + local_dictionary)
  page_index.json      # full directory + global_dictionary
  scale_meta.json      # metrics + honesty flags
```

`ScaleCompressResult.semantics` explicitly states:

- Lossless scope = **prompt_pack + cold_store**
- Prompt alone = addressable compressed view; may omit cold page bodies

---

## When to use which mode

| Corpus size / type | Mode |
| --- | --- |
| ≤ few 100k tokens, repetitive | Flat `DictCompressor` / `suite.compress(mode="flat")` |
| Multi-doc / multi-page, still fits RAM | `HierarchicalPageIndexCompress` |
| Ultra-long, need API budget | `BudgetedContextCompressor` + `recall` |
| Query needs subset only | PageIndex navigate → `recall(page_ids|query)` → optional LongLLMLingua on fetched text (**lossy OK**) |
| Session chat history | `compact_messages` / `suite.compact` (working-set management) |

---

## Optional lossy overlays (do not mix into “lossless” claims)

- **LongLLMLingua / Selective Context** on retrieved pages for cheaper QA.
- **Summaries** on PageIndex nodes for search (keep verbatim bodies in cold_store).
- **KV compression** only if you self-host inference.

---

## Success metrics

1. **Round-trip:** `decompress_all(out_dir) == original` (or per-page equality via `recall`).
2. **Budget:** `prompt_tokens_est ≤ output_token_budget`.
3. **Factor reporting:** always state denominator:
   - `input_tokens / prompt_tokens` (two-tier; can be large with mostly cold pages)
   - `input_tokens / packed_tokens_if_all_hot` (true prompt-resident; usually ~2–5× class)
4. **Task fidelity:** for analytics, prefer task metrics over decompression-only (paper used decompress as proxy).

---

## Non-goals

- Implementing Infini-attention or SnapKV inside this library.
- Claiming universal extreme-ratio lossless prompt-only compression.
- Putting zstd bitstreams in the prompt for the model to decode without tools.
