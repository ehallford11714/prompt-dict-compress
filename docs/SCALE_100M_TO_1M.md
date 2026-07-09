# Architecture: Scaling Toward 100M → 1M Tokens

**Companion research:** [SOTA_100M_TO_1M.md](./SOTA_100M_TO_1M.md)  
**Code:** `src/promptdict/scale.py`, `hierarchical.py`, `compressor.py`

---

## Goal semantics (read carefully)

| Goal wording | Meaning we adopt |
| --- | --- |
| “Lossless 100M → 1M” | **Invertible** representation of the corpus whose *active API prompt* is ≤1M tokens |
| Strict interpretation A | Entire corpus body lives **only** in the 1M prompt | **Rejected** for general text (entropy) |
| Practical interpretation B | `prompt_pack` ≤1M **plus** `cold_store` on disk; together lossless | **Adopted SOTA-practical** |
| Stretch interpretation C | Highly repetitive corpus; dictionaries + encoded bodies fit in 1M alone | **Possible for logs/templates; unproven at 100× in papers** |

Published dict+ICL work shows ~**2–5×** (60–80% reduction), not 100×. Hitting ~100× *prompt-resident* requires extreme redundancy **or** counting only the hot working set.

---

## Recommended architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW CORPUS (~100M tok)                    │
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
│ prompt_pack (≤ 1M tok)   │        │ cold_store.jsonl (lossless)  │
│ • GLOBAL_DICT            │        │ • all pages encoded          │
│ • TEMPLATE_CODEBOOK      │        │ • local dictionaries         │
│ • PAGE_INDEX (slim)      │        │ • exact decode per page_id   │
│ • HOT_ENCODED_PAGES      │        └──────────────────────────────┘
│ • COLD_STORE_REF         │                        ▲
└────────────┬─────────────┘                        │
             │                                      │
             ▼                                      │
┌──────────────────────────┐                        │
│ API LLM (ICL over dict)  │── tool: fetch page ────┘
│ analyze / QA / aggregate │
└──────────────────────────┘
```

### Why this is the right default

1. **API-portable** — only text tokens; no custom KV stack.
2. **Lossless where it matters** — cold_store + dictionaries invert to original strings (`decompress_page` / `decompress_all`).
3. **Matches SOTA practice** — PageIndex-style navigation + Anthropic “filesystem as memory” + dict+ICL for repetitive bodies.
4. **Honest about 100×** — compression factor vs *prompt* can look huge when most pages are cold; factor vs *full materialization in prompt* only for extreme redundancy.

---

## Component mapping (code)

| Layer | Module | Notes |
| --- | --- | --- |
| L0 local encode | `compressor.DictCompressor` | Meta-tokens `⟦A⟧…`; token-savings gate; round-trip check |
| In-memory hierarchy | `hierarchical.HierarchicalPageIndexCompress` | Full packed prompt for smaller corpora |
| 100M path | `scale.MillionTokenBudgetCompressor` | Streaming; writes `prompt_pack.txt`, `cold_store.jsonl`, `page_index.json` |
| Demo | `scale.run_scale_demo` | `--simulate` uses redundancy sample + **labeled** target_in |

### Outputs of a scale run

```
out_dir/
  prompt_pack.txt      # ≤ output_token_budget (target 1M)
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
| ≤ few 100k tokens, repetitive | Flat `DictCompressor` |
| Multi-doc / multi-page, still fits RAM | `HierarchicalPageIndexCompress` |
| ~1M–100M+, need API budget | `MillionTokenBudgetCompressor` + tools |
| Query needs subset only | PageIndex navigate → fetch cold pages → optional LongLLMLingua on fetched text (**lossy OK**) |
| Session chat history | Compaction / memory agents (**lossy**; separate from corpus archive) |

---

## Optional lossy overlays (do not mix into “lossless” claims)

- **LongLLMLingua / Selective Context** on retrieved pages for cheaper QA.
- **Summaries** on PageIndex nodes for search (keep verbatim bodies in cold_store).
- **KV compression** only if you self-host inference.

---

## Success metrics

1. **Round-trip:** `decompress_all(out_dir) == original` (or per-page equality).
2. **Budget:** `prompt_tokens_est ≤ 1_000_000`.
3. **Factor reporting:** always state denominator:
   - `input_tokens / prompt_tokens` (two-tier; can be ≫100 with mostly cold pages)
   - `input_tokens / packed_tokens_if_all_hot` (true prompt-resident; usually ≪100)
4. **Task fidelity:** for analytics, prefer task metrics over decompression-only (paper used decompress as proxy).

---

## Non-goals

- Implementing Infini-attention or SnapKV inside this library.
- Claiming universal 100× lossless prompt-only compression.
- Putting zstd bitstreams in the prompt for the model to decode without tools.
