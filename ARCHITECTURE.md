# PromptDictCompress architecture

## Goals

1. **Lossless** dictionary-encoding of repetitive prompts (logs, JSON, code).
2. Pack `system_dictionary` + `encoded_body` for API LLMs (ICL over the codebook).
3. **Suite pillars:** compress (encode), compact (manage agent working sets), recall (restore from cold_store) — packing ultra-long repetitive corpora into a fixed `output_budget` via hierarchical PageIndex + streaming `prompt_pack` / `cold_store`.

## Core pipeline (flat)

```
text → mine patterns (lines / n-grams) → dictionary {meta→chunk}
    → longest-first replace → encoded body
    → pack prompt → metrics
decode: reverse replace (longest meta first) → original
```

## Hierarchical (in-memory)

```
text → pages
    → Level-0 local dict per page
    → join pages → Level-1 global dict
    → PAGE_INDEX + GLOBAL_DICT + PAGE_n_DICT + ENCODED_CORPUS
```

Decode order: global expand → split pages → local expand → concat.

## Budgeted ultra-long (`BudgetedContextCompressor`)

```
page stream → Level-0 local encode → append cold_store.jsonl
           → template fingerprint counts
           → Level-1/2 global dict + template codebook + slim PAGE_INDEX
           → prompt_pack (≤ output_token_budget): dicts + index + HOT pages
           → cold_store: all encoded pages for full lossless decode
```

**Semantics:** corpus losslessness = `prompt_pack` + `cold_store`. The budgeted prompt is an addressable compressed *view*, not necessarily the entire corpus inline. (`MillionTokenBudgetCompressor` remains as a back-compat alias.)

## Compaction & recall

- **compact:** shrink chat/tool message lists under a token budget (lossless dict path or lossy stubs + `cold_refs`).
- **recall:** restore pages by `page_id` or keyword search over `cold_store`.

## Modules

| Module | Role |
|--------|------|
| `mining.py` | n-gram / line / char pattern mining |
| `compressor.py` | `DictCompressor` encode/decode/pack |
| `compress/` | Compression pillar exports |
| `hierarchical.py` | in-memory PageIndex variant |
| `scale.py` | streaming budgeted path + scale-demo generator |
| `compact.py` | working-set compaction |
| `recall.py` | cold_store recall |
| `suite.py` | `PromptMemorySuite` facade |
| `metrics.py` | tiktoken-or-chars/4 estimates |
| `cli.py` | compress / compact / recall / scale-demo / … |

## Relation to LLMIntent

Useful as a **memory compaction** preprocessor: compress repetitive tool traces / JSON memories before stuffing into agent context; decode on demand from cold store.
