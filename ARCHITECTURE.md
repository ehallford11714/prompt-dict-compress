"""Architecture overview for PromptDictCompress / promptdict."""

# PromptDictCompress architecture

## Goals

1. **Lossless** dictionary-encoding of repetitive prompts (logs, JSON, code).
2. Pack `system_dictionary + encoded_body` for API LLMs (ICL over the codebook).
3. **Hierarchical PageIndex** path toward large corpora (1M prompt budget; 100M-scale via streaming + cold store).

## Core pipeline (flat)

```
text → mine patterns (lines / n-grams) → dictionary {⟦A⟧→chunk}
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

## Streaming 100M→1M (`MillionTokenBudgetCompressor`)

```
page stream → Level-0 local encode → append cold_store.jsonl
           → template fingerprint counts
           → Level-1/2 global dict + template codebook + slim PAGE_INDEX
           → prompt_pack (≤ output_token_budget): dicts + index + HOT pages
           → cold_store: all encoded pages for full lossless decode
```

**Semantics:** corpus losslessness = `prompt_pack + cold_store`. The 1M prompt is an addressable compressed *view*, not necessarily the entire corpus inline.

## Modules

| Module | Role |
|--------|------|
| `mining.py` | n-gram / line / char pattern mining |
| `compressor.py` | `DictCompressor` encode/decode/pack |
| `hierarchical.py` | in-memory PageIndex variant |
| `scale.py` | streaming + scale-demo generator |
| `metrics.py` | tiktoken-or-chars/4 estimates |
| `cli.py` | compress / decompress / hierarchical / scale-demo |

## Relation to LLMIntent

Useful as a **memory compaction** preprocessor: compress repetitive tool traces / JSON memories before stuffing into agent context; decode on demand from cold store.
