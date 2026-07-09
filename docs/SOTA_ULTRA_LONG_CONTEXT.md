# SOTA Research Brief: Ultra-Long Context into a Fixed Budget (Lossless / Near-Lossless)

**Project:** PromptDictCompress  
**Date:** 2026-07-09  
**Scope:** Can an *ultra-long* input corpus (`input_budget` tokens) be reduced to a fixed *API-portable* prompt (`output_budget`) **losslessly** (or near-lossless with explicit caveats)?  
**Labeling:** **Published** = peer-reviewed / arXiv with measurable claims. **Engineering** = production practice / OSS. **Speculation** = information-theoretic or design inference not claimed by a paper for extreme ratios.

---

## Executive verdict

| Claim | Status |
| --- | --- |
| Strictly lossless packing of an *arbitrary* ultra-long corpus **entirely inside** a fixed `output_budget` API context | **Not plausible** (entropy / Kolmogorov bounds) |
| Strictly lossless **prompt-resident** packing for *highly repetitive* corpora (logs, templates, JSON) under a fixed `output_budget` | **Possible in principle**, **unproven at extreme ratios** in published work; published dict+ICL reports ~**2–5×** (60–80% *reduction*), not 100× |
| Lossless **corpus** management with **prompt_pack (≤ `output_budget`) + cold_store (disk)** | **SOTA-practical** and matches agent memory / filesystem-as-RAM patterns |
| Lossy extractive / summarization at 10–20× | **Published** (LLMLingua family, Selective Context); **not** lossless |
| KV-cache / Infini-attention “compression” | **System / architecture**; not API-portable prompt compression |

**Recommended design for this project:** hierarchical PageIndex + nested dictionary encoding + two-tier `prompt_pack` / `cold_store` (see [SCALE_ULTRA_LONG.md](./SCALE_ULTRA_LONG.md)).

---

## 1. Dictionary-encoding + ICL (lossless prompt compression)

### Published

**Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning**  
Campos, Lee, Kissos, Paritosh — arXiv:2604.13066 (2026 preprint)  
https://arxiv.org/abs/2604.13066 · https://arxiv.org/html/2604.13066v1

**Claims (as stated by authors; preprint, 0 citations at fetch time):**

- Training-free: replace frequent subsequences with meta-tokens; put the dictionary in the system prompt; LLM analyzes encoded text via ICL.
- Token-savings gate so dictionary overhead does not exceed savings.
- Compression **ratios up to 80%** (i.e. keep ~20% of tokens → ~**5×**), dataset-dependent; also discuss **60–80%** reduction.
- LogHub 2.0 + Claude 3.7 Sonnet: template-based exact match **>0.99**; algorithmic compression Levenshtein similarity **>0.91**.
- Decompression used as proxy for “analytical fidelity.”

**Caveats (honest):**

- This is **not** a published extreme-ratio result. 80% *reduction* ≠ 100× *compression factor* (and published work is ~2–5×).
- Fidelity is measured largely via **round-trip / similarity**, not full suite of downstream analytics at ultra-long scale.
- Works best on **repetitive** data (logs); high-entropy prose will not compress similarly.
- Nested / multi-scale patterns are algorithmic; paper notes avoiding nested meta-tokens in some mining steps.

### Related (storage, not in-prompt decode)

**LoPace** — arXiv:2602.13266 (2026 preprint)  
https://arxiv.org/html/2602.13266  

Lossless **prompt storage** via zstd / BPE packing (~4–5× mean on their set). Reconstruction is **offline**; not “LLM reads compressed bytes in the prompt.”

### Mapping to PromptDictCompress

| Paper idea | Code |
| --- | --- |
| Mine multi-scale patterns + token-savings gate | `src/promptdict/compressor.py` (`DictCompressor`) |
| Dictionary in system / packed prompt | `pack_prompt()` |
| Hierarchical / multi-page | `hierarchical.py`, `scale.py` |
| ultra-long streaming + cold store | `BudgetedContextCompressor` in `scale.py` |

---

## 2. LLMLingua / LongLLMLingua / Selective Context (lossy extractive)

### Published

| Method | Venue / ID | Link | Nature |
| --- | --- | --- | --- |
| **LLMLingua** | EMNLP 2023 · arXiv:2310.05736 | https://arxiv.org/abs/2310.05736 · https://llmlingua.com/ | Perplexity-based token pruning; up to **~20×** with task degradation |
| **LongLLMLingua** | ACL 2024 | https://aclanthology.org/2024.acl-long.91/ | Query-aware compression + reorder for long RAG |
| **LLMLingua-2** | (Microsoft series) | https://github.com/microsoft/LLMLingua/ | Faster task-agnostic classifier compression |
| **Selective Context** | EMNLP 2023 · arXiv:2310.06201 | https://arxiv.org/abs/2310.06201 | Self-information pruning; ~**2×** context cost cut in paper |

**Why not extreme-ratio lossless:**

- These methods **delete** tokens judged low-information. Deleted content is gone → **lossy by definition**.
- High ratios (10–20×) trade accuracy; characterization work finds extractive methods often beat aggressive token pruning, but still lossy ([Characterizing Prompt Compression…](https://arxiv.org/pdf/2407.08892), arXiv:2407.08892).
- Even at 20×, an ultra-long `input_budget` still may not fit a fixed `output_budget`; and remaining text is not a lossless encoding of the original.

**Role vs PromptDictCompress:** complementary for *query-focused* views over cold_store; never substitute for lossless corpus fidelity.

---

## 3. KV-cache compression (SnapKV, H2O, ZipCache, quantization)

### Published

| Method | ID | Link | Mechanism |
| --- | --- | --- | --- |
| **H2O (Heavy-Hitter Oracle)** | NeurIPS 2023 / arXiv lineage | https://arxiv.org/abs/2306.14048 (commonly cited) | Evict low cumulative-attention tokens |
| **SnapKV** | NeurIPS 2024 · arXiv:2404.14469 | https://arxiv.org/abs/2404.14469 | Prefill: observation window → keep clustered important KVs per head |
| **ZipCache** | NeurIPS 2024 · arXiv:2405.14256 | https://arxiv.org/abs/2405.14256 | Salient-token-aware **quantization** of KV |

Survey-style overview (engineering blog, 2026):  
https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/

**Lossless vs lossy at system level:**

- **Quantization** (ZipCache, KIVI-style): approximate KV tensors → **lossy** numerically; often “near-lossless” for generation quality at modest ratios.
- **Eviction** (H2O, SnapKV, StreamingLLM): discarded KVs are **gone** → lossy for exact attention over full history.
- True **bit-exact** KV retention at full length is just “don’t compress.”

**Not API-portable:** requires owning the inference stack (vLLM, TensorRT-LLM, custom kernels). Chat Completions APIs do not accept a compressed KV blob as a substitute for prompt tokens.

---

## 4. Hierarchical / PageIndex / tree-index / recursive retrieval

### Engineering / OSS (primary reference for this project)

**PageIndex (VectifyAI)** — vectorless, reasoning-based RAG via hierarchical ToC tree + agentic tree search  
https://github.com/VectifyAI/PageIndex · https://pageindex.ai/blog/pageindex-intro  

- Build structured tree (sections / pages); optional node summaries.
- Retrieve by **reasoning over the index**, not embedding similarity alone.
- Scale story: file-level trees over corpora (“PageIndex File System”); related: ChatIndex, ConDB (KV-cache-native context DB) — ecosystem claims, treat product blogs as **engineering**, not peer-reviewed ultra-long→fixed-budget proofs.

**Why it matters for ultra-long → fixed budget:**

- You do **not** need the entire ultra-long corpus in the active window.
- Hot path: **directory + dictionaries + selected pages** ≤ `output_budget`.
- Cold path: full pages on disk / object store, fetched by page_id (tool or agent).
- Nested dictionaries amplify gains when many pages share templates.

**Speculation (design):** combining PageIndex *navigation* with *lossless dict-encoded page bodies* is the right hybrid; PageIndex alone with summaries is lossy at leaf content unless leaves are stored verbatim offline.

---

## 5. Classic compression (LZ77, zstd, Brotli) + LLM-readable encodings

### Published / standard

- LZ77 / DEFLATE / gzip, zstd, Brotli: excellent **byte** compression on repetitive data (often **10–100×** on logs).
- **LoPace** (above): uses zstd for **storage** of prompts with offline decompress.

### Can you put compressed bytes in the prompt?

**Usually no (without tools).**

- Transformers consume **tokens**, not arbitrary binary codecs. Base64(zstd(corpus)) is (a) often **larger** in tokens than raw text for moderate sizes, and (b) models do **not** reliably implement zstd/Brotli decode in-weights.
- Practical pattern: **tool / host decompresses** → plaintext or dict-encoded text enters the prompt (same as LoPace’s storage role).
- **LLM-readable** compression ≈ dictionary / template / schema encodings the model can expand via ICL (dict+ICL paper), not DEFLATE bitstreams.

---

## 6. Memory / Compactor / summarization agents

### Engineering (lossy by nature)

| Source | Link | Pattern |
| --- | --- | --- |
| Anthropic — Effective context engineering | https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Compaction = summarize near limit; filesystem / memory tool as external store |
| Claude compaction / cookbook | https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools | Server-side compaction; structured summaries |
| Context compaction survey (Red Hat memory-hub) | https://github.com/redhat-ai-americas/memory-hub/blob/main/research/context-compaction-survey.md | Timing, structured vs free-form summaries |
| Industry 2026 context engineering writeups | e.g. https://pub.towardsai.net/state-of-context-engineering-in-2026-cf92d010eab1 | Progressive disclosure, hybrid windows |

**Caveat:** Compaction **must** drop detail → **lossy**. Correct use: working-set continuity, not archival fidelity. Aligns with PromptDictCompress’s split: **lossy optional views** vs **lossless cold_store**.

---

## 7. Infini-attention / Compressive Transformer / Memorizing Transformer

### Published (architectural — not prompt-level)

| Work | ID | Link |
| --- | --- | --- |
| **Compressive Transformers** | arXiv:1911.05507 | https://arxiv.org/abs/1911.05507 |
| **Memorizing Transformers** | arXiv:2203.08913 | https://arxiv.org/abs/2203.08913 |
| **Infini-attention** | arXiv:2404.07143 | https://arxiv.org/abs/2404.07143 |

Infini-attention reports large **memory-footprint** compression vs storing full KV (paper cites **>100×** vs Memorizing Transformers’ memory size in their comparison table) and long-context experiments (e.g. 1M-scale passkey with 1B model in paper claims). Independent reproduction notes reliability limits (HF blog: https://huggingface.co/blog/infini-attention).

**Relevance:** requires **model/architecture change** or continual pretrain. Does not give API users a way to stuff ultra-long input tokens into a 1M chat request.

---

## 8. Extreme ratios and ultra-long context management

### Published / commentary

- **Billion-token context** as flat attention is widely viewed as economically hard; realistic path = **bounded working context + hierarchy + retrieval + compression**  
  https://cacm.acm.org/news/the-road-to-a-billion-token-context/
- 2026 context-management surveys emphasize caching, hierarchical memory, and compression over raw window growth  
  https://zylos.ai/research/2026-01-19-llm-context-management/
- Infini-attention’s “100×” is **KV memory size**, not “ultra-long prompt tokens → fixed `output_budget` API tokens lossless.”
- Dict+ICL (2604.13066): **~2–5×** class, not extreme (e.g. 100×) prompt packing.
- LLMLingua: up to **~20× lossy**.

**No verified paper found (as of this brief) that demonstrates strictly lossless, API-portable extreme-ratio compression of arbitrary ultra-long corpora into a fixed `output_budget` context.** Claims of 100× should be labeled **speculative** unless restricted to extreme redundancy + two-tier storage semantics.

---

## Analysis questions

### A. Is strictly lossless + entirely inside a fixed `output_budget` API context from ultra-long raw tokens information-theoretically plausible?

**For arbitrary / high-entropy text: No.**

- A lossless compressor cannot map all ultra-long-token strings into a fixed `output_budget`-token code while remaining invertible: there are far more source messages than codewords (pigeonhole / Shannon).
- Typical English / code token streams have **non-trivial entropy rate**. Even strong byte compressors rarely achieve **100×** on mixed natural language; when they do, the source was already highly redundant.
- **Conditional yes:** if the corpus is generated from a **small template set** + sparse parameters (classic log lines), Kolmogorov complexity can be ≪ `output_budget` tokens. Then a dictionary + encoded body *can* fit. That is a **property of the data**, not a universal algorithm.

**Near-lossless caveat:** approximate reconstructions, summaries, or eviction can “fit” but violate invertibility.

### B. When is `prompt_pack (≤ output_budget) + cold_store (lossless on disk)` the correct SOTA-practical design?

**When any of these hold:**

1. You need **bit-exact / string-exact** recovery of the full corpus.
2. The active task only needs a **working set** (index + hot pages + tools to fetch cold pages).
3. You use **API LLMs** (no custom KV / Infini stack).
4. Redundancy is high enough that dictionaries + templates shrink the *addressable* view, but not enough for full extreme-ratio prompt-resident packing.

This matches Anthropic-style **filesystem-as-memory** and PageIndex **navigate-then-read**.

### C. Role of hierarchical PageIndex + nested dictionaries for repetitive corpora

1. **Page / section tree** → O(log n) navigation tokens instead of O(n) body tokens.
2. **Local dictionaries** per page capture page-specific boilerplate.
3. **Global / template codebook** collapses cross-page repetition (the main path to large factors).
4. **Hot pages** in prompt; **cold pages** referenced by id — lossless via cold_store.
5. Optional **lossy** leaf summaries for search only; never confuse with lossless body.

### D. Recommended architecture (user goal)

See [SCALE_ULTRA_LONG.md](./SCALE_ULTRA_LONG.md). Short form:

```
ultra-long corpus
  → stream pages
  → Level-0 local dict-encode (lossless)
  → Level-1 global/template codebook
  → Level-2 PageIndex directory
  → prompt_pack ≤ output_budget (dicts + index + hot pages)
  → cold_store.jsonl (all encoded pages; lossless decode)
  → agent/tools fetch cold pages as needed
```

---

## Ranked methods for *this* use case (API-portable, prefer lossless)

| Rank | Method | Lossless? | Portable? | Typical factor | Fit for ultra-long → fixed budget |
| ---: | --- | --- | --- | --- | --- |
| 1 | **PageIndex hierarchy + nested dict-encode + cold_store** | Yes (with cold) | Yes | Data-dependent; extreme ratios only if redundancy extreme **or** two-tier accounting | **Best** |
| 2 | **Dict-encode + ICL alone** (flat) | Yes (round-trip) | Yes | ~2–5× published | Good for logs ≪ultra-long |
| 3 | **Classic zstd on disk + tool decompress** | Yes | Yes (via tools) | Often high on logs | Storage tier, not prompt |
| 4 | **Extractive RAG / LongLLMLingua** | No | Yes | ~4–20× | Query views over cold_store |
| 5 | **Agent compaction / summarization** | No | Yes | Aggressive | Session memory only |
| 6 | **KV eviction / quant (SnapKV, H2O, ZipCache)** | No / approx | No (infra) | Memory ↓ | Serving stack |
| 7 | **Infini / Compressive / Memorizing** | Approx memory | No (weights) | Arch. | Research models |

---

## What we are *not* claiming

- We do **not** claim a peer-reviewed demonstration of lossless extreme-ratio packing of general ultra-long text into a fixed `output_budget` API context.
- We do **not** treat Infini-attention’s 100× memory figure as prompt compression.
- Scale demos in this repo that use `simulated_input_tokens` are **labeled simulations** of redundancy structure, not literal ultra-long-token materialization (see `scale.py`).

---

## Key citations (quick list)

1. Campos et al., 2026 — Dict-encoding + ICL — https://arxiv.org/abs/2604.13066  
2. Jiang et al., 2023 — LLMLingua — https://arxiv.org/abs/2310.05736  
3. Jiang et al., 2024 — LongLLMLingua — https://aclanthology.org/2024.acl-long.91/  
4. Li et al., 2023 — Selective Context — https://arxiv.org/abs/2310.06201  
5. Li et al., 2024 — SnapKV — https://arxiv.org/abs/2404.14469  
6. Zhang et al. — H2O — https://arxiv.org/abs/2306.14048  
7. He et al., 2024 — ZipCache — https://arxiv.org/abs/2405.14256  
8. Rae et al., 2019 — Compressive Transformers — https://arxiv.org/abs/1911.05507  
9. Wu et al., 2022 — Memorizing Transformers — https://arxiv.org/abs/2203.08913  
10. Munkhdalai et al., 2024 — Infini-attention — https://arxiv.org/abs/2404.07143  
11. VectifyAI PageIndex — https://github.com/VectifyAI/PageIndex  
12. Anthropic context engineering — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents  
13. LoPace, 2026 — https://arxiv.org/html/2602.13266  
14. Prompt compression characterization — https://arxiv.org/pdf/2407.08892  
