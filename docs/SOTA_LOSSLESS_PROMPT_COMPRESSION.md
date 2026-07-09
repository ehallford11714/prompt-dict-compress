# SOTA: Lossless Prompt Compression (Dictionary-Encoding + ICL)

## Epistemic status

**Paper found (2026):** *Lossless Prompt Compression via Dictionary-Encoding and In-Context Learning: Enabling Cost-Effective LLM Analysis of Repetitive Data* — [arXiv:2604.13066](https://arxiv.org/abs/2604.13066) ([HTML](https://arxiv.org/html/2604.13066v1), [DOI](https://doi.org/10.48550/arxiv.2604.13066)).

This library implements the **same design family** (training-free dictionary meta-tokens + system-prompt codebook + multi-scale pattern mining + token-savings gate). It is **not** an official reproduction of the authors’ code; algorithmic details may differ. Claims about LLM analytical equivalence (0.99+ exact match on LogHub with Claude 3.7 Sonnet) are **from the paper**, not re-validated in this MVP (MVP proves **byte-lossless** encode/decode only; no required LLM API calls).

## Method (dictionary-encoding + ICL)

1. **Mine** frequent subsequences at multiple length scales (whitespace n-grams, optional full lines for logs/JSON).
2. **Select** patterns where dictionary overhead does not exceed savings (token-savings criterion).
3. **Replace** longest-first with compact meta-tokens (this repo: `⟦A⟧`, `⟦B⟧`, … or ASCII `<<PDA>>` if source conflicts).
4. **Pack** `DICTIONARY` into the system prompt + `ENCODED_BODY`.
5. **ICL:** the LLM is instructed to treat meta-tokens as their expansions and reason on the encoded text as if uncompressed.
6. **Lossless decode** (offline): reverse substitution — ground truth for round-trip tests.

### When it works

- System logs, JSON/JSONL event streams, repetitive SQL/code templates, structured telemetry.
- High template redundancy → large token savings; paper reports up to ~60–80% reduction on suitable LogHub sets.

### When it fails / weakens

- Creative prose, unique narrative, high-entropy text (few repeated long chunks).
- Dictionary overhead can erase gains if `min_freq` / pattern length are poorly tuned.
- Analytical equivalence under ICL is an **empirical LLM property** (paper); this package guarantees only **mechanical** losslessness.

## Comparison

| Approach | Lossless? | Needs fine-tune? | Mechanism | Best for |
|----------|-----------|------------------|-----------|----------|
| **Dict + ICL (arXiv:2604.13066 / this lib)** | Yes (reconstruction) | No | Meta-tokens + codebook in prompt | Repetitive logs/JSON/code |
| **LLMLingua / LongLLMLingua / LLMLingua-2** | No (lossy) | No (uses small LM) | Drop low-perplexity / task-irrelevant tokens | General prompts, RAG context |
| **TokenShift-style** (lossy baselines in literature) | No | Varies | Token dropping / shifting | Aggressive length cuts |
| **Harvill et al. 2025 meta-tokens** | Yes | **Yes** (model must learn placeholders) | Meta-token substitution | When you can fine-tune |
| **KV-cache / ZipServ-style system compression** | N/A (runtime) | System-level | Compress KV / serving stack, not prompt text | Latency/memory at inference |
| **PageIndex (VectifyAI)** | Retrieval, not compress | No | Hierarchical ToC / reasoning RAG | Navigate long docs; complementary |

Sources: [LLMLingua](https://github.com/microsoft/llmlingua), [LongLLMLingua](https://arxiv.org/abs/2310.06839), [PageIndex](https://github.com/vectifyai/pageindex), [PageIndex intro](https://pageindex.ai/blog/pageindex-intro).

## PageIndex hierarchical variant (this repo)

PageIndex builds an **in-context hierarchical index** (JSON tree / ToC) so an LLM can *reason* which pages to read. We adapt that idea to **compression**:

| Level | Role |
|-------|------|
| **L0** | Per-page local dictionary encode |
| **L1** | Cross-page / template codebook (shared chunks → meta-tokens) |
| **L2** | PageIndex directory: `page_id → fingerprint / template_id / dict refs` |

**Path to ~1M tokens in the prompt:** keep global + nested dictionaries + slim index + **hot** encoded pages in the prompt; **cold_store** on disk holds all pages for full lossless decode. See [SCALE_ULTRA_LONG.md](SCALE_ULTRA_LONG.md) for the ultra-long → fixed-budget design.

Recursive meta-tokens: longer patterns compressed first; shorter patterns may appear inside expansions only after decode (encode avoids nesting meta inside patterns during selection).

**Dictionary paging / LRU (design + stubs):** for corpora beyond RAM, stream pages, retain an LRU of hot local dicts in the prompt pack, and page cold dicts from disk — implemented as streaming cold_store + hot_page_fraction in `BudgetedContextCompressor`.

## Citations / links

- Harvill-style / dict+ICL paper: https://arxiv.org/abs/2604.13066
- LLMLingua: https://www.microsoft.com/en-us/research/project/llmlingua/
- LongLLMLingua: https://aclanthology.org/2024.acl-long.91.pdf
- PageIndex: https://github.com/vectifyai/pageindex
- LogHub: Zhu et al.; LogHub 2.0 cited in 2604.13066

## Honesty checklist

| Claim | Status |
|-------|--------|
| 2026 dict+ICL paper exists | **Yes** — arXiv:2604.13066 |
| This repo is official paper code | **No** |
| Byte-lossless round-trip | **Yes** — unit tested |
| LLM task equivalence at 60–80% | **Paper result** — not re-run here |
| 100× prompt-resident for arbitrary text | **No** — only highly repetitive; else prompt+cold_store |
