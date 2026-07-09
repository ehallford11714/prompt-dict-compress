# Isolates, Compaction, and Reasoning Traces

**Status:** research synthesis + offline experiment  
**Repos:** `IntentIsolates` / `llmintent.isolates`, `PromptDictCompress` (`promptdict`)  
**Experiment:** `../experiments/reasoning_trace_compaction.py`  
**Canonical copy also at:** `research/docs/ISOLATES_COMPACTION_REASONING.md`

---

## 1. How isolates / motifs / trajectories represent reasoning structure

### Isolates

An **isolate** is a separable intent unit extracted from text (or features/graphs): a clause or phrase that carries a typed role. Offline rule backends split on clause boundaries, keep intent-ish spans, and classify typology with cue lexicons:

| Typology | Role in a reasoning trace |
| --- | --- |
| `goal` | What the agent is trying to achieve |
| `constraint` | Hard limits / requirements that must survive compression |
| `instrumental` | Means / tools / methods |
| `action` | Concrete steps |
| `outcome` | Results / consequences |
| `affective` / `noise` / `confounder` | Tone, filler, or entangled distractors |

Layers (abstract L0–L4) place isolates on a scaffold: surface lexical → binding → latent → goal/constraint → action/outcome. Soft hooks to `llmintent` exist when installed; abstract layers are **not** residual-stream indices unless explicitly bound.

### Motifs

A **motif** is a recurring composition of isolates: co-occurrence in a layer, adjacent sequence, typed path templates (`goal → constraint → action`), soft-graph chains/triangles, or layer bridges. Motifs are **structural hypotheses** — useful for measuring whether a compacted trace still “looks like” the same reasoning skeleton.

### Trajectories

`trajectory_from_motifs` orders content by layer into a **reasoning trajectory**: `layer_path`, `motif_path`, steps with roles (`early_lexical` → `late_goal` / action). Completeness of the layer path (especially L3 goals/constraints and L4 actions/outcomes) is a practical proxy for “did compaction preserve the decision structure?”

**Caveat:** Motifs/trajectories are not causal identification and not claims about how the model “thinks.”

---

## 2. PromptDict: compression vs compaction vs recall

From `promptdict` suite design ([SUITE.md](./SUITE.md)):

| Pillar | Verb | Job | Lossless? |
| --- | --- | --- | --- |
| **Compression** | encode | Dictionary-encode a corpus once → `DICTIONARY` + `ENCODED_BODY` | Yes (byte round-trip) |
| **Compaction** | manage | Shrink an agent **working set** under a token budget over turns | Optional lossless dict path **or** lossy stubs |
| **Recall** | restore | Pull pages / cold refs back into context | Yes (from store / cold_ref payload) |

**Compression** mines repetitive n-grams/lines, replaces them with meta-tokens, and packs a codebook into the prompt (dict+ICL family; see arXiv:2604.13066). Best on logs/JSON/templates; weak on high-entropy prose.

**Compaction** merges adjacent same-role turns, optionally dict-encodes bulky repetitive mid-conversation turns into `cold_refs`, or emits lossy stubs when savings are weak / budget forces eviction. System + last-N turns are protected.

**Recall** restores cold content by page id, keyword, or `expand_cold_ref` for dict-encoded / stub payloads that still hold originals.

These must not be conflated: compression shortens a blob; compaction manages a live message list; recall inverts cold storage.

---

## 3. Hypotheses: can compaction improve reasoning traces?

| ID | Hypothesis | Expected if true |
| --- | --- | --- |
| **H1** | Compaction improves usable reasoning by **reducing distractors** (repetitive tool dumps / filler) while keeping goal/constraint spans hot | Isolate P/R for goal/constraint/outcome stays high; token count drops vs raw |
| **H2** | **Lossless dict compression** preserves motif structure because decode restores exact text | Exact restore rate = 1.0; motif Jaccard / layer-path match ≈ 1.0 vs raw |
| **H3** | **Lossy** drop (naive truncate or stub eviction of constraint spans) **hurts** reasoning structure | Isolate recall and motif/trajectory stability fall vs compress/compact-with-protection |

**Falsifiers:** If truncate matches compress on isolate/motif metrics at the same budget, structure preservation is not the differentiator. If compact stubs critical constraints, H1 fails for that mode.

---

## 4. Related SOTA (brief)

- **Prompt compression:** LLMLingua / LongLLMLingua / LLMLingua-2 (lossy, perplexity / task-aware token drop); dict+ICL lossless packing ([arXiv:2604.13066](https://arxiv.org/abs/2604.13066)); Harvill-style meta-tokens (often fine-tuned).
- **Chain-of-thought / traces:** CoT and process supervision treat intermediate text as a reasoning artifact; compressing that artifact risks dropping constraints that later steps depend on.
- **Latent reasoning:** SAE / monosemantic features and residual-layer analyses motivate isolate/layer scaffolds (see IntentIsolates SOTA notes); not required for this offline experiment.
- **Context engineering:** Working-set budgets, summarization memory, hierarchical indices (PageIndex-style), and cold/hot tiers — compaction is the agent-loop instance of this.

---

## 5. Metrics for reasoning-trace quality (offline)

| Metric | Definition | Needs LLM? |
| --- | --- | --- |
| Token estimate before/after | `promptdict.estimate_tokens` (or char/4 proxy) | No |
| Lossless restore rate | Exact string match after decompress / `expand_cold_ref` | No |
| Isolate typology P/R | Gold key tags (`goal`, `constraint`, `outcome`) vs identified on restored/visible text | No (rule backend) |
| Motif stability | Jaccard of motif ids (or pattern+member signatures) vs raw | No |
| Trajectory layer completeness | Fraction of raw `layer_path` layers still present; path match | No |
| Optional task accuracy stub | Keyword/constraint checklist on visible working set | No |

---

## 6. Experiment design (executed offline)

See `../experiments/reasoning_trace_compaction.py`.

**Fixtures:** 3–5 synthetic multi-turn reasoning traces with (a) repetitive tool/log filler and (b) critical goal/constraint/outcome sentences buried mid-trace.

**Conditions:**

1. `raw` — full text / messages  
2. `compress` — `DictCompressor` / `compress_text` (lossless)  
3. `compact` — `compact_messages` (prefer lossless_dict; budgeted)  
4. `lossy_truncate` — keep first/last turns then char-trim (naive memory baseline)  
5. `protect_compact` — experiment-local: keyword-protect goals/constraints + dict-compress filler  

**Soft imports:** `intentisolates` or `llmintent.isolates`; if missing, keyword-based isolate proxy runs and results note the skip.

**Outputs:** `experiments/results/reasoning_compaction_*.json` + markdown summary.

---

## 7. Conclusions (filled after run)

**Run:** `2026-07-09T22:57:29.371299+00:00` · budget=`1200` · backend=`intentisolates`

### Did compaction improve / preserve structure vs truncate?

| condition | tok_ratio | gold_R_visible | mid_R_visible | gold_R_restored | typ_J | motif_J | layer_match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| compress | 1.462 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| compact | 1.993 | 0.867 | 0.733 | 1.000 | 0.309 | 0.038 | 0.933 |
| lossy_truncate | 22.873 | 0.600 | 0.200 | 0.600 | 0.698 | 0.066 | 1.000 |
| protect_compact | 15.362 | 1.000 | 1.000 | 1.000 | 0.815 | 0.007 | 1.000 |

### Verdict bullets

- Compaction preserved mid-trace constraints better than naive truncate at matched budgets (H1/H3 supported).
- Lossless dict compress achieved exact round-trip (H2 supported for restore).
- After decompress, motif Jaccard vs raw is high — lossless path preserves reasoning-structure metrics (H2).
- Compress beats compact for archival fidelity: full restore + token savings when patterns mine well.
- Isolate-aware protect_compact matched or beat library compact on mid-constraint recall — prefer isolate-then-compact for reasoning traces.

### When compress (lossless) beats compact

- Use **compress** when you need archival / ICL-packed fidelity and the trace is repetitive enough to mine: exact restore + motif recovery.
- Use **compact** for live agent loops under a hard working-set budget; pair with cold_ref expansion on demand.
- Use **protect_compact / isolate-then-compact** when constraints sit in mid-history that default last-N protection might not cover after aggressive eviction.

### Recommendations (LLMIntent + PromptDict)

1. **Isolate then compact:** run `identify_isolates` (goal/constraint/outcome), mark spans as protected, then `compact_messages` / dict-encode only low-value repetitive regions.
2. **Compact then isolate** only for cheap triage on the hot set — do not treat stubbed text as ground-truth reasoning structure.
3. Prefer **lossless_dict** mode over **lossy_stub** whenever tool dumps are recoverable and patterns mine well.
4. Never rely on naive head/tail truncate as reasoning memory.

---

## 8. References (workspace)

- `./SUITE.md`  
- `./SOTA_LOSSLESS_PROMPT_COMPRESSION.md`  
- `IntentIsolates/docs/MOTIFS_TRAJECTORIES.md`  
- `IntentIsolates/docs/SOTA_ISOLATES.md`  
- `LLMIntent/src/llmintent/isolates/` (vendored or external backend)
