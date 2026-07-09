# Reasoning-trace compaction experiments

Offline harness comparing raw / lossless compress / compact / truncate /
protect-compact on synthetic reasoning traces.

```bash
# from PromptDictCompress repo root
python experiments/reasoning_trace_compaction.py
python experiments/reasoning_trace_compaction.py --budget 1200
```

Results land in `experiments/results/reasoning_compaction_*.{json,md}` and
`reasoning_compaction_latest.*`.

See also: `../../docs/ISOLATES_COMPACTION_REASONING.md` (research synthesis).
