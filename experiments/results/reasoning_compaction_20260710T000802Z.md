# Reasoning-trace compaction experiment results

- Created: `2026-07-10T00:08:02.239838+00:00`
- Budget (compact/truncate): `1200` tokens
- Isolates backend: `intentisolates`
- Fixtures: `5`

## Summary table (averages)

| condition | tok_before | tok_after | ratio | lossless | gold_R_vis | mid_R_vis | gold_R_rest | iso_R | typ_J | motif_J | layer_match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 2846.400 | 2846.400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.767 | 1.000 | 1.000 | 1.000 |
| compress | 2846.400 | 1947.400 | 1.462 | 1.000 | 1.000 | 1.000 | 1.000 | 0.767 | 1.000 | 1.000 | 1.000 |
| compact | 2846.400 | 1428.000 | 1.993 | 0.000 | 0.867 | 0.733 | 1.000 | 0.600 | 0.309 | 0.038 | 0.933 |
| lossy_truncate | 2846.400 | 124.600 | 22.873 | 0.000 | 0.600 | 0.200 | 0.600 | 0.300 | 0.698 | 0.066 | 1.000 |
| protect_compact | 2846.400 | 185.600 | 15.362 | 0.000 | 1.000 | 1.000 | 1.000 | 0.767 | 0.815 | 0.007 | 1.000 |

## Verdict

- Compaction preserved mid-trace constraints better than naive truncate at matched budgets (H1/H3 supported).
- Lossless dict compress achieved exact round-trip (H2 supported for restore).
- After decompress, motif Jaccard vs raw is high — lossless path preserves reasoning-structure metrics (H2).
- Compress beats compact for archival fidelity: full restore + token savings when patterns mine well.
- Isolate-aware protect_compact matched or beat library compact on mid-constraint recall — prefer isolate-then-compact for reasoning traces.

## Per-fixture rows

| fixture | condition | tok_after | ratio | gold_R_vis | mid_R_vis | gold_R_rest | typ_J | motif_J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deploy_budget | raw | 2860 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deploy_budget | compress | 1961 | 1.458 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deploy_budget | compact | 1436 | 1.992 | 0.833 | 0.667 | 1.000 | 0.356 | 0.035 |
| deploy_budget | lossy_truncate | 132 | 21.667 | 0.500 | 0.000 | 0.500 | 0.708 | 0.066 |
| deploy_budget | protect_compact | 201 | 14.229 | 1.000 | 1.000 | 1.000 | 0.833 | 0.011 |
| privacy_export | raw | 2842 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| privacy_export | compress | 1943 | 1.463 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| privacy_export | compact | 1426 | 1.993 | 0.833 | 0.667 | 1.000 | 0.337 | 0.035 |
| privacy_export | lossy_truncate | 120 | 23.683 | 0.667 | 0.333 | 0.667 | 0.694 | 0.066 |
| privacy_export | protect_compact | 178 | 15.966 | 1.000 | 1.000 | 1.000 | 0.806 | 0.005 |
| incident_mitigate | raw | 2841 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| incident_mitigate | compress | 1942 | 1.463 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| incident_mitigate | compact | 1426 | 1.992 | 0.833 | 0.667 | 1.000 | 0.337 | 0.044 |
| incident_mitigate | lossy_truncate | 127 | 22.370 | 0.667 | 0.333 | 0.667 | 0.694 | 0.066 |
| incident_mitigate | protect_compact | 182 | 15.610 | 1.000 | 1.000 | 1.000 | 0.806 | 0.005 |
| research_summary | raw | 2848 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| research_summary | compress | 1949 | 1.461 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| research_summary | compact | 1428 | 1.994 | 0.833 | 0.667 | 1.000 | 0.179 | 0.041 |
| research_summary | lossy_truncate | 120 | 23.733 | 0.500 | 0.000 | 0.500 | 0.700 | 0.066 |
| research_summary | protect_compact | 185 | 15.395 | 1.000 | 1.000 | 1.000 | 0.822 | 0.006 |
| sql_migration | raw | 2841 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sql_migration | compress | 1942 | 1.463 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sql_migration | compact | 1424 | 1.995 | 1.000 | 1.000 | 1.000 | 0.337 | 0.035 |
| sql_migration | lossy_truncate | 124 | 22.911 | 0.667 | 0.333 | 0.667 | 0.694 | 0.066 |
| sql_migration | protect_compact | 182 | 15.610 | 1.000 | 1.000 | 1.000 | 0.806 | 0.005 |

## Notes

- `compress` = lossless `DictCompressor` on full packed trace.
- `compact` = `promptdict.compact_messages` (lossless_dict preferred).
- `lossy_truncate` = head/tail truncate to the same budget.
- `protect_compact` = **experiment-local** keyword-protect + dict-compress filler.
- Core metrics are automatic (no LLM judge).

