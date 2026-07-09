# Reasoning-trace compaction experiment results

- Created: `2026-07-09T22:20:42.883482+00:00`
- Budget (compact/truncate): `1800` tokens
- Isolates backend: `intentisolates`
- Fixtures: `5`

## Summary table (averages)

| condition | tok_before | tok_after | ratio | lossless | gold_R_vis | gold_R_rest | iso_R | iso_P | typ_J | motif_J | layer_match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1801.400 | 1801.400 | 1.000 | 1.000 | 1.000 | 1.000 | 0.767 | 0.491 | 1.000 | 1.000 | 1.000 |
| compress | 1801.400 | 1541.000 | 1.169 | 1.000 | 1.000 | 1.000 | 0.767 | 0.491 | 1.000 | 0.000 | 1.000 |
| compact | 1801.400 | 1754.000 | 1.027 | 0.000 | 1.000 | 1.000 | 0.767 | 0.491 | 1.000 | 0.119 | 1.000 |
| lossy_truncate | 1801.400 | 1585.600 | 1.136 | 0.000 | 1.000 | 1.000 | 0.767 | 0.491 | 1.000 | 0.487 | 1.000 |
| protect_compact | 1801.400 | 1549.400 | 1.163 | 1.000 | 1.000 | 1.000 | 0.767 | 0.491 | 1.000 | 0.013 | 1.000 |

## Verdict

- Compact vs truncate gold-phrase recall was similar; inspect motif/layer metrics.
- Lossless dict compress achieved exact round-trip (H2 supported for restore).
- Compress beats compact for archival fidelity: full restore + token savings when patterns mine well.
- Isolate-aware protect_compact matched or beat library compact on visible gold-phrase recall — prefer isolate-then-compact for reasoning traces.

## Per-fixture rows

| fixture | condition | tok_after | ratio | gold_R_vis | gold_R_rest | typ_J | motif_J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| deploy_budget | raw | 1816 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| deploy_budget | compress | 1555 | 1.168 | 1.000 | 1.000 | 1.000 | 0.000 |
| deploy_budget | compact | 1768 | 1.027 | 1.000 | 1.000 | 1.000 | 0.108 |
| deploy_budget | lossy_truncate | 1598 | 1.136 | 1.000 | 1.000 | 1.000 | 0.496 |
| deploy_budget | protect_compact | 1564 | 1.161 | 1.000 | 1.000 | 1.000 | 0.011 |
| privacy_export | raw | 1798 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| privacy_export | compress | 1537 | 1.170 | 1.000 | 1.000 | 1.000 | 0.000 |
| privacy_export | compact | 1750 | 1.027 | 1.000 | 1.000 | 1.000 | 0.123 |
| privacy_export | lossy_truncate | 1582 | 1.137 | 1.000 | 1.000 | 1.000 | 0.534 |
| privacy_export | protect_compact | 1546 | 1.163 | 1.000 | 1.000 | 1.000 | 0.011 |
| incident_mitigate | raw | 1797 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| incident_mitigate | compress | 1537 | 1.169 | 1.000 | 1.000 | 1.000 | 0.000 |
| incident_mitigate | compact | 1750 | 1.027 | 1.000 | 1.000 | 1.000 | 0.123 |
| incident_mitigate | lossy_truncate | 1582 | 1.136 | 1.000 | 1.000 | 1.000 | 0.323 |
| incident_mitigate | protect_compact | 1545 | 1.163 | 1.000 | 1.000 | 1.000 | 0.011 |
| research_summary | raw | 1801 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| research_summary | compress | 1541 | 1.169 | 1.000 | 1.000 | 1.000 | 0.000 |
| research_summary | compact | 1754 | 1.027 | 1.000 | 1.000 | 1.000 | 0.114 |
| research_summary | lossy_truncate | 1586 | 1.136 | 1.000 | 1.000 | 1.000 | 0.526 |
| research_summary | protect_compact | 1549 | 1.163 | 1.000 | 1.000 | 1.000 | 0.018 |
| sql_migration | raw | 1795 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| sql_migration | compress | 1535 | 1.169 | 1.000 | 1.000 | 1.000 | 0.000 |
| sql_migration | compact | 1748 | 1.027 | 1.000 | 1.000 | 1.000 | 0.128 |
| sql_migration | lossy_truncate | 1580 | 1.136 | 1.000 | 1.000 | 1.000 | 0.557 |
| sql_migration | protect_compact | 1543 | 1.163 | 1.000 | 1.000 | 1.000 | 0.014 |

## Notes

- `compress` = lossless `DictCompressor` on full packed trace.
- `compact` = `promptdict.compact_messages` (lossless_dict preferred).
- `lossy_truncate` = head/tail truncate to the same budget.
- `protect_compact` = **experiment-local** keyword-protect + dict-compress filler.
- Core metrics are automatic (no LLM judge).

