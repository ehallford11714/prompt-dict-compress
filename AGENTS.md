# Agents — PromptDictCompress

Lossless **dictionary prompt compression**, hierarchical PageIndex packing, working-set compact/recall for context budgets.

## Connect

1. `pip install -e .`
2. MCP: `python -m promptdict.mcp` — Cursor: [`.cursor/mcp.json`](.cursor/mcp.json)
3. Skill: [`.cursor/skills/promptdict/SKILL.md`](.cursor/skills/promptdict/SKILL.md)

## Tools

| Tool | Use |
|------|-----|
| `compress_text` | Flat dict-encode |
| `compress_hierarchical` | PageIndex hierarchical pack |

## CLI

```powershell
promptdict compress path\to\file.txt -o packed.json
promptdict hierarchical path\to\corpus.txt -o hier.json
promptdict compact -f messages.json -o compacted.json --budget 8000
```

Prefer compression for repetitive corpora; use compact/recall for agent working sets.
