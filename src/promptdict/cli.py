"""CLI for promptdict — compress / compact / recall / suite."""



from __future__ import annotations



import argparse

import json

import sys

from pathlib import Path



from . import __version__

from .compact import compact_messages, expand_cold_ref

from .compressor import DictCompressor

from .hierarchical import HierarchicalPageIndexCompress

from .metrics import compression_metrics, estimate_tokens

from .recall import recall

from .scale import BudgetedContextCompressor, run_scale_demo

from .suite import PromptMemorySuite





def _cmd_compress(args: argparse.Namespace) -> int:

    text = Path(args.file).read_text(encoding="utf-8", errors="replace")

    comp = DictCompressor(

        min_freq=args.min_freq,

        max_dict_size=args.max_dict,

        use_line_patterns=not args.no_lines,

    )

    result = comp.compress(text)

    out = Path(args.out)

    out.write_text(result.to_json(), encoding="utf-8")

    m = result.metrics

    print(

        f"Wrote {out} | orig_tok≈{m.original_tokens} packed≈{m.packed_tokens} "

        f"factor≈{m.compression_factor:.3f} dict={m.dictionary_entries}"

    )

    return 0





def _cmd_decompress(args: argparse.Namespace) -> int:

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))

    kind = data.get("kind", "")

    if kind in ("hierarchical_pageindex",):

        h = HierarchicalPageIndexCompress()

        text = h.decompress_dict(data)

    elif data.get("cold_store_path") or data.get("out_dir"):

        out_dir = data.get("out_dir") or Path(args.file).parent

        text = BudgetedContextCompressor().decompress_all(out_dir)

    else:

        encoded, dictionary = DictCompressor.from_packed_dict(data)

        text = DictCompressor().decompress(encoded, dictionary)

    if args.out:

        Path(args.out).write_text(text, encoding="utf-8")

        print(f"Wrote {args.out} ({len(text)} chars)")

    else:

        sys.stdout.write(text)

    return 0





def _cmd_hierarchical(args: argparse.Namespace) -> int:

    text = Path(args.file).read_text(encoding="utf-8", errors="replace")

    h = HierarchicalPageIndexCompress(page_size=args.page_size)

    result = h.compress(text, page_size=args.page_size)

    out = Path(args.out or "hierarchical_packed.json")

    out.write_text(result.to_json(), encoding="utf-8")

    m = result.metrics

    print(

        f"Wrote {out} | pages={m['n_pages']} orig_tok≈{m['original_tokens_est']} "

        f"packed≈{m['packed_tokens_est']} factor≈{m['compression_factor']:.3f}"

    )

    return 0





def _cmd_metrics(args: argparse.Namespace) -> int:

    path = Path(args.file)

    raw = path.read_text(encoding="utf-8", errors="replace")

    try:

        data = json.loads(raw)

    except json.JSONDecodeError:

        print(

            json.dumps(

                {"path": str(path), "tokens_est": estimate_tokens(raw), "chars": len(raw)},

                indent=2,

            )

        )

        return 0

    if "metrics" in data:

        print(json.dumps(data["metrics"], indent=2))

        return 0

    if "encoded" in data and "dictionary" in data:

        encoded = data["encoded"]

        dictionary = data["dictionary"]

        original = DictCompressor().decompress(encoded, dictionary)

        m = compression_metrics(

            original, encoded, dictionary, packed_prompt=data.get("packed_prompt")

        )

        print(json.dumps(m.to_dict(), indent=2))

        return 0

    print(json.dumps(data, indent=2)[:4000])

    return 0





def _cmd_scale_demo(args: argparse.Namespace) -> int:

    result = run_scale_demo(

        target_in=args.input_budget,

        target_out=args.output_budget,

        simulate=args.simulate,

        out_dir=args.out_dir,

        max_materialized_pages=args.max_pages,

    )

    print(json.dumps(result.to_dict(), indent=2))

    comp = BudgetedContextCompressor()

    try:

        p0 = comp.decompress_page(result.out_dir, 0)

        print(f"\nSample page 0 decode OK ({len(p0)} chars)")

    except Exception as e:

        print(f"\nSample decode issue: {e}", file=sys.stderr)

        return 1

    return 0





def _cmd_scale_compress(args: argparse.Namespace) -> int:

    comp = BudgetedContextCompressor(

        input_token_budget=args.input_budget,

        output_token_budget=args.output_budget,

        page_chars=args.page_chars,

    )

    result = comp.compress_stream(args.file, args.out_dir)

    print(json.dumps(result.to_dict(), indent=2))

    return 0





def _cmd_compact(args: argparse.Namespace) -> int:

    raw = Path(args.file).read_text(encoding="utf-8", errors="replace")

    try:

        data = json.loads(raw)

    except json.JSONDecodeError as e:

        print(f"compact expects a JSON message list: {e}", file=sys.stderr)

        return 1

    if isinstance(data, dict) and "messages" in data:

        messages = data["messages"]

    elif isinstance(data, list):

        messages = data

    else:

        print("JSON must be a message list or {\"messages\": [...]}", file=sys.stderr)

        return 1

    result = compact_messages(

        messages,

        budget=args.budget,

        mode=args.mode,

        keep_last_n=args.keep_last,

    )

    out = Path(args.out)

    out.write_text(result.to_json(), encoding="utf-8")

    print(

        f"Wrote {out} | msgs {result.metrics['n_input_messages']}→"

        f"{result.metrics['n_output_messages']} "

        f"tok≈{result.original_tokens_est}→{result.compacted_tokens_est} "

        f"mode={result.mode} cold_refs={len(result.cold_refs)}"

    )

    return 0





def _cmd_recall(args: argparse.Namespace) -> int:

    page_ids = None

    if args.page_id is not None:

        page_ids = [args.page_id]

    if args.page_ids:

        extra = [p.strip() for p in args.page_ids.split(",") if p.strip()]

        page_ids = (page_ids or []) + extra

    try:

        result = recall(

            store=args.store,

            page_ids=page_ids,

            query=args.query,

            top_k=args.top_k,

            method=args.method,

        )

    except NotImplementedError as e:

        print(str(e), file=sys.stderr)

        return 1

    except (FileNotFoundError, KeyError, ValueError) as e:

        print(str(e), file=sys.stderr)

        return 1

    if args.out:

        Path(args.out).write_text(result.packed_fragment, encoding="utf-8")

        print(f"Wrote {args.out} | hits={len(result.hits)}")

    else:

        sys.stdout.write(result.packed_fragment)

        if not result.packed_fragment.endswith("\n"):

            sys.stdout.write("\n")

    return 0





def _cmd_suite(args: argparse.Namespace) -> int:

    """Print suite overview / run a tiny smoke path when --demo."""

    if not args.demo:

        print(

            "PromptMemorySuite pillars:\n"

            "  compress  — dictionary-encode / hierarchical / budgeted pack\n"

            "  compact   — manage agent message working set under a budget\n"

            "  recall    — restore pages from cold_store by id or keyword\n\n"

            "Python:\n"

            "  from promptdict import PromptMemorySuite\n"

            "  suite = PromptMemorySuite(output_budget=1_000_000)\n"

            "  packed = suite.compress(text)\n"

            "  compacted = suite.compact(messages)\n"

            "  restored = suite.recall(page_id=0, store='.scale_demo')\n\n"

            "CLI: promptdict compress|compact|recall|scale-demo|...\n"

            "Docs: docs/SUITE.md\n"

            f"Version: {__version__}"

        )

        return 0



    suite = PromptMemorySuite(output_budget=args.output_budget, compact_budget=2_000)

    sample = ("INFO status=200 msg=ok request_id=1\n" * 40) + (

        "ERROR timeout retry=true queue=default\n" * 40

    )

    flat = suite.compress(sample, mode="flat")

    assert isinstance(flat, type(flat))  # CompressResult

    messages = [

        {"role": "system", "content": "You are a log analyst."},

        {"role": "user", "content": sample},

        {"role": "assistant", "content": "Saw repeated INFO/ERROR templates."},

        {"role": "user", "content": "Summarize errors."},

    ]

    compacted = suite.compact(messages, budget=2_000)

    out = Path(args.out_dir)

    scaled = suite.compress(sample * 3, mode="budgeted", out_dir=out)

    restored = suite.recall(page_id=0, store=out)

    print(

        json.dumps(

            {

                "version": __version__,

                "flat_factor": flat.metrics.compression_factor,

                "compact_tokens": compacted.compacted_tokens_est,

                "scale_out": str(getattr(scaled, "out_dir", out)),

                "recall_chars": len(restored.text),

            },

            indent=2,

        )

    )

    return 0





def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(

        prog="promptdict",

        description=(

            "Prompt memory suite: lossless dictionary compression, "

            "working-set compaction, and cold_store recall "

            "(ultra-long corpora → fixed context budget via prompt_pack + cold_store)"

        ),

    )

    p.add_argument("--version", action="version", version=f"promptdict {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)



    c = sub.add_parser("compress", help="Dictionary-encode a file (compression pillar)")

    c.add_argument("--file", "-f", required=True)

    c.add_argument("--out", "-o", default="packed.json")

    c.add_argument("--min-freq", type=int, default=3)

    c.add_argument("--max-dict", type=int, default=256)

    c.add_argument("--no-lines", action="store_true")

    c.set_defaults(func=_cmd_compress)



    d = sub.add_parser("decompress", help="Lossless decode packed JSON / scale artifact")

    d.add_argument("--file", "-f", required=True)

    d.add_argument("--out", "-o", default=None)

    d.set_defaults(func=_cmd_decompress)



    h = sub.add_parser("hierarchical", help="PageIndex hierarchical compress")

    h.add_argument("--file", "-f", required=True)

    h.add_argument("--page-size", type=int, default=4000)

    h.add_argument("--out", "-o", default="hierarchical_packed.json")

    h.set_defaults(func=_cmd_hierarchical)



    m = sub.add_parser("metrics", help="Show token metrics for a file or packed JSON")

    m.add_argument("--file", "-f", required=True)

    m.set_defaults(func=_cmd_metrics)



    s = sub.add_parser(

        "scale-demo",

        help="Simulate packing an ultra-long corpus into an output_budget",

    )

    s.add_argument(

        "--input-budget",

        "--target-in",

        type=int,

        default=10_000_000,

        dest="input_budget",

        help="Simulated / target input token budget",

    )

    s.add_argument(

        "--output-budget",

        "--target-out",

        type=int,

        default=1_000_000,

        dest="output_budget",

        help="Prompt-resident token budget",

    )

    s.add_argument("--simulate", action="store_true", default=True)

    s.add_argument("--no-simulate", action="store_false", dest="simulate")

    s.add_argument("--out-dir", default=".scale_demo")

    s.add_argument("--max-pages", type=int, default=400)

    s.set_defaults(func=_cmd_scale_demo)



    sc = sub.add_parser(

        "scale-compress",

        help="Streaming hierarchical compress file → out_dir (budgeted two-tier)",

    )

    sc.add_argument("--file", "-f", required=True)

    sc.add_argument("--out-dir", "-o", default="scale_out")

    sc.add_argument(

        "--input-budget",

        "--target-in",

        type=int,

        default=10_000_000,

        dest="input_budget",

    )

    sc.add_argument(

        "--output-budget",

        "--target-out",

        type=int,

        default=1_000_000,

        dest="output_budget",

    )

    sc.add_argument("--page-chars", type=int, default=16000)

    sc.set_defaults(func=_cmd_scale_compress)



    cp = sub.add_parser(

        "compact",

        help="Compact a JSON message list under a token budget (compaction pillar)",

    )

    cp.add_argument("--file", "-f", required=True, help="JSON messages file")

    cp.add_argument("--out", "-o", default="compacted.json")

    cp.add_argument("--budget", type=int, default=8_000)

    cp.add_argument(

        "--mode",

        choices=["auto", "lossless_dict", "lossy_stub"],

        default="auto",

    )

    cp.add_argument("--keep-last", type=int, default=4)

    cp.set_defaults(func=_cmd_compact)



    r = sub.add_parser(

        "recall",

        help="Recall pages from cold_store by page_id or keyword (recall pillar)",

    )

    r.add_argument(

        "--store",

        "-s",

        required=True,

        help="cold_store.jsonl or scale out_dir",

    )

    r.add_argument("--page-id", default=None)

    r.add_argument("--page-ids", default=None, help="Comma-separated page ids")

    r.add_argument("--query", "-q", default=None)

    r.add_argument("--top-k", type=int, default=5)

    r.add_argument(

        "--method",

        choices=["keyword", "embedding"],

        default="keyword",

    )

    r.add_argument("--out", "-o", default=None)

    r.set_defaults(func=_cmd_recall)



    su = sub.add_parser("suite", help="Show PromptMemorySuite overview (or --demo)")

    su.add_argument("--demo", action="store_true", help="Run a tiny end-to-end smoke demo")

    su.add_argument("--output-budget", type=int, default=1_000_000)

    su.add_argument("--out-dir", default=".suite_demo")

    su.set_defaults(func=_cmd_suite)



    return p





def main(argv: list[str] | None = None) -> None:

    parser = build_parser()

    args = parser.parse_args(argv)

    raise SystemExit(args.func(args))





if __name__ == "__main__":

    main()


