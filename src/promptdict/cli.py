"""CLI for promptdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .compressor import DictCompressor
from .hierarchical import HierarchicalPageIndexCompress
from .metrics import compression_metrics, estimate_tokens
from .scale import MillionTokenBudgetCompressor, run_scale_demo


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
        text = MillionTokenBudgetCompressor().decompress_all(out_dir)
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
        target_in=args.target_in,
        target_out=args.target_out,
        simulate=args.simulate,
        out_dir=args.out_dir,
        max_materialized_pages=args.max_pages,
    )
    print(json.dumps(result.to_dict(), indent=2))
    comp = MillionTokenBudgetCompressor()
    try:
        p0 = comp.decompress_page(result.out_dir, 0)
        print(f"\nSample page 0 decode OK ({len(p0)} chars)")
    except Exception as e:
        print(f"\nSample decode issue: {e}", file=sys.stderr)
        return 1
    return 0


def _cmd_scale_compress(args: argparse.Namespace) -> int:
    comp = MillionTokenBudgetCompressor(
        input_token_budget=args.target_in,
        output_token_budget=args.target_out,
        page_chars=args.page_chars,
    )
    result = comp.compress_stream(args.file, args.out_dir)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="promptdict",
        description=(
            "Lossless dictionary-encoding prompt compression "
            "(+ hierarchical / 100M→1M scale path)"
        ),
    )
    p.add_argument("--version", action="version", version=f"promptdict {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="Dictionary-encode a file")
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

    s = sub.add_parser("scale-demo", help="Simulate 100M→1M hierarchical budget compression")
    s.add_argument("--target-in", type=int, default=100_000_000)
    s.add_argument("--target-out", type=int, default=1_000_000)
    s.add_argument("--simulate", action="store_true", default=True)
    s.add_argument("--no-simulate", action="store_false", dest="simulate")
    s.add_argument("--out-dir", default=".scale_demo")
    s.add_argument("--max-pages", type=int, default=400)
    s.set_defaults(func=_cmd_scale_demo)

    sc = sub.add_parser("scale-compress", help="Streaming hierarchical compress file → out_dir")
    sc.add_argument("--file", "-f", required=True)
    sc.add_argument("--out-dir", "-o", default="scale_out")
    sc.add_argument("--target-in", type=int, default=100_000_000)
    sc.add_argument("--target-out", type=int, default=1_000_000)
    sc.add_argument("--page-chars", type=int, default=16000)
    sc.set_defaults(func=_cmd_scale_compress)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
