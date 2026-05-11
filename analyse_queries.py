"""Summarise query timings recorded by Shakespeare.py to query_metrics.jsonl.

Usage:
    venv/bin/python analyse_queries.py            # full report
    venv/bin/python analyse_queries.py --tail 20  # only the last 20 queries
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = Path(__file__).parent / "query_metrics.jsonl"


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def summary_line(label: str, vals: list[float]) -> str:
    if not vals:
        return f"{label:24s} no data"
    return (
        f"{label:24s} "
        f"n={len(vals):4d}  "
        f"mean={statistics.mean(vals):6.1f}s  "
        f"p50={percentile(vals, 50):6.1f}s  "
        f"p90={percentile(vals, 90):6.1f}s  "
        f"p99={percentile(vals, 99):6.1f}s  "
        f"max={max(vals):6.1f}s"
    )


def histogram(vals: list[float], buckets: list[float]) -> None:
    """Simple ASCII histogram. `buckets` is the upper edge of each bucket."""
    counts = [0] * (len(buckets) + 1)
    for v in vals:
        placed = False
        for i, edge in enumerate(buckets):
            if v <= edge:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    width = max(counts) or 1
    edges_lbl = [f"≤{e:>5.1f}s" for e in buckets] + [f">{buckets[-1]:>5.1f}s"]
    for lbl, c in zip(edges_lbl, counts):
        bar = "█" * int(40 * c / width)
        print(f"  {lbl}  {c:4d}  {bar}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=int, help="Only analyse the last N queries.")
    ap.add_argument("--metrics", type=Path, default=METRICS)
    args = ap.parse_args()

    if not args.metrics.exists():
        print(f"No metrics file at {args.metrics}. Ask the UI a few questions first.")
        return

    rows = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if args.tail:
        rows = rows[-args.tail:]
    if not rows:
        print("No data.")
        return

    print(f"queries: {len(rows)}")
    print(f"range:   {rows[0]['ts']}  →  {rows[-1]['ts']}")
    print()

    print("─── overall ─────────────────────────────────────────────────────")
    print(summary_line("total_s", [r["total_s"] for r in rows]))
    print(summary_line("setup_s (condense+retr)", [r["setup_s"] for r in rows]))
    ft = [r["first_token_s"] for r in rows if r.get("first_token_s") is not None]
    print(summary_line("first_token_s (TTFT)", ft))

    print()
    print("─── single-turn vs follow-up ───────────────────────────────────")
    single = [r["total_s"] for r in rows if r["history_turns"] == 0]
    follow = [r["total_s"] for r in rows if r["history_turns"] > 0]
    print(summary_line("single-turn total_s", single))
    print(summary_line("follow-up  total_s", follow))
    if single and follow:
        ratio = statistics.mean(follow) / statistics.mean(single)
        print(f"  follow-ups are {ratio:.2f}× the cost of single-turns "
              f"(condense_plus_context adds an extra LLM call)")

    print()
    print("─── by detected play ───────────────────────────────────────────")
    by_play: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_play[r.get("work_code") or "(none / no filter)"].append(r["total_s"])
    for code in sorted(by_play, key=lambda k: -len(by_play[k])):
        print(summary_line(code, by_play[code]))

    print()
    print("─── total_s histogram ──────────────────────────────────────────")
    histogram(
        [r["total_s"] for r in rows],
        buckets=[5, 10, 15, 20, 30, 45, 60, 90],
    )


if __name__ == "__main__":
    main()
