#!/usr/bin/env python3
import re, argparse
from pathlib import Path
import pandas as pd

def collect(prefix: str, root: str) -> pd.DataFrame:
    root = Path(root)
    paths = sorted(root.glob(f"work/{prefix}*_metrics/metrics_summary.csv"))
    rows = []
    for p in paths:
        parent = p.parent.name  # e.g., reranked_blend_learned_w050_metrics
        m = re.search(r"_w(\d{2,3})_metrics", parent)  # _w050_
        if m:
            w = int(m.group(1)) / 100.0
        else:
            m = re.search(r"_w(\d+(?:\.\d+)?)_metrics", parent)  # _w0.65_
            if not m:
                print(f"[WARN] could not parse weight from: {p}")
                continue
            w = float(m.group(1))
        df = pd.read_csv(p)
        row = {"weight": w}
        for k, v in df.iloc[0].items():
            row[k] = v
        rows.append(row)
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.sort_values("weight")
    return tab

def print_table(tab: pd.DataFrame, label: str):
    if tab.empty:
        print(f"\nNo runs found for {label}.")
        return
    cols = [c for c in ["weight","auroc_median","auroc_mean",
                        "top10_any_hit_mean","top25_any_hit_mean","top50_any_hit_mean","patients"]
            if c in tab.columns]
    print(f"\nBlend weight sweep ({label}):")
    print(tab[cols].to_string(index=False))
    # choose best by Top-25 any-hit, tie-break by AUROC median
    by = [c for c in ["top25_any_hit_mean","auroc_median"] if c in tab.columns]
    best = tab.sort_values(by, ascending=[False, False]).iloc[0]
    print(f"Recommended weight ({label}): {round(float(best['weight']), 2)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="reranked_blend_en_w",
                    help="Folder prefix under work/. Examples: reranked_blend_en_w, reranked_blend_learned_w")
    ap.add_argument("--both", action="store_true",
                    help="Show both 'en' and 'learned' sweeps if present.")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    if args.both:
        t_en = collect("reranked_blend_en_w", args.root)
        t_learned = collect("reranked_blend_learned_w", args.root)
        print_table(t_en, "baseline_en")
        print_table(t_learned, "baseline_learned")
    else:
        t = collect(args.prefix, args.root)
        if t.empty:
            print(f"No metrics_summary.csv found for prefix '{args.prefix}'.")
            maybe = sorted(Path(args.root).glob("work/reranked_blend_*_metrics"))
            if maybe:
                print("\nFound these directories:")
                for p in maybe:
                    print(" -", p)
        else:
            print_table(t, args.prefix)
