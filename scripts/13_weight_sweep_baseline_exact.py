import os, glob, argparse, re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def minmax(x):
    x = pd.Series(x).astype(float)
    lo, hi = x.min(skipna=True), x.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - lo) / (hi - lo)

def load_approved(tsv_path, col):
    ref = pd.read_csv(tsv_path, sep="\t")
    if col not in ref.columns:
        raise SystemExit(f"Approved TSV missing column '{col}'. Has: {list(ref.columns)}")
    approved = set(ref[col].astype(str).str.strip().str.lower().unique())
    return approved

def patient_from_filename(path):
    # e.g., features_BC0043.parquet -> BC0043
    m = re.search(r"features_([A-Za-z0-9]+)\.parquet$", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)

def eval_patient(df, approved_names, drug_col, k_list=(25,), wr=0.60, wp=0.20):
    """Return dict with top-k any-hit flags per k and AUROC for one patient."""
    # 1) build signed, normalized features (paper recipe)
    #    reversal_score: more negative = stronger -> flip sign
    #    prox_z: more negative = closer -> flip sign
    #    jaccard: already positive overlap
    if "reversal_score" not in df.columns or "prox_z" not in df.columns or "jaccard" not in df.columns:
        raise ValueError("Missing expected columns: reversal_score, prox_z, jaccard")

    R = minmax(-pd.to_numeric(df["reversal_score"], errors="coerce"))
    P = minmax(-pd.to_numeric(df["prox_z"], errors="coerce"))
    O = minmax(pd.to_numeric(df["jaccard"], errors="coerce").fillna(0.0))

    wo = 1.0 - wr - wp
    if wo < 0:
        raise ValueError("wo became negative; skip this (wr, wp).")

    score = wr*R + wp*P + wo*O

    # 2) labels
    drugs = df[drug_col].astype(str).str.strip().str.lower()
    y = drugs.isin(approved_names).astype(int).values

    # If all zeros or all ones, AUROC is undefined; guard
    auroc = np.nan
    if y.sum() > 0 and y.sum() < len(y):
        try:
            auroc = roc_auc_score(y, score.values)
        except Exception:
            auroc = np.nan

    # 3) top-k any-hit
    order = np.argsort(-score.values)  # descending
    any_hits = {}
    for k in k_list:
        top_idx = order[:k]
        any_hits[k] = int(y[top_idx].sum() > 0)

    return any_hits, auroc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--approved-col", default="drug", help="column in TSV with approved drug names")
    ap.add_argument("--drug-col", default="drug", help="column in parquet with drug names")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-fig1", default=None, help="PNG for Top-25 any-hit sweep")
    ap.add_argument("--out-fig2", default=None, help="PNG for AUROC sweep")
    args = ap.parse_args()

    approved = load_approved(args.approved-tsv if hasattr(args, "approved-tsv") else args.approved_tsv, args.approved_col)

    paths = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not paths:
        raise SystemExit(f"No parquet found in {args.features_dir}")

    rows = []
    wr_grid = [0.50, 0.55, 0.60, 0.65, 0.70]
    wp_grid = [0.10, 0.15, 0.20]

    # collect patient IDs and per-patient metrics
    for wr in wr_grid:
        for wp in wp_grid:
            wo = 1.0 - wr - wp
            if wo < 0:
                continue

            top25_flags = []
            aurocs = []
            n_pat = 0

            for fp in paths:
                df = pd.read_parquet(fp)
                # keep only rows with required columns
                missing = [c for c in ("reversal_score","prox_z","jaccard",args.drug_col) if c not in df.columns]
                if missing:
                    print(f"Skipping {os.path.basename(fp)} (missing {missing})")
                    continue

                try:
                    any_hits, auroc = eval_patient(df, approved, args.drug_col, k_list=(25,), wr=wr, wp=wp)
                except Exception as e:
                    print(f"Skipping {os.path.basename(fp)} ({e})")
                    continue

                top25_flags.append(any_hits[25])
                aurocs.append(auroc)
                n_pat += 1

            if n_pat == 0:
                continue

            # aggregate
            top25_mean = float(np.nanmean(top25_flags))
            auroc_med = float(np.nanmedian(aurocs))

            rows.append({
                "wr": wr, "wp": wp, "wo": round(wo, 2),
                "patients": n_pat,
                "top25_any_hit_mean": top25_mean,
                "auroc_median": auroc_med
            })

    out = pd.DataFrame(rows).sort_values(["wp","wr"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("\nWrote summary:", args.out_csv)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # -------- Plots
    def plot_metric(metric_col, ylabel, title, out_png):
        if not out_png:
            return
        plt.figure(figsize=(8.5,4.2), dpi=160)
        for wp, sub in out.groupby("wp"):
            sub = sub.sort_values("wr")
            plt.plot(sub["wr"], sub[metric_col], marker="o", label=f"wp={wp:.2f}")
        # highlight (0.60, 0.20, 0.20) if present
        m = out[(out.wr==0.60) & (out.wp==0.20)]
        if not m.empty:
            x, y = m.iloc[0][["wr", metric_col]]
            plt.scatter([x],[y], s=160, facecolors="none", edgecolors="k", linewidths=2)
            plt.annotate("(0.60,0.20,0.20)", (x,y), xytext=(6,6), textcoords="offset points", fontsize=9)
        plt.ylim(0,1)
        plt.xlabel("Weight on reversal (wr)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend(title="Weight on proximity (wp)", ncol=3, fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png)
        print("Wrote figure:", out_png)

    plot_metric("top25_any_hit_mean",
                "Top-25 any-hit (mean across patients)",
                "Held-out-style grid (exact baseline recipe); wo = 1 − wr − wp",
                args.out_fig1 or os.path.splitext(args.out_csv)[0] + "_top25.png")

    plot_metric("auroc_median",
                "AUROC (median across patients)",
                "Held-out-style grid (exact baseline recipe); wo = 1 − wr − wp",
                args.out_fig2 or os.path.splitext(args.out_csv)[0] + "_auroc.png")

if __name__ == "__main__":
    main()
