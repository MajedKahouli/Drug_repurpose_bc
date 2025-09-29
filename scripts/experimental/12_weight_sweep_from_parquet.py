import argparse, glob, os, re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def minmax(x):
    x = pd.to_numeric(x, errors="coerce")
    m = np.nanmin(x.values)
    M = np.nanmax(x.values)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - m) / (M - m)

def load_approved(tsv_path, drug_col):
    df = pd.read_csv(tsv_path, sep="\t")
    if drug_col not in df.columns:
        raise SystemExit(f"approved TSV missing column '{drug_col}'")
    return set(df[drug_col].astype(str).str.strip().str.lower())

def patient_id_from_path(path):
    # expects .../features_BC0043.parquet  -> BC0043
    m = re.search(r"features_([^./\\]+)\.parquet$", path.replace("\\","/"))
    return m.group(1) if m else os.path.basename(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--drug-col", default="drug")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-plot", default=None)
    args = ap.parse_args()

    approved = load_approved(args.approved-tsv if hasattr(args, "approved-tsv") else args.approved_tsv, args.drug_col)
    features_dir = args.features_dir

    files = sorted(glob.glob(os.path.join(features_dir, "features_*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {features_dir}")

    rows = []

    # weight grid
    wr_grid = [0.50, 0.55, 0.60, 0.65, 0.70]
    wp_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

    for wr in wr_grid:
        for wp in wp_grid:
            wo = 1.0 - wr - wp
            if wo < 0:  # invalid combo, skip
                continue

            per_patient = []
            for fp in files:
                df = pd.read_parquet(fp)
                needed = {"drug", "reversal_score", "prox_z", "jaccard"}
                if not needed.issubset(df.columns):
                    print(f"Skipping {os.path.basename(fp)} (missing {sorted(list(needed - set(df.columns)))})")
                    continue

                # unify
                drugs = df["drug"].astype(str).str.strip()
                # transform to "higher is better"
                R = -pd.to_numeric(df["reversal_score"], errors="coerce")   # more negative reversal_score => larger R
                P = -pd.to_numeric(df["prox_z"], errors="coerce")           # more negative z => larger P
                O =  pd.to_numeric(df["jaccard"], errors="coerce")          # already 0..1

                # per-patient min-max
                Rn = minmax(R)
                Pn = minmax(P)
                On = minmax(O)

                score = wr*Rn + wp*Pn + wo*On
                rank_idx = score.sort_values(ascending=False).index

                labels = drugs.str.lower().isin(approved).astype(int)

                # metrics
                def any_hit_at_k(k):
                    top_idx = rank_idx[:k]
                    return 1.0 if labels.loc[top_idx].sum() > 0 else 0.0

                top10 = any_hit_at_k(10)
                top25 = any_hit_at_k(25)
                top50 = any_hit_at_k(50)

                # AUROC (skip if only one class)
                try:
                    if labels.nunique() == 2:
                        auc = roc_auc_score(labels.values, score.values)
                    else:
                        auc = np.nan
                except ValueError:
                    auc = np.nan

                per_patient.append((patient_id_from_path(fp), top10, top25, top50, auc))

            if not per_patient:
                continue

            per = pd.DataFrame(per_patient, columns=["patient","top10","top25","top50","auroc"])
            row = {
                "wr": wr, "wp": wp, "wo": wo,
                "patients": len(per),
                "top10_any_hit_mean": per["top10"].mean(),
                "top25_any_hit_mean": per["top25"].mean(),
                "top50_any_hit_mean": per["top50"].mean(),
                "auroc_median": per["auroc"].median(skipna=True),
                "auroc_mean": per["auroc"].mean(skipna=True),
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No valid feature files processed. Check columns and paths.")

    out_df = pd.DataFrame(rows).sort_values(["wp","wr"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print("Wrote summary:", args.out_csv)

    # plot Top-25 vs wr (one line per wp)
    if args.out_plot is None:
        args.out_plot = os.path.join(os.path.dirname(args.out_csv), "fig_weight_sweep_top25_HELDOUT.png")

    plt.figure(figsize=(8,4.2), dpi=180)
    for wp, sub in out_df.groupby("wp"):
        sub = sub.sort_values("wr")
        plt.plot(sub["wr"], sub["top25_any_hit_mean"], marker="o", label=f"wp={wp:.2f}")
    # highlight (0.60,0.20,0.20) if present
    m = out_df[(out_df.wr==0.60) & (out_df.wp==0.20)]
    if not m.empty:
        x, y = m.iloc[0][["wr","top25_any_hit_mean"]]
        plt.scatter([x],[y], s=140, facecolors='none', edgecolors='k', linewidths=2)
        plt.annotate("(0.60,0.20,0.20)", (x,y), xytext=(6,6), textcoords="offset points", fontsize=9)

    plt.ylim(0,1)
    plt.xlabel("Weight on reversal (wr)")
    plt.ylabel("Top-25 any-hit (mean across patients)")
    plt.title("Held-out-style grid (per-patient min–max); wo = 1 − wr − wp")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Weight on proximity (wp)", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.out_plot)
    print("Wrote plot:", args.out_plot)

if __name__ == "__main__":
    main()
