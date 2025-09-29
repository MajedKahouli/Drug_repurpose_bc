#!/usr/bin/env python3
import os, glob, argparse, math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

def norm(x): return str(x).strip().lower()

def eval_file(tsv_path, approved_set, ks=(10,25,50)):
    pid = os.path.splitext(os.path.basename(tsv_path))[0].replace("reranked_elasticnet_","")
    df = pd.read_csv(tsv_path, sep="\t")
    if "drug" not in df.columns or "ml_score" not in df.columns:
        raise ValueError(f"{tsv_path} must have columns 'drug' and 'ml_score'")
    df = df[["drug","ml_score"]].dropna()
    df["drug_lc"] = df["drug"].map(norm)
    df["label"] = df["drug_lc"].isin(approved_set).astype(int)
    df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)

    y = df["label"].to_numpy(int)
    s = df["ml_score"].to_numpy(float)

    out = {"patient": pid, "n_drugs": len(df), "n_approved_in_list": int(y.sum())}

    # AUROC (only if both classes present)
    out["auroc"] = roc_auc_score(y, s) if (y.sum() > 0 and y.sum() < len(y)) else np.nan

    # top-k recovery + Fisher enrichment
    A_total = int(y.sum()); N = len(y)
    for k in ks:
        k = min(k, N)
        yk = y[:k]
        A_top = int(yk.sum())
        out[f"top{k}_approved"] = A_top
        out[f"top{k}_hit"] = 1 if A_top > 0 else 0

        table = np.array([[A_top, k - A_top],
                          [A_total - A_top, (N - k) - (A_total - A_top)]], dtype=int)
        try:
            _, p = fisher_exact(table, alternative="greater")
        except Exception:
            p = np.nan
        out[f"top{k}_fisher_p"] = p

    return out

def main():
    ap = argparse.ArgumentParser(description="Validate metrics from existing reranked_elasticnet_*.tsv (no retraining).")
    ap.add_argument("--rankings-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[10,25,50])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    approved = pd.read_csv(args.approved_tsv, sep="\t")
    # accept common name columns
    for c in ["drug","drug_name","name","compound","pert_iname"]:
        if c in approved.columns:
            appr = set(approved[c].dropna().map(norm))
            break
    else:
        appr = set(approved.iloc[:,0].dropna().map(norm))

    files = sorted(glob.glob(os.path.join(args.rankings_dir, "reranked_elasticnet_*.tsv")))
    if not files:
        raise SystemExit(f"No reranked files found in {args.rankings_dir}")

    rows = []
    for fp in files:
        try:
            rows.append(eval_file(fp, appr, ks=args.ks))
        except Exception as e:
            print(f"[WARN] {os.path.basename(fp)}: {e}")

    per = pd.DataFrame(rows).sort_values("patient")
    per_path = os.path.join(args.out_dir, "reranked_elasticnet_metrics_per_patient.csv")
    per.to_csv(per_path, index=False)

    # summary
    summ = {"n_patients": int(len(per)),
            "auroc_median": float(np.nanmedian(per["auroc"]))}
    for k in args.ks:
        summ[f"top{k}_hit_rate"] = float(np.nanmean(per[f"top{k}_hit"]))
        summ[f"top{k}_median_approved"] = float(np.nanmedian(per[f"top{k}_approved"]))
        summ[f"top{k}_median_fisher_p"] = float(np.nanmedian(per[f"top{k}_fisher_p"]))
    summ_path = os.path.join(args.out_dir, "reranked_elasticnet_metrics_summary.csv")
    pd.DataFrame([summ]).to_csv(summ_path, index=False)

    print("Saved:")
    print(" -", per_path)
    print(" -", summ_path)

if __name__ == "__main__":
    main()
