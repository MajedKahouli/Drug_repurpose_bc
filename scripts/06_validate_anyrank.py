#!/usr/bin/env python3
import argparse, os, glob, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

def norm(s): return str(s).strip().lower()

def eval_file(fp, approved, score_col, ks=(10,25,50)):
    pid = re.sub(r".*[_-]([A-Za-z0-9]+)\.tsv$", r"\1", os.path.basename(fp))
    df = pd.read_csv(fp, sep="\t")
    if "drug" not in df.columns or score_col not in df.columns:
        raise ValueError(f"{os.path.basename(fp)} missing drug/{score_col}")
    df["drug_lc"] = df["drug"].map(norm)
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    y = df["drug_lc"].isin(approved).astype(int).to_numpy()
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy(float)

    out = {"patient": pid, "n_drugs": len(df), "n_approved_in_list": int(y.sum())}
    out["auroc"] = roc_auc_score(y, s) if (y.sum()>0 and y.sum()<len(y)) else np.nan
    for k in ks:
        k = min(k, len(df))
        A_top = int(y[:k].sum()); out[f"top{k}_hits"] = A_top; out[f"top{k}_any_hit"] = int(A_top>0)
        A_total = int(y.sum()); N = len(y)
        table = np.array([[A_top, k - A_top],[A_total - A_top, (N - k) - (A_total - A_top)]], dtype=int)
        try: _, p = fisher_exact(table, alternative="greater")
        except Exception: p = np.nan
        out[f"top{k}_fisher_p"] = p
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pattern", default="*.tsv")
    ap.add_argument("--score-col", default="ml_score")
    args = ap.parse_args(); os.makedirs(args.out_dir, exist_ok=True)

    approved = pd.read_csv(args.approved_tsv, sep="\t")
    for c in ["drug","drug_name","name","compound","pert_iname"]:
        if c in approved.columns:
            appr = set(approved[c].dropna().map(norm)); break
    else:
        appr = set(approved.iloc[:,0].dropna().map(norm))

    rows = []
    for fp in sorted(glob.glob(os.path.join(args.rankings_dir, args.pattern))):
        try: rows.append(eval_file(fp, appr, args.score_col))
        except Exception as e: print(f"[WARN] {os.path.basename(fp)}: {e}")

    per = pd.DataFrame(rows).sort_values("patient")
    per.to_csv(os.path.join(args.out_dir, "metrics_per_patient.csv"), index=False)

    summ = {"patients": int(len(per)),
            "auroc_mean": float(np.nanmean(per["auroc"])) if "auroc" in per else np.nan,
            "auroc_median": float(np.nanmedian(per["auroc"])) if "auroc" in per else np.nan}
    for k in (10,25,50):
        if f"top{k}_any_hit" in per:
            summ[f"top{k}_any_hit_mean"] = float(np.nanmean(per[f"top{k}_any_hit"]))
    pd.DataFrame([summ]).to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False)
    print("Saved metrics_per_patient.csv and metrics_summary.csv")

if __name__ == "__main__":
    main()
