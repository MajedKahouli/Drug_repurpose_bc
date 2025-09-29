#!/usr/bin/env python3
import argparse, os, glob, pandas as pd
from pathlib import Path

def norm_drug(s): return str(s).strip().lower()

def load_baseline(path):
    df = pd.read_csv(path, sep="\t")
    # accept either 'baseline_score' or 'score_resolved'
    if "baseline_score" not in df.columns and "score_resolved" in df.columns:
        df = df.rename(columns={"score_resolved":"baseline_score"})
    if "drug" not in df.columns:
        raise ValueError(f"{path} missing 'drug'")
    if "baseline_score" not in df.columns:
        raise ValueError(f"{path} missing 'baseline_score' / 'score_resolved'")
    if "drug_lc" not in df.columns:
        df["drug_lc"] = df["drug"].map(norm_drug)
    return df[["drug","drug_lc","baseline_score"]]

def load_ml(path):
    df = pd.read_csv(path, sep="\t")
    # EN outputs should have 'ml_score'
    if "ml_score" not in df.columns:
        # try common fallbacks
        for c in ["score","pred","pred_score","yhat"]:
            if c in df.columns:
                df = df.rename(columns={c:"ml_score"})
                break
    if "ml_score" not in df.columns:
        raise ValueError(f"{path} missing 'ml_score'")
    if "drug" not in df.columns:
        raise ValueError(f"{path} missing 'drug'")
    if "drug_lc" not in df.columns:
        df["drug_lc"] = df["drug"].map(norm_drug)
    return df[["drug","drug_lc","ml_score"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--ml-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weight", type=float, required=True, help="weight on ML score (0..1)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # discover patient ids from baseline folder
    b_paths = sorted(glob.glob(os.path.join(args.baseline_dir, "baseline_*.tsv")))
    if not b_paths:
        raise SystemExit(f"No baseline_*.tsv in {args.baseline_dir}")

    for bp in b_paths:
        pid = Path(bp).stem.replace("baseline_","")  # e.g. BC0043
        # ML files often named like reranked_elasticnet_BCxxxx.tsv
        mlp = os.path.join(args.ml_dir, f"reranked_elasticnet_{pid}.tsv")
        if not os.path.exists(mlp):
            # try a few alternates
            alts = glob.glob(os.path.join(args.ml_dir, f"*{pid}*.tsv"))
            if not alts:
                print(f"[WARN] {pid}: no ML file found in {args.ml_dir}; skipping")
                continue
            mlp = alts[0]

        B = load_baseline(bp)
        M = load_ml(mlp)

        # inner join to ensure we rank common drugs only
        df = B.merge(M, on=["drug_lc","drug"], how="inner", validate="one_to_one")
        if df.empty:
            print(f"[WARN] {pid}: empty merge; skipping")
            continue

        w = float(args.weight)
        df["ml_score_orig"] = df["ml_score"]
        df["blend_score"]   = w*df["ml_score_orig"] + (1.0 - w)*df["baseline_score"]
        # Back-compat: expose the blended score as 'ml_score' too
        df["ml_score"]      = df["blend_score"]

        # tidy & write
        out_cols = ["drug","blend_score","baseline_score","ml_score_orig","ml_score","drug_lc"]
        df = df[out_cols].sort_values("blend_score", ascending=False)
        outp = os.path.join(args.out_dir, f"reranked_blend_{pid}.tsv")
        df.to_csv(outp, sep="\t", index=False)
        print(f"[OK] {pid}: wrote reranked_blend_{pid}.tsv  (n={len(df)})")

if __name__ == "__main__":
    main()
