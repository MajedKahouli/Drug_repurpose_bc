#!/usr/bin/env python3
# Recompute baseline_* with learned weights AND include the component columns EN expects

import argparse, os, glob, numpy as np, pandas as pd

REV_CANDS = ["reversal_score","reversal_norm","reversal","reversal_cosine"]
PROX_CANDS = ["prox_z","proximity_norm","proximity_z","prox_raw"]
OVL_CANDS = ["jaccard","overlap_jaccard","overlap_norm","overlap","weighted_overlap","overlap_count"]

def norm_drug(s): return str(s).strip().lower()
def pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None
def zscore(col):
    x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    m, s = np.nanmean(x), np.nanstd(x)
    out = np.zeros_like(x) if (not np.isfinite(s) or s==0) else (x - m)/s
    out[~np.isfinite(out)] = 0.0
    return out
def load_features(path):
    if path.endswith(".parquet"): df = pd.read_parquet(path)
    else: df = pd.read_csv(path, sep="\t")
    if "drug" not in df.columns:
        for alt in ("compound","pert_iname","drug_name","name"):
            if alt in df.columns: df = df.rename(columns={alt:"drug"}); break
    df["drug_lc"] = df["drug"].map(norm_drug)
    return df

def main():
    ap = argparse.ArgumentParser(description="Enriched baselines with learned weights + r/p/o columns")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--weights-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    W = pd.read_csv(args.weights_csv)
    wr = float(W.loc[W["feature"]=="reversal","weight_median_nonneg"].iloc[0])
    wp = float(W.loc[W["feature"]=="proximity","weight_median_nonneg"].iloc[0])
    wo = float(W.loc[W["feature"]=="overlap","weight_median_nonneg"].iloc[0])

    os.makedirs(args.out_dir, exist_ok=True)
    fps = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not fps: fps = sorted(glob.glob(os.path.join(args.features_dir, "features_*.tsv")))
    if not fps: raise SystemExit("No features files found.")

    for p in fps:
        pid = os.path.basename(p).split(".")[0].replace("features_","")
        F = load_features(p)
        r_col = pick(F, REV_CANDS); p_col = pick(F, PROX_CANDS); o_col = pick(F, OVL_CANDS)
        if not (r_col and p_col and o_col):
            print(f"[WARN] {pid}: missing components r={r_col}, p={p_col}, o={o_col} — skipping")
            continue

        Rz, Pz, Oz = zscore(F[r_col]), zscore(F[p_col]), zscore(F[o_col])
        baseline_score = wr*Rz + wp*Pz + wo*Oz

        out = pd.DataFrame({
            "drug": F["drug"], "drug_lc": F["drug_lc"],
            "reversal_score": pd.to_numeric(F[r_col], errors="coerce"),
            "prox_z":         pd.to_numeric(F[p_col], errors="coerce"),
            "jaccard":        pd.to_numeric(F[o_col], errors="coerce"),
            "baseline_score": baseline_score
        })
        out["score_resolved"] = out["baseline_score"]
        out = out.replace([np.inf, -np.inf], np.nan).sort_values("baseline_score", ascending=False)
        out.to_csv(os.path.join(args.out_dir, f"baseline_{pid}.tsv"), sep="\t", index=False)
        print(f"[OK] wrote baseline_{pid}.tsv")
if __name__ == "__main__":
    main()
