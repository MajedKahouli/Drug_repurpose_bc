#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd

REV_CANDS = ["reversal_norm","reversal","reversal_score","reversal_cosine"]
PROX_CANDS = ["proximity_norm","prox_z","proximity_z","prox_raw"]
OVL_CANDS = ["overlap_norm","overlap_jaccard","jaccard","overlap","weighted_overlap","overlap_count"]
def norm_drug(s): return str(s).strip().lower()

def pick(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def zscore(col):
    x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(s) or s == 0: return np.zeros_like(x)
    x = (x - m) / s; x[~np.isfinite(x)] = 0.0
    return x

def load_features(path):
    if path.endswith(".parquet"): df = pd.read_parquet(path)
    else: df = pd.read_csv(path, sep="\t")
    if "drug" not in df.columns:
        for alt in ("compound","pert_iname","drug_name","name"):
            if alt in df.columns: df = df.rename(columns={alt:"drug"}); break
    df["drug_lc"] = df["drug"].map(norm_drug)
    return df

def main():
    ap = argparse.ArgumentParser(description="Recompute baseline_* with learned weights.")
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

    for p in fps:
        pid = os.path.basename(p).split(".")[0].replace("features_","")
        F = load_features(p)
        r = pick(F, REV_CANDS); pr = pick(F, PROX_CANDS); o = pick(F, OVL_CANDS)
        if not (r and pr and o):
            print(f"[WARN] {pid}: missing one of r/p/o — skipping")
            continue
        R = zscore(F[r]); P = zscore(F[pr]); O = zscore(F[o])
        score = wr*R + wp*P + wo*O
        out = pd.DataFrame({"drug": F["drug"], "baseline_score": score})
        out = out.sort_values("baseline_score", ascending=False)
        out.to_csv(os.path.join(args.out_dir, f"baseline_{pid}.tsv"), sep="\t", index=False)
        print(f"[OK] wrote baseline_{pid}.tsv")
if __name__ == "__main__":
    main()
