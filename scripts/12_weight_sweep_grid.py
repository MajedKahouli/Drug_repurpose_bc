#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Fallback-friendly column candidates
REV_CANDS = ["reversal_score","reversal_norm","reversal","reversal_cosine"]
PROX_CANDS = ["prox_z","proximity_norm","proximity_z","prox_raw","prox_score"]
OVL_CANDS = ["jaccard","overlap_jaccard","overlap_norm","overlap","weighted_overlap","overlap_count"]

def norm_drug(s): return str(s).strip().lower()

def pick(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def load_features(p):
    df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p, sep="\t")
    if "drug" not in df.columns:
        for alt in ("compound","pert_iname","drug_name","name"):
            if alt in df.columns:
                df = df.rename(columns={alt:"drug"})
                break
    df["drug_lc"] = df["drug"].map(norm_drug)
    return df

def zscore_safe(x):
    x = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s == 0:
        z = np.zeros_like(x)
    else:
        z = (x - m) / s
        z[~np.isfinite(z)] = 0.0
    return z

def any_hit_topk(scores, y, k):
    # scores: 1D np.array; y: boolean/int {0,1}
    order = np.argsort(-scores)
    topk = order[:k]
    return int(np.any(y[topk] == 1))

def auc_safe(scores, y):
    # Need both classes present
    if y.sum() == 0 or y.sum() == len(y): return np.nan
    try:
        return roc_auc_score(y, scores)
    except Exception:
        return np.nan

def main():
    ap = argparse.ArgumentParser(description="Grid sweep of Stage-1 weights (wr, wp, wo) with Top-K and AUROC.")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--wr-min", type=float, default=0.50)
    ap.add_argument("--wr-max", type=float, default=0.70)
    ap.add_argument("--wr-step", type=float, default=0.01)
    ap.add_argument("--wp-min", type=float, default=0.10)
    ap.add_argument("--wp-max", type=float, default=0.30)
    ap.add_argument("--wp-step", type=float, default=0.01)
    ap.add_argument("--wo-min", type=float, default=0.05, help="minimum overlap weight to keep")
    ap.add_argument("--k-list", default="10,25,50", help="comma-separated K values")
    ap.add_argument("--out-csv", default="work/weight_sweep_grid_summary.csv")
    ap.add_argument("--topn", type=int, default=3, help="print top-N rows by the selection rule")
    args = ap.parse_args()

    Ks = [int(k.strip()) for k in args.k_list.split(",") if k.strip()]
    assert 25 in Ks, "Please include 25 in --k-list since Top-25 is the primary metric."

    # Approved drug set (lowercased)
    A = pd.read_csv(args.approved_tsv, sep="\t")
    a_col = "drug"
    for c in ["compound","pert_iname","drug_name","name"]:
        if c in A.columns: a_col = c
    approved = set(A[a_col].astype(str).str.strip().str.lower())

    # Feature files (patients)
    feats = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not feats:
        feats = sorted(glob.glob(os.path.join(args.features_dir, "features_*.tsv")))
    if not feats:
        raise SystemExit("No features_* files found in --features-dir")

    # Preload per-patient arrays: (Rz, Pz, Oz, y)
    patients = []
    cache = []
    for p in feats:
        df = load_features(p)
        r = pick(df.columns, REV_CANDS); pr = pick(df.columns, PROX_CANDS); ov = pick(df.columns, OVL_CANDS)
        if not (r and pr and ov):
            print(f"[WARN] {Path(p).name}: missing one of required columns (rev/prox/ovl); skipping")
            continue
        Rz = zscore_safe(df[r]); Pz = zscore_safe(df[pr]); Oz = zscore_safe(df[ov])
        y = df["drug_lc"].isin(approved).astype(int).to_numpy()
        cache.append((Rz, Pz, Oz, y))
        patients.append(Path(p).stem.replace("features_",""))
    if not cache:
        raise SystemExit("No valid patient feature files after filtering columns.")

    # Build grid
    wrs = np.round(np.arange(args.wr_min, args.wr_max + 1e-9, args.wr_step), 4)
    wps = np.round(np.arange(args.wp_min, args.wp_max + 1e-9, args.wp_step), 4)
    rows = []
    for wr in wrs:
        for wp in wps:
            wo = round(1.0 - wr - wp, 4)
            if wo < args.wo_min:  # enforce minimum overlap weight and non-negative
                continue
            if wo > 0.90:  # sanity
                continue
            # Aggregate metrics across patients
            top_hits = {k: 0 for k in Ks}
            aucs = []
            n_pat = 0
            for (Rz,Pz,Oz,y) in cache:
                s = wr*Rz + wp*Pz + wo*Oz
                for k in Ks:
                    top_hits[k] += any_hit_topk(s, y, k)
                aucs.append(auc_safe(s, y))
                n_pat += 1
            # summarize
            row = {
                "wr": float(wr), "wp": float(wp), "wo": float(wo),
                "patients": n_pat,
                "auroc_mean": float(np.nanmean(aucs)),
                "auroc_median": float(np.nanmedian(aucs)),
            }
            for k in Ks:
                row[f"top{k}_any_hit_mean"] = top_hits[k] / n_pat
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("Grid produced no valid weight triplets; check bounds/steps.")
    out = out.sort_values(["top25_any_hit_mean","auroc_median","top50_any_hit_mean"], ascending=[False, False, False])

    os.makedirs(Path(args.out_csv).parent, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")

    # Print top-N
    cols = ["wr","wp","wo","patients","auroc_median","auroc_mean"] + \
           [c for c in out.columns if c.startswith("top") and c.endswith("_any_hit_mean")]
    print("\nTop candidates:")
    print(out[cols].head(args.topn).to_string(index=False))
