#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse, numpy as np, pandas as pd

# ---------- small utils ----------
def to_numpy_float(x):
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    return np.asarray(x, dtype=float)

def zscore(vec):
    x = to_numpy_float(vec); mu = np.nanmean(x); sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0: return np.zeros_like(x, dtype=float)
    z = (x - mu) / sd; z[np.isnan(z)] = 0.0; return z

def minmax01(vec):
    x = to_numpy_float(vec); vmin = np.nanmin(x); vmax = np.nanmax(x)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(x, dtype=float)
    y = (x - vmin) / (vmax - vmin); y[np.isnan(y)] = 0.0; return y

def auroc_safe(y_true, y_score):
    from sklearn.metrics import roc_auc_score
    yt = to_numpy_float(y_true); ys = to_numpy_float(y_score)
    m = ~(np.isnan(yt) | np.isnan(ys)); yt = yt[m]; ys = ys[m]
    if yt.size == 0 or np.unique(yt).size < 2: return np.nan
    try: return float(roc_auc_score(yt, ys))
    except Exception: return np.nan

def norm_name(s):
    return "" if pd.isna(s) else str(s).strip().lower()

# ---------- auto-detect column names ----------
CANDIDATES = {
    "drug":       ["drug","compound","pert_iname","drug_name","name"],
    "rev":        ["reversal","reversal_cosine","cosine","rev","rev_score","reversal_score"],
    "prox":       ["proximity_z","string_proximity_z","ppi_z","z_proximity","proximityZ","prox_z"],
    "overlap":    ["overlap","jaccard","target_overlap","overlap_jaccard","intxn_overlap"],
}

def autodetect_columns(cols, wanted):
    """Return first match from CANDIDATES[wanted] present in cols, else None."""
    cl = {c.lower(): c for c in cols}
    for cand in CANDIDATES[wanted]:
        if cand.lower() in cl: return cl[cand.lower()]
    return None

def read_patient_file(path, drug_col, rev_col, prox_col, overlap_col):
    ext = os.path.splitext(path)[1].lower()
    df = pd.read_parquet(path) if ext==".parquet" else pd.read_csv(path, sep="\t")
    # auto-detect if not provided
    drug_col     = drug_col     or autodetect_columns(df.columns, "drug")
    rev_col      = rev_col      or autodetect_columns(df.columns, "rev")
    prox_col     = prox_col     or autodetect_columns(df.columns, "prox")
    overlap_col  = overlap_col  or autodetect_columns(df.columns, "overlap")
    missing = [("drug",drug_col),("rev",rev_col),("prox",prox_col),("overlap",overlap_col)]
    miss = [k for k,v in missing if v is None]
    if miss:
        raise SystemExit(f"Missing required columns {miss} in {path}. Found: {list(df.columns)}")
    df["_drug_col"] = drug_col; df["_rev_col"]=rev_col; df["_prox_col"]=prox_col; df["_ov_col"]=overlap_col
    df["_drug_norm"] = df[drug_col].map(norm_name)
    return df

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--drug-col", default=None)
    ap.add_argument("--rev-col", default=None)
    ap.add_argument("--prox-col", default=None)
    ap.add_argument("--overlap-col", default=None)
    ap.add_argument("--topk", nargs="+", type=int, default=[10,25,50])
    args = ap.parse_args()

    appr_df = pd.read_csv(args.approved_tsv, sep=None, engine="python")
    appr_name_col = args.drug_col if args.drug_col and args.drug_col in appr_df.columns else appr_df.columns[0]
    approved = set(appr_df[appr_name_col].map(norm_name))

    # collect both parquet and tsv
    files = sorted([p for g in (os.path.join(args.features_dir,"*.parquet"),
                                os.path.join(args.features_dir,"*.tsv"))
                      for p in glob.glob(g)])
    if not files: raise SystemExit(f"No TSV or Parquet files found in {args.features_dir}")
    per_patient = {}
    for p in files:
        df = read_patient_file(p, args.drug_col, args.rev_col, args.prox_col, args.overlap_col)
        pid = os.path.splitext(os.path.basename(p))[0]
        per_patient[pid] = df
    pids = sorted(per_patient.keys())
    print(f"Found {len(pids)} patient files")

    wrs = [0.50,0.55,0.60,0.65,0.70]
    wps = [0.10,0.15,0.20,0.25,0.30]
    rows = []

    for wr in wrs:
        for wp in wps:
            wo = 1.0 - wr - wp
            if wo < -1e-9:  # require non-negative overlap weight
                continue
            anyhit = {k: [] for k in args.topk}
            aurocs = []

            for pid, df in per_patient.items():
                y = df["_drug_norm"].map(lambda s: 1 if s in approved else 0).to_numpy(int)
                Rflip = -pd.to_numeric(df[df["_rev_col"].iloc[0]], errors="coerce")
                Ppos  = -pd.to_numeric(df[df["_prox_col"].iloc[0]], errors="coerce")
                Oraw  =  pd.to_numeric(df[df["_ov_col"].iloc[0]], errors="coerce")

                Rn, Pn, On = minmax01(Rflip), minmax01(Ppos), minmax01(Oraw)
                score = wr*Rn + wp*Pn + wo*On
                aurocs.append(auroc_safe(y, score))

                order = np.argsort(-score, kind="mergesort")
                ys = y[order]
                for k in args.topk:
                    anyhit[k].append(1 if np.any(ys[:min(k, ys.size)]==1) else 0)

            outrow = {
                "wr": wr, "wp": wp, "wo": round(wo,2),
                "patients": len(pids),
                **{f"top{k}_any_hit_mean": float(np.nanmean(anyhit[k])) for k in args.topk},
                "auroc_median": float(np.nanmedian(aurocs)),
                "auroc_mean": float(np.nanmean(aurocs)),
            }
            rows.append(outrow)

    out = pd.DataFrame(rows).sort_values(["wp","wr"]).reset_index(drop=True)
    cols = ["wr","wp","wo","patients"] + [f"top{k}_any_hit_mean" for k in args.topk] + ["auroc_median","auroc_mean"]
    out = out[cols]
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

if __name__ == "__main__":
    main()
