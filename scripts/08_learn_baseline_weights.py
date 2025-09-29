#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

REV_CANDS = ["reversal_norm","reversal","reversal_score","reversal_cosine"]
PROX_CANDS = ["proximity_norm","prox_z","proximity_z","prox_raw"]
OVL_CANDS = ["overlap_norm","overlap_jaccard","jaccard","overlap","weighted_overlap","overlap_count"]

def norm_drug(s): return str(s).strip().lower()
def pick1(cols, cands):
    for c in cands:
        if c in cols: return c
    return None

def zscore(col):
    x = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
    m, s = np.nanmean(x), np.nanstd(x)
    if not np.isfinite(s) or s == 0: return np.zeros_like(x)
    x = (x - m) / s
    x[~np.isfinite(x)] = 0.0
    return x

def load_features(path):
    if path.endswith(".parquet"): df = pd.read_parquet(path)
    else: df = pd.read_csv(path, sep="\t")
    if "drug" not in df.columns:
        for alt in ("compound","pert_iname","drug_name","name"):
            if alt in df.columns: df = df.rename(columns={alt:"drug"}); break
    df["drug_lc"] = df["drug"].map(norm_drug)
    return df

def load_approved(tsv):
    df = pd.read_csv(tsv, sep="\t")
    for c in ["drug","Drug","drug_name","compound","pert_iname","name"]:
        if c in df.columns: return set(df[c].dropna().map(norm_drug))
    return set(df.iloc[:,0].dropna().map(norm_drug))

def main():
    ap = argparse.ArgumentParser(description="Learn baseline weights (LOPO donors).")
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--topn-neg", type=int, default=400)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()

    # discover patients
    fp = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not fp: fp = sorted(glob.glob(os.path.join(args.features_dir, "features_*.tsv")))
    pids = [os.path.basename(p).split(".")[0].replace("features_","") for p in fp]
    Feats = {}
    for pid in pids:
        try:
            F = load_features(os.path.join(args.features_dir, f"features_{pid}.parquet"))
        except Exception:
            F = load_features(os.path.join(args.features_dir, f"features_{pid}.tsv"))
        Feats[pid] = F

    approved = load_approved(args.approved_tsv)

    fold_rows = []
    for holdout in pids:
        donors = [p for p in pids if p != holdout]
        X_parts, y_parts = [], []
        have_cols = set()
        for d in donors:
            F = Feats[d]
            r = pick1(F.columns, REV_CANDS)
            p = pick1(F.columns, PROX_CANDS)
            o = pick1(F.columns, OVL_CANDS)
            if not (r and p and o): continue

            df = F[["drug_lc", r, p, o]].copy()
            df["r"] = zscore(df[r]); df["p"] = zscore(df[p]); df["o"] = zscore(df[o])
            df["y"] = df["drug_lc"].isin(approved).astype(int)

            pos = df[df["y"]==1]
            if pos.empty: continue
            neg = df[df["y"]==0].sample(min(args.topn_neg, len(df)-len(pos)), random_state=13) if len(df)>len(pos) else df[df["y"]==0]

            X_parts.append(neg[["r","p","o"]].to_numpy(dtype=float))
            y_parts.append(np.zeros(len(neg), dtype=int))
            X_parts.append(pos[["r","p","o"]].to_numpy(dtype=float))
            y_parts.append(np.ones(len(pos), dtype=int))
            have_cols |= {"r","p","o"}

        if not X_parts: continue
        X = np.vstack(X_parts); y = np.concatenate(y_parts)

        clf = LogisticRegression(C=args.C, penalty="l2", solver="lbfgs", max_iter=500, n_jobs=None)
        clf.fit(X, y)
        coef = clf.coef_.ravel()  # r,p,o
        # non-negative normalized weights (our prior form)
        w = np.clip(coef, 0, None)
        s = w.sum() if w.sum()>0 else 1.0
        w = w / s

        fold_rows.append({"holdout": holdout,
                          "coef_r": float(coef[0]), "coef_p": float(coef[1]), "coef_o": float(coef[2]),
                          "w_r": float(w[0]), "w_p": float(w[1]), "w_o": float(w[2])})

    folds = pd.DataFrame(fold_rows).sort_values("holdout")
    if folds.empty:
        raise SystemExit("No folds learned — check your feature columns.")

    # cohort summary (use medians)
    sm = folds.median(numeric_only=True)
    out = pd.DataFrame({
        "feature": ["reversal","proximity","overlap"],
        "coef_median": [sm["coef_r"], sm["coef_p"], sm["coef_o"]],
        "weight_median_nonneg": [sm["w_r"], sm["w_p"], sm["w_o"]],
    })
    out.to_csv(args.out_csv, index=False)
    # also save per-fold for appendix
    folds.to_csv(os.path.splitext(args.out_csv)[0] + "_per_fold.csv", index=False)
    print("Saved:", args.out_csv, "and per-fold CSV.")
if __name__ == "__main__":
    main()
