#!/usr/bin/env python3
# LOPO XGBoost (with sklearn fallback) re-ranker

import argparse, os, glob, warnings, sys
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

# Try xgboost; fall back if unavailable
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Accept the feature names produced by your pipeline
FEATURE_CANDIDATES = [
    # reversal / expression-based
    "reversal", "reversal_cosine", "reversal_norm", "reversal_score",
    # network proximity
    "prox_z", "proximity_z", "proximity_norm", "prox_raw",
    # overlap / target-set similarity
    "overlap", "overlap_jaccard", "overlap_dice", "overlap_norm",
    "jaccard", "overlap_count", "weighted_overlap",
    # any precomputed overall score
    "overall", "blend", "score",
]

# Baseline score column fallbacks (your baseline_score first)
BASELINE_SCORE_CANDIDATES = [
    "baseline_score", "score", "overall", "blend",
    # as a last resort we can recompute a blend if parts exist
    # (reversal_norm/prox_norm/overlap_norm) with 0.6/0.2/0.2
]

def load_approved_set(tsv_path: str) -> set:
    df = pd.read_csv(tsv_path, sep="\t")
    for col in ["drug", "Drug", "DRUG", "drug_name", "compound", "pert_iname"]:
        if col in df.columns:
            return set(df[col].astype(str).str.strip().str.lower())
    raise ValueError(f"{tsv_path} must have a column with drug names (e.g. 'drug').")

def safe_lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()

def pick_feature_columns(df: pd.DataFrame) -> list:
    cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not cols:
        raise ValueError(f"No known feature columns found in features file. Have: {list(df.columns)}")
    return cols

def baseline_pick_or_blend(df_base: pd.DataFrame) -> pd.Series:
    # 1) prefer an existing score column
    for c in BASELINE_SCORE_CANDIDATES:
        if c in df_base.columns:
            return df_base[c]

    # 2) compute a blend if parts exist (0.6/0.2/0.2)
    rev_norm = None
    prox_norm = None
    ovl_norm = None
    for cand in ["reversal_norm", "reversal", "reversal_score"]:
        if cand in df_base.columns:
            rev_norm = df_base[cand]
            break
    for cand in ["proximity_norm", "prox_norm", "prox_z", "proximity_z"]:
        if cand in df_base.columns:
            prox_norm = df_base[cand]
            break
    for cand in ["overlap_norm", "overlap_jaccard", "overlap", "jaccard"]:
        if cand in df_base.columns:
            ovl_norm = df_base[cand]
            break

    if (rev_norm is not None) and (prox_norm is not None) and (ovl_norm is not None):
        def minmax(s):
            s = pd.to_numeric(s, errors="coerce")
            lo, hi = np.nanmin(s.values), np.nanmax(s.values)
            if hi > lo:
                return (s - lo) / (hi - lo)
            return s.fillna(0.0)
        r = minmax(rev_norm); p = minmax(prox_norm); o = minmax(ovl_norm)
        return 0.6 * r + 0.2 * p + 0.2 * o

    raise ValueError("Baseline TSV has no usable score/overall/parts to blend.")

def fisher_enrichment(topk_set, approved_set, universe):
    a = len([d for d in topk_set if d in approved_set])
    b = len(topk_set) - a
    c = len([d for d in universe if d in approved_set]) - a
    d = len(universe) - len(topk_set) - c
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, p

def compute_metrics(df_ranked, approved_set):
    drugs = safe_lower(df_ranked["drug"]).tolist()
    ml = df_ranked["ml_score"].values
    y = np.array([1 if d in approved_set else 0 for d in drugs])

    auroc = np.nan
    if y.sum() > 0 and y.sum() < len(y):
        auroc = roc_auc_score(y, ml)

    res = {}
    for k in (10, 25, 50):
        k_eff = min(k, len(drugs))
        topk = drugs[:k_eff]
        hits = sum(1 for d in topk if d in approved_set)
        _, p = fisher_enrichment(set(topk), approved_set, set(drugs))
        res[f"top{k}_hits"] = hits
        res[f"top{k}_fisher_p"] = p
        res[f"top{k}_any_hit"] = 1 if hits > 0 else 0
    res["auroc"] = auroc
    return res

def load_features(path: str) -> pd.DataFrame:
    # Read Parquet or TSV based on extension
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # Map possible name columns to "drug"
    if "drug" not in df.columns:
        for alt in ("compound", "pert_iname", "drug_name"):
            if alt in df.columns:
                df = df.rename(columns={alt: "drug"})
                break
    if "drug" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} has no 'drug' column.")

    df["drug_lc"] = safe_lower(df["drug"])

    # Force known feature cols to numeric floats; replace infs
    num_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if num_cols:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    return df

def load_baseline(bdir: str, pid: str) -> pd.DataFrame:
    p = os.path.join(bdir, f"baseline_{pid}.tsv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    base = pd.read_csv(p, sep="\t")
    if "drug" not in base.columns:
        raise ValueError(f"{p} missing 'drug' column.")
    base["drug_lc"] = safe_lower(base["drug"])
    base["score_resolved"] = baseline_pick_or_blend(base)
    base["score_resolved"] = pd.to_numeric(base["score_resolved"], errors="coerce")
    return base

def build_model(n_estimators, max_depth, learning_rate, seed):
    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            reg_lambda=1.0,
            reg_alpha=0.0
        )
        pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),  # keep sparse-safety
            ("xgb", model)
        ])
        scorer = lambda M, A: M.predict_proba(A)[:, 1]
        name = "xgboost"
    else:
        model = HistGradientBoostingClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=seed
        )
        pipe = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            ("hgb", model)
        ])
        scorer = lambda M, A: M.predict_proba(A)[:, 1]
        name = "hgb_fallback"
    return pipe, scorer, name

def to_float_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Select cols, coerce to numeric, replace infs with NaN, return float64 ndarray (np.nan only, no pd.NA)."""
    if not cols:
        return np.empty((len(df), 0), dtype=np.float64)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.to_numpy(dtype=np.float64)

def main():
    ap = argparse.ArgumentParser(description="LOPO ML re-ranker (XGB/HGB fallback)")
    ap.add_argument("--features-dir", required=True, help="Folder with features_*.parquet or features_*.tsv")
    ap.add_argument("--baseline-dir", required=True, help="Folder with baseline_*.tsv")
    ap.add_argument("--approved-tsv", required=True, help="TSV with a 'drug' column")
    ap.add_argument("--out-rankings", required=True, help="Folder to write mlrank_*.tsv")
    ap.add_argument("--out-metrics", required=True, help="Folder to write metrics CSVs")
    ap.add_argument("--patients", default="", help="Comma list to restrict; default all")
    ap.add_argument("--topn-decoys", type=int, default=300, help="Negatives per patient to include in training")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_rankings, exist_ok=True)
    os.makedirs(args.out_metrics, exist_ok=True)

    approved = load_approved_set(args.approved_tsv)

    # Accept Parquet first; fallback to TSV
    feat_paths = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not feat_paths:
        feat_paths = sorted(glob.glob(os.path.join(args.features_dir, "features_*.tsv")))
    all_pids = [os.path.basename(p).split(".")[0].replace("features_", "") for p in feat_paths]
    if args.patients.strip():
        keep = set([p.strip() for p in args.patients.split(",") if p.strip()])
        pids = [p for p in all_pids if p in keep]
    else:
        pids = all_pids

    # Preload features & baselines
    features = {}
    baselines = {}
    for pid in pids:
        try:
            features[pid] = load_features(os.path.join(args.features_dir, f"features_{pid}.parquet"))
        except Exception:
            try:
                features[pid] = load_features(os.path.join(args.features_dir, f"features_{pid}.tsv"))
            except Exception as e:
                warnings.warn(f"{pid}: features load failed: {e}")
                continue
        try:
            baselines[pid] = load_baseline(args.baseline_dir, pid)
        except Exception as e:
            warnings.warn(f"{pid}: baseline load failed: {e}")

    # Determine feature columns (union across patients), then prune all-NaN
    union_cols = set()
    for df in features.values():
        union_cols |= set(pick_feature_columns(df))
    feat_cols = [c for c in FEATURE_CANDIDATES if c in union_cols]
    if not feat_cols:
        sys.exit("No usable feature columns across patients.")

    keep_cols = []
    for c in feat_cols:
        any_finite = False
        for df in features.values():
            if c in df.columns:
                s = pd.to_numeric(df[c], errors="coerce").to_numpy()
                if np.isfinite(s).any():
                    any_finite = True
                    break
        if any_finite:
            keep_cols.append(c)
    feat_cols = keep_cols
    if not feat_cols:
        sys.exit("All candidate feature columns are NaN. Check your features.")

    # LOPO training/scoring
    metrics_rows = []

    for holdout in pids:
        if holdout not in features:
            warnings.warn(f"{holdout}: skipped (no features).")
            continue
        if holdout not in baselines:
            warnings.warn(f"{holdout}: skipped (no baseline).")
            continue

        # Build training set from other patients
        X_tr_parts, y_tr_parts = [], []
        for pid in pids:
            if pid == holdout or pid not in features or pid not in baselines:
                continue

            Feat = features[pid]
            Base = baselines[pid]

            # Positives: approved drugs
            pos_mask = Feat["drug_lc"].isin(approved)
            pos_arr = to_float_matrix(Feat.loc[pos_mask], feat_cols)
            n_pos = len(pos_arr)

            # Negatives: top-N non-approved by baseline score
            base_sorted = Base.sort_values("score_resolved", ascending=False)
            neg_drugs = [d for d in base_sorted["drug_lc"].tolist() if d not in approved][:args.topn_decoys]
            neg_arr = to_float_matrix(Feat[Feat["drug_lc"].isin(neg_drugs)], feat_cols)
            n_neg = len(neg_arr)

            if n_pos == 0 or n_neg == 0:
                continue

            X_tr_parts.append(pos_arr)
            y_tr_parts.append(np.ones(n_pos, dtype=int))
            X_tr_parts.append(neg_arr)
            y_tr_parts.append(np.zeros(n_neg, dtype=int))

        if not X_tr_parts:
            warnings.warn(f"{holdout}: skipped (no training data from other patients).")
            continue

        X_train = np.vstack(X_tr_parts).astype(np.float64)
        y_train = np.concatenate(y_tr_parts)

        # Fit model
        pipe, scorer, model_name = build_model(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            seed=args.seed
        )
        pipe.fit(X_train, y_train)

        # Score holdout over all drugs
        F_hold = features[holdout]
        allX = to_float_matrix(F_hold, feat_cols)
        probs = scorer(pipe, allX)

        # Ranked table
        out_df = F_hold.copy()
        out_df["ml_score"] = probs
        out_df = out_df.sort_values("ml_score", ascending=False)

        # Metrics
        m = compute_metrics(out_df[["drug", "ml_score"]], approved)
        m["patient"] = holdout
        m["n_pos"] = int((F_hold["drug_lc"].isin(approved)).sum())
        m["n_cand"] = int(len(F_hold))
        metrics_rows.append(m)

        # Save ranking
        os.makedirs(args.out_rankings, exist_ok=True)
        out_path = os.path.join(args.out_rankings, f"mlrank_{holdout}.tsv")
        keep_out = ["drug", "ml_score"] + [c for c in feat_cols if c in out_df.columns]
        out_df[keep_out].to_csv(out_path, sep="\t", index=False)
        print(f"[{model_name}] wrote {out_path} ({len(out_df)} rows)")

    # Save metrics
    if metrics_rows:
        os.makedirs(args.out_metrics, exist_ok=True)
        per = pd.DataFrame(metrics_rows).sort_values("patient")
        per_path = os.path.join(args.out_metrics, "metrics_per_patient.csv")
        per.to_csv(per_path, index=False)

        summ = {
            "patients": int(len(per)),
            "auroc_mean": float(np.nanmean(per["auroc"])),
            "auroc_median": float(np.nanmedian(per["auroc"])),
        }
        for k in (10, 25, 50):
            summ[f"top{k}_any_hit_mean"] = float(per[f"top{k}_any_hit"].mean())
        summ_df = pd.DataFrame([summ])
        summ_path = os.path.join(args.out_metrics, "metrics_summary.csv")
        summ_df.to_csv(summ_path, index=False)
        print(f"[metrics] saved {per_path} and {summ_path}")
    else:
        print("No patients completed.")

if __name__ == "__main__":
    # Keep warnings visible for easy debugging
    # warnings.filterwarnings("ignore")
    main()
