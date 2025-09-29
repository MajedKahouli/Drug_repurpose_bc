#!/usr/bin/env python3
# LOPO XGBoost re-ranker (enhanced): baseline feature, class weighting, early stopping (version-agnostic),
# optional rank:pairwise objective, Parquet/TSV features, robust NaN handling.

import argparse, os, glob, sys, warnings, math, random, inspect
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

# Try xgboost; fallback to sklearn HGB if absent
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False


# ------------------------------ Helpers ------------------------------

FEATURE_CANDIDATES = [
    # reversal / expression
    "reversal", "reversal_cosine", "reversal_norm", "reversal_score",
    # network proximity
    "prox_z", "proximity_z", "proximity_norm", "prox_raw",
    # overlap / target similarity
    "overlap", "overlap_jaccard", "overlap_dice", "overlap_norm",
    "jaccard", "overlap_count", "weighted_overlap",
    # precomputed overall
    "overall", "blend", "score",
    # added from baseline
    "baseline_feat",
]

BASELINE_SCORE_CANDIDATES = [
    "baseline_score", "score", "overall", "blend"
]

def safe_lower(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def load_approved_set(path: str) -> set:
    df = pd.read_csv(path, sep="\t")
    for c in ["drug","Drug","DRUG","drug_name","compound","pert_iname","name"]:
        if c in df.columns:
            return set(safe_lower(df[c].dropna()))
    return set(safe_lower(df.iloc[:,0].dropna()))

def to_float_matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    """Select cols, coerce to numeric, replace infs with NaN, return float64 ndarray (np.nan only)."""
    if not cols:
        return np.empty((len(df), 0), dtype=np.float64)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.to_numpy(dtype=np.float64)

def fisher_enrichment(topk_set, approved_set, universe):
    a = len([d for d in topk_set if d in approved_set])
    b = len(topk_set) - a
    c = len([d for d in universe if d in approved_set]) - a
    d = len(universe) - len(topk_set) - c
    _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, p

def compute_metrics(df_ranked: pd.DataFrame, approved: set) -> dict:
    drugs = safe_lower(df_ranked["drug"]).tolist()
    scores = pd.to_numeric(df_ranked["ml_score"], errors="coerce").to_numpy(float)
    y = np.array([1 if d in approved else 0 for d in drugs], dtype=int)

    out = {}
    out["auroc"] = roc_auc_score(y, scores) if (y.sum()>0 and y.sum()<len(y)) else np.nan
    U = set(drugs)
    for k in (10, 25, 50):
        k_eff = min(k, len(drugs))
        topk = drugs[:k_eff]
        hits, p = fisher_enrichment(set(topk), approved, U)
        out[f"top{k}_hits"] = int(hits)
        out[f"top{k}_any_hit"] = int(hits > 0)
        out[f"top{k}_fisher_p"] = float(p)
    return out

def pick_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_CANDIDATES if c in df.columns]

def baseline_pick_or_blend(B: pd.DataFrame) -> pd.Series:
    for c in BASELINE_SCORE_CANDIDATES:
        if c in B.columns:
            return pd.to_numeric(B[c], errors="coerce")
    # fallback blend (0.6/0.2/0.2) if parts exist
    r = B.get("reversal_norm", B.get("reversal", B.get("reversal_score")))
    p = B.get("proximity_norm", B.get("prox_z", B.get("proximity_z")))
    o = B.get("overlap_norm", B.get("overlap_jaccard", B.get("jaccard")))
    if r is not None and p is not None and o is not None:
        def mm(s):
            s = pd.to_numeric(s, errors="coerce")
            lo, hi = np.nanmin(s.values), np.nanmax(s.values)
            return (s - lo)/(hi - lo) if (hi > lo) else s.fillna(0.0)
        return 0.6*mm(r) + 0.2*mm(p) + 0.2*mm(o)
    raise ValueError("Baseline missing usable score columns")

def load_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep="\t")
    if "drug" not in df.columns:
        for alt in ("compound","pert_iname","drug_name","name"):
            if alt in df.columns:
                df = df.rename(columns={alt:"drug"}); break
    if "drug" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} has no 'drug' column")
    df["drug_lc"] = safe_lower(df["drug"])
    # numeric-ize known features
    for c in [c for c in FEATURE_CANDIDATES if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_baseline(bdir: str, pid: str) -> pd.DataFrame:
    p = os.path.join(bdir, f"baseline_{pid}.tsv")
    if not os.path.exists(p): raise FileNotFoundError(p)
    B = pd.read_csv(p, sep="\t")
    if "drug" not in B.columns: raise ValueError(f"{p} missing 'drug'")
    B["drug_lc"] = safe_lower(B["drug"])
    B["score_resolved"] = baseline_pick_or_blend(B)
    return B[["drug","drug_lc","score_resolved"]]

def train_val_split(donor_pids: list, val_frac: float, seed: int):
    if len(donor_pids) <= 2 or val_frac <= 0:
        return donor_pids, []
    rnd = random.Random(seed)
    order = donor_pids[:]
    rnd.shuffle(order)
    n_val = max(1, int(math.ceil(len(order) * val_frac)))
    val = sorted(order[:n_val])
    trn = sorted(order[n_val:])
    return trn, val


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="LOPO XGBoost re-ranker (baseline feature + class weight + early stopping; optional rank:pairwise)"
    )
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-rankings", required=True)
    ap.add_argument("--out-metrics", required=True)
    ap.add_argument("--patients", default="", help="Comma list to restrict; default all")
    ap.add_argument("--topn-decoys", type=int, default=300, help="Negatives per donor patient")
    ap.add_argument("--objective", choices=["binary","rankpairwise"], default="binary",
                    help="binary=AUROC focus; rankpairwise=Top-K focus")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.12, help="Fraction of donor patients used for early stopping")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.out_rankings, exist_ok=True)
    os.makedirs(args.out_metrics, exist_ok=True)

    approved = load_approved_set(args.approved_tsv)

    # discover patients
    feats = sorted(glob.glob(os.path.join(args.features_dir, "features_*.parquet")))
    if not feats:
        feats = sorted(glob.glob(os.path.join(args.features_dir, "features_*.tsv")))
    all_pids = [os.path.basename(p).split(".")[0].replace("features_","") for p in feats]
    if args.patients.strip():
        keep = set([p.strip() for p in args.patients.split(",") if p.strip()])
        pids = [p for p in all_pids if p in keep]
    else:
        pids = all_pids

    # preload data
    features, baselines = {}, {}
    for pid in pids:
        # features
        f_parq = os.path.join(args.features_dir, f"features_{pid}.parquet")
        f_tsv  = os.path.join(args.features_dir, f"features_{pid}.tsv")
        try:
            features[pid] = load_features(f_parq if os.path.exists(f_parq) else f_tsv)
        except Exception as e:
            warnings.warn(f"{pid}: features load failed: {e}")
            continue
        # baseline
        try:
            baselines[pid] = load_baseline(args.baseline_dir, pid)
        except Exception as e:
            warnings.warn(f"{pid}: baseline load failed: {e}")

    # determine usable feature columns (union)
    union = set()
    for df in features.values():
        union |= set(pick_feature_columns(df))
    feat_cols = [c for c in FEATURE_CANDIDATES if c in union]
    # (we'll add baseline_feat per-patient after merging baseline)

    metrics_rows = []

    for holdout in pids:
        if holdout not in features or holdout not in baselines:
            warnings.warn(f"{holdout}: skipped (missing features or baseline)")
            continue

        # donor patients (all others)
        donors = [p for p in pids if p != holdout and p in features and p in baselines]
        if not donors:
            warnings.warn(f"{holdout}: skipped (no donors)")
            continue
        trn_pids, val_pids = train_val_split(donors, args.val_frac, args.seed)

        # assemble train/val matrices at patient level
        def build_arrays(pid_list):
            X_parts, y_parts, groups = [], [], []
            for pid in pid_list:
                Feat = features[pid].copy()
                Base = baselines[pid]
                # merge baseline score as feature
                Feat = Feat.merge(Base[["drug_lc","score_resolved"]],
                                  on="drug_lc", how="left").rename(columns={"score_resolved":"baseline_feat"})
                # per-patient usable feature columns (include baseline_feat if present)
                cols = [c for c in FEATURE_CANDIDATES if c in Feat.columns]
                if "baseline_feat" in Feat.columns and "baseline_feat" not in cols:
                    cols.append("baseline_feat")

                # positives: approved drugs for this donor
                pos_mask = Feat["drug_lc"].isin(approved)
                pos = Feat.loc[pos_mask, cols]
                # negatives: top-N non-approved from baseline
                base_sorted = Base.sort_values("score_resolved", ascending=False)
                neg_drugs = [d for d in base_sorted["drug_lc"].tolist() if d not in approved][:args.topn_decoys]
                neg = Feat.loc[Feat["drug_lc"].isin(neg_drugs), cols]

                if len(pos) == 0 or len(neg) == 0:
                    continue

                Xi = to_float_matrix(pd.concat([pos, neg], axis=0, ignore_index=True), cols)
                yi = np.concatenate([np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)])
                X_parts.append(Xi); y_parts.append(yi); groups.append(len(yi))
            if X_parts:
                X = np.vstack(X_parts).astype(np.float64)
                y = np.concatenate(y_parts)
            else:
                X = np.empty((0, len(feat_cols)), dtype=np.float64); y = np.array([], dtype=int)
            return X, y, groups

        X_tr, y_tr, grp_tr = build_arrays(trn_pids)
        X_va, y_va, grp_va = build_arrays(val_pids) if val_pids else (None, None, None)

        total_pos = int((y_tr == 1).sum())
        total_neg = int((y_tr == 0).sum())
        if len(y_tr) == 0 or total_pos == 0 or total_neg == 0:
            warnings.warn(f"{holdout}: skipped (train has no pos/neg)")
            continue

        # ---------------- fit model ----------------
        model_name = "xgboost"
        if HAS_XGB:
            if args.objective == "rankpairwise":
                ranker = xgb.XGBRanker(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    objective="rank:pairwise",
                    random_state=args.seed,
                    reg_lambda=1.0, reg_alpha=0.0,
                )
                fit_sig = inspect.signature(ranker.fit)
                has_esr = "early_stopping_rounds" in fit_sig.parameters
                has_callbacks = "callbacks" in fit_sig.parameters and hasattr(xgb, "callback") and hasattr(xgb.callback, "EarlyStopping")

                if X_va is not None and len(y_va):
                    if has_esr:
                        ranker.fit(X_tr, y_tr, group=grp_tr,
                                   eval_set=[(X_va, y_va)], eval_group=[grp_va],
                                   early_stopping_rounds=args.early_stopping_rounds)
                    elif has_callbacks:
                        ranker.fit(X_tr, y_tr, group=grp_tr,
                                   eval_set=[(X_va, y_va)], eval_group=[grp_va],
                                   callbacks=[xgb.callback.EarlyStopping(
                                       rounds=args.early_stopping_rounds, save_best=True, maximize=False
                                   )])
                    else:
                        ranker.fit(X_tr, y_tr, group=grp_tr,
                                   eval_set=[(X_va, y_va)], eval_group=[grp_va])
                else:
                    ranker.fit(X_tr, y_tr, group=grp_tr)

                model = ranker
                predict_fn = lambda X: model.predict(X)
                model_name = "xgboost_rank"

            else:  # binary classification
                spw = max(1.0, total_neg / max(1, total_pos))
                clf = xgb.XGBClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    subsample=args.subsample,
                    colsample_bytree=args.colsample_bytree,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=args.seed,
                    reg_lambda=1.0, reg_alpha=0.0,
                    scale_pos_weight=spw,
                )
                fit_sig = inspect.signature(clf.fit)
                has_esr = "early_stopping_rounds" in fit_sig.parameters
                has_callbacks = "callbacks" in fit_sig.parameters and hasattr(xgb, "callback") and hasattr(xgb.callback, "EarlyStopping")

                if X_va is not None and len(y_va):
                    if has_esr:
                        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=args.early_stopping_rounds)
                    elif has_callbacks:
                        clf.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            callbacks=[xgb.callback.EarlyStopping(
                                rounds=args.early_stopping_rounds, save_best=True, maximize=False
                            )]
                        )
                    else:
                        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
                else:
                    clf.fit(X_tr, y_tr)

                model = clf
                predict_fn = lambda X: model.predict_proba(X)[:, 1]
                model_name = f"xgboost_bin(spw={spw:.1f})"

        elif HAS_HGB:
            clf = HistGradientBoostingClassifier(
                max_depth=args.max_depth,
                learning_rate=args.learning_rate,
                max_iter=args.n_estimators,
                random_state=args.seed,
            )
            clf.fit(X_tr, y_tr)
            model = clf
            predict_fn = lambda X: model.predict_proba(X)[:,1]
            model_name = "hgb_fallback"
        else:
            sys.exit("Neither xgboost nor sklearn HGB available.")

        # ---------------- score holdout ----------------
        F_hold = features[holdout].copy()
        # add baseline feature to holdout too
        F_hold = F_hold.merge(baselines[holdout][["drug_lc","score_resolved"]],
                              on="drug_lc", how="left").rename(columns={"score_resolved":"baseline_feat"})
        use_cols = [c for c in FEATURE_CANDIDATES if c in F_hold.columns]
        X_all = to_float_matrix(F_hold, use_cols)
        probs = predict_fn(X_all)

        out = F_hold.copy()
        out["ml_score"] = probs
        out = out.sort_values("ml_score", ascending=False)
        out_path = os.path.join(args.out_rankings, f"mlrank_{holdout}.tsv")
        out[["drug","ml_score"]].to_csv(out_path, sep="\t", index=False)
        print(f"[{model_name}] wrote {out_path} ({len(out)} rows)")

        # metrics
        m = compute_metrics(out[["drug","ml_score"]], approved)
        m["patient"] = holdout
        m["n_pos"] = int((F_hold["drug_lc"].isin(approved)).sum())
        m["n_cand"] = int(len(F_hold))
        metrics_rows.append(m)

    # ---------------- save metrics ----------------
    if metrics_rows:
        per = pd.DataFrame(metrics_rows).sort_values("patient")
        per_path = os.path.join(args.out_metrics, "metrics_per_patient.csv")
        per.to_csv(per_path, index=False)

        summ = {"patients": int(len(per)),
                "auroc_mean": float(np.nanmean(per["auroc"])),
                "auroc_median": float(np.nanmedian(per["auroc"]))}
        for k in (10,25,50):
            summ[f"top{k}_any_hit_mean"] = float(np.nanmean(per[f"top{k}_any_hit"]))
        summ_path = os.path.join(args.out_metrics, "metrics_summary.csv")
        pd.DataFrame([summ]).to_csv(summ_path, index=False)
        print(f"[metrics] saved {per_path} and {summ_path}")
    else:
        print("No patients completed.")

if __name__ == "__main__":
    # Keep warnings visible for easy debugging
    # warnings.filterwarnings("ignore")
    main()
