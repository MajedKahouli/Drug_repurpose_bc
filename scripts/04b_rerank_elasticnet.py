#!/usr/bin/env python3
import argparse, os, glob, re, warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURE_CHOICES = {"overlap_count", "jaccard", "weighted_overlap"}

def infer_patient_id(path):
    m = re.search(r"baseline_(.+)\.tsv$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot infer patient ID from {path}")
    return m.group(1)

def load_approved_set(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    # be forgiving about column naming
    cols = {c.lower(): c for c in df.columns}
    drug_col = cols.get("drug") or cols.get("name") or cols.get("compound")
    if not drug_col:
        raise ValueError(f"{tsv_path} must contain a 'drug' column")
    return set(df[drug_col].astype(str).str.strip().str.lower().unique())

def load_baseline(path, overlap_feature):
    df = pd.read_csv(path, sep="\t")
    needed = {"drug", "reversal_score", overlap_feature, "prox_z", "baseline_score"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    # normalize types / names
    df = df.copy()
    df["drug"] = df["drug"].astype(str)
    df["label"] = 0  # placeholder until we tag positives per approved set
    return df

def topk_recovery(y_true, y_score, ks=(10, 25, 50)):
    out = {}
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    for k in ks:
        k = min(k, len(y_sorted))
        out[f"top{k}_any"] = float((y_sorted[:k].sum() > 0))
    return out

def enrichment_at_k(y_true, y_score, k):
    # 2x2 enrichment among top-k vs rest; return odds-ratio-ish p-proxy not exact Fisher here
    # We’ll compute simple precision@k and prevalence for a quick, interpretable number
    k = min(k, len(y_true))
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    pos_at_k = int(y_sorted[:k].sum())
    prec_at_k = pos_at_k / max(k, 1)
    prevalence = float(np.mean(y_true))
    return {"prec@%d" % k: prec_at_k, "prevalence": prevalence}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True, help="folder with baseline_*.tsv")
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-rankings", required=True)
    ap.add_argument("--out-metrics", required=True)
    ap.add_argument("--overlap-feature", default="jaccard", choices=list(FEATURE_CHOICES))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--l1_ratio", type=float, default=0.3)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out_rankings, exist_ok=True)

    approved = load_approved_set(args.approved_tsv)

    # discover patients from baseline files
    base_files = sorted(glob.glob(os.path.join(args.baseline_dir, "baseline_*.tsv")))
    if not base_files:
        raise SystemExit(f"No baseline_*.tsv found in {args.baseline_dir}")

    # load all baseline tables first (once)
    per_patient = {}
    for p in base_files:
        pid = infer_patient_id(p)
        try:
            df = load_baseline(p, args.overlap_feature)
        except Exception as e:
            warnings.warn(f"{pid}: baseline load failed: {e}")
            continue
        # make labels
        df["label"] = df["drug"].str.strip().str.lower().isin(approved).astype(int)
        per_patient[pid] = df

    if not per_patient:
        warnings.warn("No patients had valid baselines. Nothing to do.")
        return

    # model pipeline (handles NaNs)
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            C=args.C,
            l1_ratio=args.l1_ratio,
            max_iter=args.max_iter,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    metrics_rows = []
    pids = list(per_patient.keys())

    # leave-one-patient-out
    for holdout in pids:
        # build training set from every other patient
        X_tr, y_tr = [], []
        for pid, df in per_patient.items():
            if pid == holdout:
                continue
            X_tr.append(df[["baseline_score", "reversal_score", args.overlap_feature, "prox_z"]].to_numpy(dtype=float))
            y_tr.append(df["label"].to_numpy(dtype=int))
        if not X_tr:
            warnings.warn(f"{holdout}: skipped (no training data).")
            continue
        X_tr = np.vstack(X_tr)
        y_tr = np.concatenate(y_tr)

        # fit
        try:
            pipe.fit(X_tr, y_tr)
        except Exception as e:
            warnings.warn(f"{holdout}: training failed with error: {e}")
            continue

        # score holdout
        df_ho = per_patient[holdout].copy()
        X_ho = df_ho[["baseline_score", "reversal_score", args.overlap_feature, "prox_z"]].to_numpy(dtype=float)
        proba = pipe.predict_proba(X_ho)[:, 1]
        df_ho["ml_score"] = proba

        # write re-ranked
        out_path = os.path.join(args.out_rankings, f"reranked_elasticnet_{holdout}.tsv")
        df_out = df_ho[["drug", "baseline_score", "reversal_score", args.overlap_feature, "prox_z", "ml_score", "label"]].sort_values("ml_score", ascending=False)
        df_out.to_csv(out_path, sep="\t", index=False)

        # metrics on holdout
        y_true = df_ho["label"].to_numpy(int)
        y_score = proba
        row = {"patient": holdout}
        # AUROC (only if both classes present)
        if (y_true.sum() > 0) and (y_true.sum() < len(y_true)):
            try:
                row["auroc"] = float(roc_auc_score(y_true, y_score))
            except Exception as e:
                row["auroc"] = np.nan
                warnings.warn(f"{holdout}: AUROC failed: {e}")
        else:
            row["auroc"] = np.nan
            warnings.warn(f"{holdout}: AUROC skipped (only one class present).")

        row.update(topk_recovery(y_true, y_score, ks=(10, 25, 50)))
        row.update(enrichment_at_k(y_true, y_score, k=25))
        metrics_rows.append(row)

        print(f"✓ {holdout}: wrote {out_path}")

    if metrics_rows:
        mdf = pd.DataFrame(metrics_rows)
        mdf.to_csv(args.out_metrics, index=False)
        print(f"Wrote metrics to {args.out_metrics}")
        # quick summary to stdout
        with pd.option_context("display.max_columns", None):
            print(mdf.describe(include="all"))
    else:
        print("No metrics to write (no patients processed).")

if __name__ == "__main__":
    main()
