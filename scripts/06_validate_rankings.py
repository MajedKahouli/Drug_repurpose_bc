#!/usr/bin/env python
import os, glob, argparse, math, random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact

def load_approved(path):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip() and not line.startswith("#"))

def norm_name(x):
    return str(x).strip().lower()

def eval_file(tsv_path, approved_set, ks=(10,25,50)):
    df = pd.read_csv(tsv_path, sep="\t")
    # Expect df has columns: drug (or compound), score (higher = better ranking)
    # Try to infer columns robustly:
    drug_col = "drug" if "drug" in df.columns else ("compound" if "compound" in df.columns else df.columns[0])
    score_col = "score" if "score" in df.columns else ("blend" if "blend" in df.columns else df.columns[1])

    df = df[[drug_col, score_col]].copy()
    df[drug_col] = df[drug_col].map(norm_name)
    df = df.dropna(subset=[drug_col, score_col])
    df = df.drop_duplicates(subset=[drug_col], keep="first")

    # label: approved vs not
    df["label"] = df[drug_col].isin(approved_set).astype(int)
    # sort by score desc
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    # AUROC (handle degenerate cases)
    y_true = df["label"].values
    y_score = df[score_col].values
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        auroc = np.nan
    else:
        auroc = roc_auc_score(y_true, y_score)

    # Enrichment & top-k recovery
    results = {"auroc": auroc}
    N = len(df)
    A_total = int(y_true.sum())  # total approved in list
    for k in ks:
        k = min(k, N)
        topk = df.iloc[:k]
        A_top = int(topk["label"].sum())
        # Fisher's exact: approved/non-approved in top-k vs remaining
        table = np.array([[A_top, k - A_top],
                          [A_total - A_top, (N - k) - (A_total - A_top)]])
        try:
            _, pval = fisher_exact(table, alternative="greater")
        except Exception:
            pval = np.nan

        results[f"top{k}_approved"] = A_top
        results[f"top{k}_p_fisher"] = pval
        results[f"top{k}_hit"] = 1 if A_top > 0 else 0
    return results, df

def permute_empirical(df, approved_set, ks, n_perm=0, score_col="score"):
    # Build labels once
    labels = df[df.columns[0]].map(norm_name).isin(approved_set).astype(int).values
    N = len(df)
    out = {f"perm_top{k}_p": np.nan for k in ks}
    out["perm_auroc_p"] = np.nan
    if n_perm <= 0:
        return out

    # observed
    try:
        obs_auroc = roc_auc_score(labels, df[score_col].values)
    except Exception:
        obs_auroc = np.nan

    # precompute observed top-k approved
    obs_top = {}
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    for k in ks:
        k = min(k, N)
        obs_top[k] = int(df_sorted.iloc[:k, :][df_sorted.columns[0]].map(norm_name).isin(approved_set).sum())

    # permutations of labels (drug names fixed, shuffle labels)
    more_extreme_auroc = 0
    more_extreme_top = {k: 0 for k in ks}
    for _ in range(n_perm):
        perm_labels = np.random.permutation(labels)
        # auroc
        try:
            p_auroc = roc_auc_score(perm_labels, df[score_col].values)
        except Exception:
            p_auroc = np.nan
        if (not math.isnan(obs_auroc)) and (not math.isnan(p_auroc)) and (p_auroc >= obs_auroc):
            more_extreme_auroc += 1
        # top-k
        # since scores unchanged, top-k set stays same; just recompute approved count under permuted labels
        perm_top = {}
        for k in ks:
            k = min(k, N)
            perm_top[k] = int(perm_labels[:k].sum())  # because df_sorted’s first k entries correspond to top-k drugs
            if perm_top[k] >= obs_top[k]:
                more_extreme_top[k] += 1

    # +1 smoothing
    denom = n_perm + 1
    if not math.isnan(obs_auroc):
        out["perm_auroc_p"] = (more_extreme_auroc + 1) / denom
    for k in ks:
        out[f"perm_top{k}_p"] = (more_extreme_top[k] + 1) / denom
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings-dir", required=True, help="Folder with baseline_*.tsv from 04a")
    ap.add_argument("--approved-file", required=True, help="Text file listing approved drug names (one per line)")
    ap.add_argument("--ks", nargs="+", type=int, default=[10,25,50])
    ap.add_argument("--permutations", type=int, default=0, help="Number of label permutations for empirical p-values (0 to skip)")
    ap.add_argument("--out-dir", required=True, help="Folder to write summary CSVs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    approved = load_approved(args.approved_file)
    files = sorted(glob.glob(os.path.join(args.rankings_dir, "baseline_*.tsv")))
    if not files:
        raise SystemExit(f"No ranking files found in {args.rankings_dir}")

    per_patient = []
    for fp in files:
        pid = os.path.splitext(os.path.basename(fp))[0].replace("baseline_","")
        metrics, df = eval_file(fp, approved, ks=args.ks)

        # optional permutations for empirical p
        perm = {}
        try:
            # try to infer score column again (same logic)
            cols = df.columns.tolist()
            drug_col = cols[0]
            score_col = "score" if "score" in df.columns else ("blend" if "blend" in df.columns else cols[1])
            perm = permute_empirical(df[[drug_col, score_col]], approved, args.ks, n_perm=args.permutations, score_col=score_col)
        except Exception:
            perm = {f"perm_top{k}_p": np.nan for k in args.ks}
            perm["perm_auroc_p"] = np.nan

        row = {"patient": pid, **metrics, **perm}
        per_patient.append(row)

    per_patient_df = pd.DataFrame(per_patient)
    # Aggregate summaries
    agg = {}
    agg["n_patients"] = len(per_patient_df)
    # top-k hit rates
    for k in args.ks:
        agg[f"top{k}_hit_rate"] = float(np.nanmean(per_patient_df[f"top{k}_hit"]))
        agg[f"top{k}_median_approved_in_topk"] = float(np.nanmedian(per_patient_df[f"top{k}_approved"]))
        agg[f"top{k}_median_fisher_p"] = float(np.nanmedian(per_patient_df[f"top{k}_p_fisher"]))
        if args.permutations > 0:
            agg[f"top{k}_median_perm_p"] = float(np.nanmedian(per_patient_df[f"perm_top{k}_p"]))
    # AUROC
    agg["auroc_median"] = float(np.nanmedian(per_patient_df["auroc"]))
    if args.permutations > 0:
        agg["auroc_median_perm_p"] = float(np.nanmedian(per_patient_df["perm_auroc_p"]))

    # Save
    per_patient_df.to_csv(os.path.join(args.out_dir, "baseline_validation_per_patient.csv"), index=False)
    pd.DataFrame([agg]).to_csv(os.path.join(args.out_dir, "baseline_validation_summary.csv"), index=False)

    print("\nSaved:")
    print(" -", os.path.join(args.out_dir, "baseline_validation_per_patient.csv"))
    print(" -", os.path.join(args.out_dir, "baseline_validation_summary.csv"))
    print("\nQuick summary:")
    for k in args.ks:
        print(f" top-{k} hit rate:", f"{agg[f'top{k}_hit_rate']:.3f}")
    print(" median AUROC:", f"{agg['auroc_median']:.3f}")
