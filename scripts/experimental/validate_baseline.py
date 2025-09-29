#!/usr/bin/env python
import argparse, os, glob, sys, json
import pandas as pd
import numpy as np

# Optional deps: if missing, we’ll skip those stats gracefully
try:
    from scipy.stats import fisher_exact
except Exception:
    fisher_exact = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None

def norm_name(x):
    if pd.isna(x): return None
    return str(x).strip().lower()

def load_approved(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    # pick a sensible column
    for col in ["drug", "drug_name", "name"]:
        if col in df.columns:
            return set(df[col].dropna().map(norm_name))
    # if no expected column, just take first col
    return set(df.iloc[:,0].dropna().map(norm_name))

def pick_score_col(df):
    # try blended first; otherwise any plausible score column
    candidates = ["blended", "score", "total_score", "final", "rank_score"]
    for c in candidates:
        if c in df.columns:
            return c
    # last resort: take the last numeric column
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if len(num_cols)==0:
        raise ValueError("No numeric score column found in ranking file.")
    return num_cols[-1]

def evaluate_one(file_path, approved_set, ks=(10,25,50), n_perm=0):
    pid = os.path.basename(file_path).replace("baseline_","").replace(".tsv","")
    df = pd.read_csv(file_path, sep="\t")
    # normalize names
    if "drug" not in df.columns:
        # try case-insensitive
        match = [c for c in df.columns if c.strip().lower()=="drug"]
        if match:
            df.rename(columns={match[0]:"drug"}, inplace=True)
        else:
            raise ValueError(f"{file_path}: no 'drug' column.")
    df["__drug_norm"] = df["drug"].map(norm_name)

    score_col = pick_score_col(df)
    # Higher score == better rank (your blender is higher-is-better)
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df)+1)

    # labels & scores for AUROC
    y = df["__drug_norm"].map(lambda d: 1 if d in approved_set else 0).values
    s = df[score_col].values

    # metrics
    out = {
        "patient": pid,
        "n_drugs": len(df),
        "n_approved_in_list": int(y.sum()),
    }

    # top-k recovery & enrichment
    total_approved = int(y.sum())
    total_non = len(y) - total_approved
    for k in ks:
        y_top = y[:k]
        a_top = int(y_top.sum())
        out[f"top{k}_any"] = int(a_top > 0)
        out[f"top{k}_approved"] = a_top

        if fisher_exact is not None:
            a_rest = total_approved - a_top
            non_top = k - a_top
            non_rest = total_non - non_top
            # 2x2: [[approved_top, approved_rest], [non_top, non_rest]]
            table = np.array([[a_top, a_rest],[non_top, non_rest]], dtype=int)
            # greater = enrichment of approved in top-k
            try:
                _, p = fisher_exact(table, alternative="greater")
            except Exception:
                p = np.nan
            out[f"top{k}_fisher_p"] = p
        else:
            out[f"top{k}_fisher_p"] = np.nan

    # AUROC
    if roc_auc_score is not None and (y.sum()>0) and (y.sum()<len(y)):
        try:
            auroc = roc_auc_score(y, s)
        except Exception:
            auroc = np.nan
    else:
        auroc = np.nan
    out["auroc"] = auroc

    # permutation p-value for AUROC (optional, n_perm=0 to skip)
    if n_perm and roc_auc_score is not None and np.isfinite(auroc):
        rng = np.random.default_rng(42)
        perms = []
        y_copy = y.copy()
        for _ in range(n_perm):
            rng.shuffle(y_copy)
            try:
                perms.append(roc_auc_score(y_copy, s))
            except Exception:
                perms.append(np.nan)
        perms = np.array(perms)
        # one-sided: how often perm AUROC >= observed
        if np.isfinite(perms).any():
            p_emp = (np.sum(perms >= auroc) + 1.0) / (np.sum(np.isfinite(perms)) + 1.0)
        else:
            p_emp = np.nan
        out["auroc_perm_p"] = p_emp
    else:
        out["auroc_perm_p"] = np.nan

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings-dir", required=True, help="Folder with baseline_*.tsv")
    ap.add_argument("--approved-tsv",   required=True, help="TSV with approved drugs (col: drug/drug_name/name)")
    ap.add_argument("--out",            required=True, help="Output folder (will be created)")
    ap.add_argument("--ks", default="10,25,50", help="Comma-separated top-k values")
    ap.add_argument("--n-perm", type=int, default=0, help="Permutations for AUROC p-value (0 to skip)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ks = tuple(int(x) for x in args.ks.split(","))
    approved = load_approved(args.approved_tsv)

    files = sorted(glob.glob(os.path.join(args.rankings_dir, "baseline_*.tsv")))
    if not files:
        print(f"No files matched {args.rankings_dir}/baseline_*.tsv", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        try:
            rows.append(evaluate_one(fp, approved, ks=ks, n_perm=args.n_perm))
        except Exception as e:
            print(f"[WARN] Failed on {fp}: {e}", file=sys.stderr)

    res = pd.DataFrame(rows).sort_values("patient")
    res.to_csv(os.path.join(args.out, "baseline_metrics_per_patient.csv"), index=False)

    # aggregate summary
    agg = {
        "n_patients": len(res),
        "mean_auroc": float(res["auroc"].dropna().mean()) if "auroc" in res else np.nan,
    }
    for k in ks:
        agg[f"top{k}_recovery_rate"] = float(res[f"top{k}_any"].mean())
        agg[f"top{k}_mean_approved"] = float(res[f"top{k}_approved"].mean())
        if f"top{k}_fisher_p" in res:
            agg[f"top{k}_median_fisher_p"] = float(res[f"top{k}_fisher_p"].median())

    pd.DataFrame([agg]).to_csv(os.path.join(args.out, "baseline_metrics_summary.csv"), index=False)

    print(f"Wrote {os.path.join(args.out, 'baseline_metrics_per_patient.csv')}")
    print(f"Wrote {os.path.join(args.out, 'baseline_metrics_summary.csv')}")
    print(json.dumps(agg, indent=2))
    
if __name__ == "__main__":
    main()
