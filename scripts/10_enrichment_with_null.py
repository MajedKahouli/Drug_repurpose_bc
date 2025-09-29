#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd
from math import comb

def norm(s): return str(s).strip().lower()

def expected_anyhit(N, m, K):
    # 1 - C(N-m, K)/C(N, K)
    if K> N: K=N
    return 1.0 - (comb(N-m, K) / comb(N, K)) if N>0 and N>=K and (N-m)>=0 else np.nan

def eval_dir(rank_dir, approved_tsv, pattern="*.tsv", ks=(10,25,50), perms=1000, seed=13, out_dir=None):
    appr = pd.read_csv(approved_tsv, sep="\t")
    for c in ["drug","Drug","drug_name","compound","pert_iname","name"]:
        if c in appr.columns:
            approved = set(appr[c].dropna().map(norm)); break
    else:
        approved = set(appr.iloc[:,0].dropna().map(norm))

    files = sorted(glob.glob(os.path.join(rank_dir, pattern)))
    rows = []
    rng = np.random.default_rng(seed)

    # collect per-patient stats + permutation cohort means
    perm_means = {k: [] for k in ks}
    for fp in files:
        df = pd.read_csv(fp, sep="\t")
        if "drug" not in df.columns or "ml_score" not in df.columns: continue
        df = df.sort_values("ml_score", ascending=False)
        drugs = df["drug"].map(norm).tolist()
        y = np.array([1 if d in approved else 0 for d in drugs], dtype=int)
        N = len(y); m = int(y.sum())

        rec = {"patient": os.path.basename(fp).split(".")[0], "N": N, "m": m}
        for k in ks:
            k_eff = min(k, N)
            any_hit = int(y[:k_eff].sum() > 0)
            exp = expected_anyhit(N, m, k_eff)
            rec[f"top{k}_any_hit"] = any_hit
            rec[f"top{k}_exp_random"] = exp
            rec[f"top{k}_enrichment"] = (any_hit / exp) if exp and exp>0 else np.nan
        rows.append(rec)

    per = pd.DataFrame(rows).sort_values("patient")
    summ = {"patients": int(len(per))}
    for k in ks:
        obs_mean = per[f"top{k}_any_hit"].mean()
        exp_mean = per[f"top{k}_exp_random"].mean()
        summ[f"top{k}_any_hit_mean"] = float(obs_mean)
        summ[f"top{k}_exp_random_mean"] = float(exp_mean)
        summ[f"top{k}_enrichment_mean"] = float(obs_mean/exp_mean) if exp_mean>0 else np.nan

    # permutation null for cohort mean (preserving N and m per patient)
    for b in range(perms):
        means = {}
        for k in ks: means[k]=[]
        for _, r in per.iterrows():
            N, m = int(r["N"]), int(r["m"])
            if N==0 or m==0: 
                for k in ks: means[k].append(0.0); 
                continue
            pos = np.array([1]*m + [0]*(N-m))
            rng.shuffle(pos)
            for k in ks:
                ke = min(int(k), N)
                means[k].append(1.0 if pos[:ke].sum() > 0 else 0.0)
        for k in ks:
            perm_means[k].append(np.mean(means[k]))

    for k in ks:
        lo, hi = np.percentile(perm_means[k], [2.5, 97.5])
        summ[f"top{k}_perm_CI95_lo"] = float(lo)
        summ[f"top{k}_perm_CI95_hi"] = float(hi)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        per.to_csv(os.path.join(out_dir, "enrichment_per_patient.csv"), index=False)
        pd.DataFrame([summ]).to_csv(os.path.join(out_dir, "enrichment_summary.csv"), index=False)
    return per, summ

def main():
    ap = argparse.ArgumentParser(description="Enrichment@K + permutation null bands.")
    ap.add_argument("--rankings-dir", required=True)
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pattern", default="*.tsv")
    ap.add_argument("--perms", type=int, default=1000)
    args = ap.parse_args()
    eval_dir(args.rankings_dir, args.approved_tsv, args.pattern, (10,25,50), args.perms, 13, args.out_dir)
    print("Saved enrichment_per_patient.csv and enrichment_summary.csv")
if __name__ == "__main__":
    main()
