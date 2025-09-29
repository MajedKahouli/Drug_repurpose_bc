import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import zscore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)   # folder with reversal_*.tsv and proximity_*.tsv
    ap.add_argument("--alpha", type=float, default=0.5)  # weight on reversal
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rev_files = sorted(Path(args.scores_dir).glob("reversal_scores_*.tsv"))
    for rf in rev_files:
        pid = rf.stem.replace("reversal_scores_","")
        pf = Path(args.scores_dir) / f"proximity_{pid}.tsv"
        if not pf.exists():
            print(f"[WARN] missing proximity for {pid}, skipping"); continue
        R = pd.read_csv(rf, sep="\t")
        P = pd.read_csv(pf, sep="\t")
        df = R.merge(P[["drug","proximity_score"]], on="drug", how="inner")

        # z-score within patient so scales are comparable
        df["Z_rev"] = zscore(df["reversal_score"].values, nan_policy="omit")
        df["Z_prox"] = zscore(df["proximity_score"].values, nan_policy="omit")
        alpha = args.alpha
        df["final_score"] = alpha*df["Z_rev"] + (1-alpha)*df["Z_prox"]
        df = df.sort_values("final_score", ascending=False)
        df.to_csv(outdir / f"final_rank_{pid}.tsv", sep="\t", index=False)
        print(f"[OK] {pid}: {df.shape[0]} drugs ranked")

if __name__ == "__main__":
    main()
