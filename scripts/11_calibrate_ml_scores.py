#!/usr/bin/env python3
import argparse, os, glob, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

def norm(s): return str(s).strip().lower()

def load_approved(tsv):
    df = pd.read_csv(tsv, sep="\t")
    for c in ["drug","Drug","drug_name","compound","pert_iname","name"]:
        if c in df.columns: return set(df[c].dropna().map(norm))
    return set(df.iloc[:,0].dropna().map(norm))

def main():
    ap = argparse.ArgumentParser(description="Calibrate ML scores per holdout using donors.")
    ap.add_argument("--ml-dir", required=True, help="Folder with reranked_*.tsv (Elastic-net or XGB).")
    ap.add_argument("--approved-tsv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", choices=["isotonic","platt"], default="isotonic")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.ml_dir, "*.tsv")))
    pids = [os.path.basename(f).split(".")[0].split("_")[-1] for f in files]
    appr = load_approved(args.approved_tsv)

    for holdout in pids:
        donors = [p for p in pids if p != holdout]
        # build calibration set from donors
        X_cal, y_cal = [], []
        for d in donors:
            fp = [f for f in files if f.endswith(f"_{d}.tsv")]
            if not fp: continue
            df = pd.read_csv(fp[0], sep="\t")
            if "drug" not in df.columns or "ml_score" not in df.columns: continue
            df["drug_lc"] = df["drug"].map(norm)
            s = pd.to_numeric(df["ml_score"], errors="coerce").to_numpy(dtype=float)
            y = df["drug_lc"].isin(appr).astype(int).to_numpy()
            if y.sum()==0 or y.sum()==len(y): continue
            X_cal.append(s); y_cal.append(y)
        if not X_cal:
            print(f"[WARN] {holdout}: no valid donor data, copying uncalibrated.")
            src = [f for f in files if f.endswith(f"_{holdout}.tsv")][0]
            pd.read_csv(src, sep="\t").to_csv(os.path.join(args.out_dir, os.path.basename(src)), sep="\t", index=False)
            continue

        X = np.concatenate(X_cal); y = np.concatenate(y_cal)

        if args.method == "isotonic":
            calib = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calib.fit(X, y)
            apply_fn = calib.transform
        else:  # Platt (logistic on score)
            m = LogisticRegression(solver="lbfgs", max_iter=200)
            m.fit(X.reshape(-1,1), y)
            apply_fn = (lambda v: m.predict_proba(np.asarray(v).reshape(-1,1))[:,1])

        # apply to holdout
        src = [f for f in files if f.endswith(f"_{holdout}.tsv")]
        if not src: continue
        dfh = pd.read_csv(src[0], sep="\t")
        s = pd.to_numeric(dfh["ml_score"], errors="coerce").to_numpy(dtype=float)
        dfh["ml_score_cal"] = apply_fn(s)
        outp = os.path.join(args.out_dir, os.path.basename(src[0]))
        dfh.to_csv(outp, sep="\t", index=False)
        print(f"[OK] wrote calibrated {os.path.basename(outp)}")
if __name__ == "__main__":
    main()
