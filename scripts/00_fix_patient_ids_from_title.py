import argparse, re
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

df = pd.read_csv(args.inp)

def pid_from_title(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    m = re.search(r'\b(BC\d{3,5})[NT]\b', s)    # e.g., BC0043N / BC0043T
    if m: return m.group(1)
    m = re.search(r'\b(BC\d{3,5})\b', s)        # fallback: just BC#####
    return m.group(1) if m else ""

if "patient_id" not in df.columns:
    df["patient_id"] = ""

mask = df["patient_id"].isna() | (df["patient_id"].astype(str).str.strip().isin(["", "na", "NA"]))
df.loc[mask, "patient_id"] = df.loc[mask, "title"].apply(pid_from_title)

# quick validation: exactly one tumor + one normal per patient
pairs = df.groupby("patient_id")["condition"].value_counts().unstack(fill_value=0)
ok = (pairs.get("normal", 0) == 1) & (pairs.get("tumor", 0) == 1)
good = ok.sum()

print(f"[INFO] Patients with clean pairs: {good}")
if good == 0:
    print("[WARN] No valid pairs found. Please inspect the CSV.")
else:
    print(pairs.head())

df.to_csv(args.out, index=False)
print(f"[OK] Wrote: {args.out}")
