import os, sys, pandas as pd, glob

# Preferred sources, in order
CANDIDATES = [
    "data/targets/drug.target.interaction.tsv",
    "data/targets/interactions.tsv",
    "data/targets/drug_targets.tsv",   # already normalized? then just copy/clean
]

def pick_source():
    for p in CANDIDATES:
        if os.path.exists(p):
            return p
    # fallback: try any .tsv in data/targets
    hits = sorted(glob.glob("data/targets/*.tsv"))
    if hits:
        return hits[0]
    return None

src = pick_source()
if not src:
    sys.exit("[ERROR] No source targets file found under data/targets/. Put a TSV there and re-run.")

print(f"[INFO] Using source: {src}")
df = pd.read_csv(src, sep="\t", dtype=str, low_memory=False)
lc = {c.lower(): c for c in df.columns}

# Try to infer the drug and target columns
drug_col = (lc.get("drug") or lc.get("drug_name") or lc.get("pert_iname")
            or lc.get("compound") or lc.get("name"))
tgt_col  = (lc.get("target") or lc.get("gene") or lc.get("gene_symbol")
            or lc.get("symbol") or lc.get("target_gene"))

if not drug_col or not tgt_col:
    print("[ERROR] Could not infer (drug, target) columns from the file.")
    print(f"Columns present: {list(df.columns)}")
    sys.exit(1)

out = df[[drug_col, tgt_col]].rename(columns={drug_col:"drug", tgt_col:"target"}).dropna()
out["drug"]   = out["drug"].astype(str).str.strip()
out["target"] = out["target"].astype(str).str.strip()
out = out[(out["drug"]!="") & (out["target"]!="")].drop_duplicates()

os.makedirs("data", exist_ok=True)
outpath = "data/drug_targets.tsv"
out.to_csv(outpath, sep="\t", index=False)
print(f"[OK] Wrote {outpath}  rows={len(out):,}")
