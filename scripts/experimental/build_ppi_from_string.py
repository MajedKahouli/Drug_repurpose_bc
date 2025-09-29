import pandas as pd, os, sys

links   = "data/networks/string/9606.protein.physical.links.v12.0.txt"
aliases = "data/networks/string/9606.protein.aliases.v12.0.txt"

if not os.path.exists(links):
    sys.exit(f"[ERROR] Missing {links}")
if not os.path.exists(aliases):
    sys.exit(f"[ERROR] Missing {aliases}")

print("[INFO] reading aliases…")
ali = pd.read_csv(
    aliases, sep="\t", dtype=str,
    names=["protein_id","alias","source"], comment="#"
).dropna()

# Keep symbol-like aliases only
ali = ali[ali["alias"].str.match(r"^[A-Za-z0-9\-]+$")]

# Prefer HGNC / Gene_Name etc
pref = ["Ensembl_HGNC","HGNC","Gene_Name","GeneCards","UniProtKB","Ensembl_UniProt"]
ali["rank"] = ali["source"].apply(lambda s: pref.index(s) if s in pref else 99)
ali = ali.sort_values(["protein_id","rank"])
id2sym = ali.drop_duplicates("protein_id").set_index("protein_id")["alias"]

print("[INFO] reading physical links…")
lnk = pd.read_csv(
    links, sep=r"\s+", engine="python",
    dtype={"protein1":str,"protein2":str,"combined_score":int}
)
lnk = lnk[lnk["combined_score"] >= 700].copy()

print("[INFO] mapping protein ids -> gene symbols…")
lnk["geneA"] = lnk["protein1"].map(id2sym)
lnk["geneB"] = lnk["protein2"].map(id2sym)

df = lnk.dropna(subset=["geneA","geneB"])[["geneA","geneB"]].astype(str)
df = df[df["geneA"] != df["geneB"]].drop_duplicates()

os.makedirs("data", exist_ok=True)
out = "data/ppi_edges.tsv"
df.to_csv(out, sep="\t", index=False)
print(f"[OK] wrote {out} rows={len(df):,}")
