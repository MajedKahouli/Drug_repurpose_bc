import argparse
from pathlib import Path
import pandas as pd, numpy as np
import networkx as nx
from tqdm import tqdm

def load_string_graph(links_path: Path, min_score=700):
    # links columns: protein1 protein2 combined_score
    df = pd.read_csv(links_path, sep=" ", compression="gzip" if str(links_path).endswith(".gz") else None)
    df = df[df["combined_score"] >= min_score]
    G = nx.Graph()
    G.add_edges_from(zip(df["protein1"], df["protein2"]))
    return G

def load_alias_map(alias_path: Path):
    # alias columns: protein_id alias source
    df = pd.read_csv(alias_path, sep="\t", compression="gzip" if str(alias_path).endswith(".gz") else None,
                     header=None, names=["protein","alias","source"])
    df = df[df["source"].str.contains("Gene name", na=False)]
    return df.groupby("alias")["protein"].first().to_dict()  # symbol -> one STRING node

def read_targets(target_tsv: Path):
    # expect columns: drug (or drug_name), target_gene (HGNC symbol)
    t = pd.read_csv(target_tsv, sep="\t")
    # normalize column names
    cols = {c.lower(): c for c in t.columns}
    drug_col = cols.get("drug", cols.get("drug_name"))
    gene_col = cols.get("target_gene", cols.get("gene", cols.get("target")))
    if drug_col is None or gene_col is None:
        raise ValueError("Need columns: drug/drug_name and target_gene/gene/target")
    return t[[drug_col, gene_col]].rename(columns={drug_col:"drug", gene_col:"gene"})

def dysregulated_genes(deg_path: Path, p=0.05, lfc=0.5):
    d = pd.read_csv(deg_path, sep="\t")
    cols = {c.lower(): c for c in d.columns}
    pcol = cols.get("adj.p.val", cols.get("adj_p_val", cols.get("fdr", "adj.P.Val")))
    lcol = cols.get("logfc", "logFC")
    gcol = cols.get("gene", "gene")
    if pcol not in d.columns: pcol = "P.Value" if "P.Value" in d.columns else list(d.columns)[1]
    x = d[(d[pcol] < p) & (d[lcol].abs() >= lfc)][gcol].dropna().astype(str).tolist()
    return x

def avg_shortest_path(G, S_targets, S_patient):
    # compute average shortest path length between any target and any patient gene
    # using precomputed single-source BFS for speed on small sets
    if not S_targets or not S_patient: return np.nan
    dists = []
    for t in S_targets:
        if t not in G: 
            continue
        lengths = nx.single_source_shortest_path_length(G, t, cutoff=4)  # local neighborhood
        for g in S_patient:
            if g in lengths:
                dists.append(lengths[g])
    if not dists:
        return np.inf
    return float(np.mean(dists))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deg-dir", required=True)
    ap.add_argument("--targets", required=True)  # TSV with drug,target_gene
    ap.add_argument("--string-links", required=True)  # 9606.protein.links.v12.0.txt.gz
    ap.add_argument("--string-alias", required=True)  # 9606.protein.aliases.v12.0.txt.gz
    ap.add_argument("--p", type=float, default=0.05)
    ap.add_argument("--lfc", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading STRING graph …")
    G = load_string_graph(Path(args.string_links))
    print("[INFO] Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

    sym2prot = load_alias_map(Path(args.string_alias))
    T = read_targets(Path(args.targets))

    # map gene symbols to STRING nodes
    T["node"] = T["gene"].map(sym2prot)
    T = T.dropna(subset=["node"])
    # collect per-drug node sets
    drug2nodes = T.groupby("drug")["node"].apply(lambda s: set(s.tolist())).to_dict()

    deg_files = sorted(Path(args.deg_dir).glob("DEGs_*.tsv"))
    for f in deg_files:
        pid = f.stem.replace("DEGs_","")
        patient_genes = dysregulated_genes(f, p=args.p, lfc=args.lfc)
        P_nodes = set([sym2prot.get(g) for g in patient_genes if g in sym2prot])

        rows = []
        for drug, nodes in tqdm(drug2nodes.items(), desc=f"{pid}", leave=False):
            d = avg_shortest_path(G, nodes, P_nodes)
            rows.append((drug, d))
        df = pd.DataFrame(rows, columns=["drug","avg_shortest_path"])
        # convert to proximity score (smaller distance = better → higher score)
        # use negative distance; handle inf as very poor (min-1)
        finite = df["avg_shortest_path"].replace([np.inf, -np.inf], np.nan)
        worst = np.nanmax(finite.values)
        df["proximity_score"] = -df["avg_shortest_path"].replace(np.inf, worst+1.0)
        df.to_csv(outdir / f"proximity_{pid}.tsv", sep="\t", index=False)
        print(f"[OK] {pid}: wrote {df.shape[0]} rows")

if __name__ == "__main__":
    main()
