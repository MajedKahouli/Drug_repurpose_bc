#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, re, glob, json, math, warnings, numpy as np
from collections import defaultdict

import cupy as cp
import cudf
import cugraph
import rmm

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------
# helpers
# ------------------------------
def _normalize_ppi_cols(df):
    cols = {c.lower(): c for c in df.columns}
    a_col = cols.get('genea') or cols.get('protein1') or list(df.columns)[0]
    b_col = cols.get('geneb') or cols.get('protein2') or list(df.columns)[1]
    return a_col, b_col

def load_ppi_gpu(ppi_path, min_degree=1):
    df = cudf.read_csv(ppi_path, sep='\t', dtype=str)
    a_col, b_col = _normalize_ppi_cols(df)
    edges = df[[a_col, b_col]].rename(columns={a_col: "src", b_col: "dst"}).astype("str")
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(edges, source="src", destination="dst")
    if min_degree > 1:
        deg = G.degree()
        keep = deg[deg["degree"] >= min_degree]["vertex"]
        G, _ = cugraph.subgraph(G, keep)
    return G

def degree_bins_cpu(G, nbins=10):
    deg = G.degree().sort_values("degree")
    if len(deg) == 0:
        return {}, set()
    dvals = deg["degree"].to_pandas().to_numpy()
    verts = deg["vertex"].astype(str).to_pandas().to_numpy()
    qs = np.linspace(0.0, 1.0, nbins + 1)
    bounds = np.quantile(dvals, qs)
    bounds[0], bounds[-1] = -np.inf, np.inf
    idx = np.searchsorted(bounds, dvals, side="right") - 1
    bins = {b: verts[idx == b] for b in range(nbins)}
    return bins, set(map(str, verts))

rng_default = np.random.default_rng()

def sample_matched_nodes_cpu(bins, k, rng=rng_default):
    if k <= 0 or not bins:
        return np.array([], dtype=object)
    all_nodes = np.concatenate([v for v in bins.values()]) if bins else np.array([], dtype=object)
    if all_nodes.size == 0:
        return np.array([], dtype=object)
    out, keys, i = [], sorted(bins.keys()), 0
    while len(out) < k:
        key = keys[i % len(keys)]
        pool = bins.get(key, np.array([], dtype=object))
        out.append((pool if pool.size else all_nodes)[rng.integers(0, (pool if pool.size else all_nodes).size)])
        i += 1
    return np.array(out, dtype=object)

def bfs_from_sources(G, sources):
    if not sources:
        return cudf.DataFrame({"vertex": [], "distance": []})
    df = cugraph.bfs(G, start=sources[0])[["vertex","distance"]].rename(columns={"distance":"d0"})
    for i, s in enumerate(sources[1:], start=1):
        di = cugraph.bfs(G, start=s)[["vertex","distance"]].rename(columns={"distance":f"d{i}"})
        df = df.merge(di, on="vertex", how="outer")
    mins = cp.nanmin(df[[c for c in df.columns if c.startswith("d")]].to_cupy(), axis=1)
    return cudf.DataFrame({"vertex": df["vertex"], "distance": cudf.Series(mins)})

def parse_module_arg(s):
    if s.startswith("topn:"):
        return ("topn", int(s.split(":", 1)[1]))
    if s.startswith("thresh:"):
        return ("thresh", s.split(":", 1)[1])
    raise ValueError("module must be topn:<N> or thresh:<expr>")

def build_patient_module(deg_df, module_spec):
    cols = {c.lower(): c for c in deg_df.columns}
    gene_col = cols.get("gene") or list(deg_df.columns)[0]
    lf_col   = cols.get("logfc")
    pv_col   = cols.get("pval") or cols.get("pvalue")
    padj_col = cols.get("padj") or cols.get("fdr") or cols.get("adj_p")
    deg_df = deg_df.rename(columns={gene_col: "gene"})
    kind, val = module_spec
    if kind == "topn":
        n = int(val)
        if lf_col:
            up = deg_df.sort_values(lf_col, ascending=False).head(n)
            down = deg_df.sort_values(lf_col, ascending=True).head(n)
            mod = cudf.concat([up, down])["gene"].astype(str)
        else:
            mod = deg_df.head(2*n)["gene"].astype(str)
        return set(mod.to_pandas())
    mask = cudf.Series(cp.ones(len(deg_df), dtype=cp.bool_))
    if kind == "thresh":
        if lf_col and ("logfc" in val.lower()):
            m = re.search(r"\|?logfc\|?\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I)
            if m:
                op, thr = m.group(1), float(m.group(2))
                comp = deg_df[lf_col].abs() if '|' in val else deg_df[lf_col]
                mask &= {"<": comp < thr, "<=": comp <= thr, ">": comp > thr, ">=": comp >= thr}[op]
        if padj_col:
            m = re.search(r"padj\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I)
            if m:
                op, thr = m.groups(); thr = float(thr)
                mask &= {"<": deg_df[padj_col] < thr, "<=": deg_df[padj_col] <= thr,
                         ">": deg_df[padj_col] > thr, ">=": deg_df[padj_col] >= thr}[op]
        if pv_col:
            m = re.search(r"pval\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I)
            if m:
                op, thr = m.groups(); thr = float(thr)
                mask &= {"<": deg_df[pv_col] < thr, "<=": deg_df[pv_col] <= thr,
                         ">": deg_df[pv_col] > thr, ">=": deg_df[pv_col] >= thr}[op]
        return set(deg_df.loc[mask, "gene"].astype(str).to_pandas())
    raise ValueError("Unknown module spec")

def compute_proximity_z_gpu(G, A, B, bins, vertex_set, n_null=200, rng=rng_default):
    A = [a for a in set(A) if a in vertex_set]
    B = [b for b in set(B) if b in vertex_set]
    if not A or not B:
        return float("nan"), float("nan")
    dist_df = bfs_from_sources(G, B)
    if len(dist_df) == 0:
        return float("nan"), float("nan")
    dist_map = dist_df.set_index("vertex")["distance"].to_pandas().to_dict()
    obs = np.asarray([float(dist_map.get(a, np.inf)) for a in A], np.float32)
    obs = obs[np.isfinite(obs)]
    if obs.size == 0:
        return float("nan"), float("nan")
    d_obs = float(obs.mean())

    kA, kB = len(A), len(B)
    null_vals = []
    for _ in range(n_null):
        A_rand = sample_matched_nodes_cpu(bins, kA, rng)
        B_rand = sample_matched_nodes_cpu(bins, kB, rng)
        dist_rand = bfs_from_sources(G, list(map(str, B_rand)))
        rand_map = dist_rand.set_index("vertex")["distance"].to_pandas().to_dict()
        vals = np.asarray([float(rand_map.get(str(a), np.inf)) for a in A_rand], np.float32)
        vals = vals[np.isfinite(vals)]
        if vals.size: null_vals.append(float(vals.mean()))
    if len(null_vals) < 5: return float("nan"), d_obs
    mu, sd = float(np.mean(null_vals)), float(np.std(null_vals))
    if not math.isfinite(sd) or sd == 0.0: return float("nan"), d_obs
    return float((d_obs - mu) / sd), d_obs

# ------------------------------
def _rmm_banner():
    try:
        cur = rmm.mr.get_current_device_resource_type()
    except Exception as e:
        cur = f"unknown ({e})"
    free, total = cp.cuda.runtime.memGetInfo()
    used_mb = (total - free) // (1024*1024)
    print(f"RMM pool: {cur}")
    print(f"GPU memory currently used (MB): {used_mb}")

def _find_deg_file(degs_dir, pid):
    # try several case patterns
    candidates = []
    for pat in (f"degs_{pid}.tsv", f"DEGs_{pid}.tsv", f"*{pid}*.tsv"):
        candidates.extend(glob.glob(os.path.join(degs_dir, pat)))
    return candidates[0] if candidates else None

# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reversal-dir', required=True)
    ap.add_argument('--degs-dir',     required=True)
    ap.add_argument('--targets',      required=False, default='data/drug_targets.tsv')
    ap.add_argument('--ppi',          required=False, default='data/ppi_edges.tsv')
    ap.add_argument('--out',          required=False, default='work/features_gpu')
    ap.add_argument('--module',       default='topn:150')
    ap.add_argument('--n-null',       type=int, default=200)
    ap.add_argument('--ppi-min-degree', type=int, default=1)
    ap.add_argument('--cache-dir',    default=None)
    ap.add_argument('--tsv',          action='store_true')
    ap.add_argument('--only-patients', help='CSV or space-separated list of patient IDs to run', default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.cache_dir: os.makedirs(args.cache_dir, exist_ok=True)

    _rmm_banner()
    print("Loading PPI on GPU…")
    G = load_ppi_gpu(args.ppi, min_degree=args.ppi_min_degree)
    print(f"Graph: {G.number_of_vertices()} nodes, {G.number_of_edges()} edges")

    print("Computing degree bins…")
    bins, vertex_set = degree_bins_cpu(G)

    print("Loading drug targets…")
    tdf = cudf.read_csv(args.targets, sep='\t', dtype=str)[["drug","target"]]
    drug2targets = defaultdict(set)
    for d, g in tdf.to_pandas().itertuples(index=False):
        drug2targets[d].add(g)

    def cache_path_for(drug):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", drug)
        return os.path.join(args.cache_dir, f"bfs_{safe}.parquet") if args.cache_dir else None

    def get_bfs_for_drug(drug):
        pth = cache_path_for(drug)
        if pth and os.path.exists(pth):
            try: return cudf.read_parquet(pth)
            except Exception: pass
        targets = [t for t in drug2targets.get(drug, []) if t in vertex_set]
        df = bfs_from_sources(G, targets) if targets else cudf.DataFrame({"vertex": [], "distance": []})
        if pth and len(df):
            try: df.to_parquet(pth)
            except Exception: pass
        return df

    # determine which patients to run
    rev_files_all = sorted(glob.glob(os.path.join(args.reversal_dir, 'reversal_scores_*.tsv')))
    if args.only-patients:
        want = set(re.split(r"[,\s]+", args.only_patients.strip()))
        rev_files = [rf for rf in rev_files_all
                     if re.search(r'reversal_scores_(.+)\.tsv$', os.path.basename(rf)).group(1) in want]
    else:
        rev_files = rev_files_all

    module_spec = parse_module_arg(args.module)

    for rf in rev_files:
        pid = re.search(r'reversal_scores_(.+)\.tsv$', os.path.basename(rf)).group(1)
        print(f"\nPatient {pid} …")
        rev = cudf.read_csv(rf, sep='\t', dtype={"drug":"str"})
        if ("drug" not in rev.columns) or ("reversal_score" not in rev.columns):
            raise ValueError(f"{rf} must have columns [drug, reversal_score]")

        deg_path = _find_deg_file(args.degs_dir, pid)
        if not deg_path:
            print(f"WARNING: no DEGs found for {pid}; skipping")
            continue

        degs = cudf.read_csv(deg_path, sep='\t')
        module_genes = build_patient_module(degs, module_spec)

        cols = {c.lower(): c for c in degs.columns}
        lf_col = cols.get('logfc')
        weights = {}
        if lf_col:
            tmp = degs[["gene", lf_col]].rename(columns={lf_col: "logFC"})
            tmp["w"] = tmp["logFC"].abs()
            weights = dict(zip(tmp["gene"].astype(str).to_pandas(), tmp["w"].to_pandas()))

        recs = []
        for drug, revscore in rev[["drug", "reversal_score"]].to_pandas().itertuples(index=False):
            B = set([t for t in drug2targets.get(drug, []) if t in vertex_set])
            A = module_genes
            inter = B.intersection(A); union = B.union(A)
            overlap_count = len(inter)
            jaccard = (len(inter) / len(union)) if len(union) else float("nan")
            weighted_overlap = float(sum(weights.get(g, 0.0) for g in inter)) if inter else 0.0

            dist_from_B = get_bfs_for_drug(drug)
            if len(dist_from_B):
                dist_map = dist_from_B.set_index("vertex")["distance"].to_pandas().to_dict()
                vals = np.asarray([float(dist_map.get(a, np.inf)) for a in A], np.float32)
                finite = vals[np.isfinite(vals)]
                prox_raw = float(finite.mean()) if finite.size else float("nan")
            else:
                prox_raw = float("nan")

            prox_z, _ = compute_proximity_z_gpu(G, list(A), list(B), bins, vertex_set, n_null=args.n_null)

            recs.append({
                "drug": drug, "reversal_score": float(revscore),
                "overlap_count": overlap_count, "jaccard": jaccard,
                "weighted_overlap": weighted_overlap, "prox_z": prox_z, "prox_raw": prox_raw,
                "n_targets_hit": len(B), "n_deg_module": len(A),
            })

        out_df = cudf.DataFrame(recs).sort_values("reversal_score", ascending=False)
        if args.tsv:
            out_path = os.path.join(args.out, f"features_{pid}.tsv")
            out_df.to_pandas().to_csv(out_path, sep="\t", index=False)
        else:
            out_path = os.path.join(args.out, f"features_{pid}.parquet")
            out_df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(out_df)} rows)")

if __name__ == "__main__":
    main()
