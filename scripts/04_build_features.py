
#!/usr/bin/env python3
import argparse
import os
import re
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Memory
from scipy.stats import zscore

# ------------------------------
# Helpers
# ------------------------------

def load_ppi(ppi_path, min_degree=1):
    df = pd.read_csv(ppi_path, sep='\t', header=0)
    # Attempt to normalize column names
    cols = {c.lower(): c for c in df.columns}
    a_col = cols.get('genea') or cols.get('protein1') or list(df.columns)[0]
    b_col = cols.get('geneb') or cols.get('protein2') or list(df.columns)[1]
    edges = df[[a_col, b_col]].astype(str)
    G = nx.Graph()
    G.add_edges_from(edges.itertuples(index=False, name=None))
    # Optionally prune isolated/low-degree nodes
    if min_degree > 1:
        drop = [n for n, d in G.degree() if d < min_degree]
        G.remove_nodes_from(drop)
    return G

def degree_bins(G, nbins=10):
    degrees = dict(G.degree())
    if not degrees:
        return {}
    vals = np.array(list(degrees.values()))
    # Bin by quantiles to roughly preserve degree distribution
    qs = np.quantile(vals, np.linspace(0,1,nbins+1))
    qs[0] = -np.inf; qs[-1] = np.inf
    bins = defaultdict(list)
    for n, d in degrees.items():
        # find bin index
        idx = int(np.searchsorted(qs, d, side='right') - 1)
        bins[idx].append(n)
    return bins

def sample_matched_nodes(bins, k):
    # Sample k nodes with degree matched (approx) using proportional bins
    # If k larger than available, sample with replacement across bins
    if k <= 0:
        return []
    # Flatten bins for fallback
    all_nodes = [n for nodes in bins.values() for n in nodes]
    if not all_nodes:
        return []
    out = []
    # Round-robin across bins to keep composition similar
    keys = sorted(bins.keys())
    i = 0
    while len(out) < k:
        key = keys[i % len(keys)]
        pool = bins[key]
        if pool:
            out.append(np.random.choice(pool))
        else:
            out.append(np.random.choice(all_nodes))
        i += 1
    return out

def bfs_multi_source(G, sources):
    # Unweighted BFS distances from multiple sources
    # Return dict: node -> min distance to any source
    if len(sources) == 0:
        return {}
    # networkx.single_source_shortest_path_length can take only one source; implement a queue-based BFS
    from collections import deque
    dist = {}
    dq = deque()
    for s in sources:
        if s in G:
            dist[s] = 0
            dq.append(s)
    while dq:
        u = dq.popleft()
        du = dist[u]
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = du + 1
                dq.append(v)
    return dist

# ------------------------------
# Proximity computation with caching
# ------------------------------

def compute_proximity_z(G, A, B, bins, n_null=200, rng=None):
    """
    A: list/set of patient module nodes
    B: list/set of drug target nodes
    Returns: prox_z (negative = closer-than-random), prox_raw (observed avg min-dist)
    """
    if rng is None:
        rng = np.random.default_rng()
    A = [a for a in set(A) if a in G]
    B = [b for b in set(B) if b in G]
    if len(A) == 0 or len(B) == 0:
        return np.nan, np.nan

    # Precompute distances from B once
    dist_from_B = bfs_multi_source(G, B)
    if len(dist_from_B) == 0:
        return np.nan, np.nan

    # Observed average min distance
    d_obs_vals = [dist_from_B.get(a, np.inf) for a in A]
    finite = [d for d in d_obs_vals if np.isfinite(d)]
    if len(finite) == 0:
        return np.nan, np.nan
    d_obs = float(np.mean(finite))

    # Null distribution via degree-matched sampling
    kA, kB = len(A), len(B)
    null_vals = []
    for _ in range(n_null):
        A_rand = sample_matched_nodes(bins, kA)
        B_rand = sample_matched_nodes(bins, kB)
        dist_rand = bfs_multi_source(G, B_rand)
        vals = [dist_rand.get(a, np.inf) for a in A_rand]
        vals = [d for d in vals if np.isfinite(d)]
        if len(vals) == 0:
            continue
        null_vals.append(np.mean(vals))
    if len(null_vals) < 5:
        return np.nan, d_obs
    mu = float(np.mean(null_vals))
    sd = float(np.std(null_vals)) or np.nan
    if not np.isfinite(sd) or sd == 0:
        return np.nan, d_obs
    z = (d_obs - mu) / sd
    return z, d_obs

# ------------------------------
# Main builder
# ------------------------------

def parse_module_arg(s):
    # supports: "topn:150" or "thresh:|logFC|>=1.0,padj<=0.05" or "thresh:|logFC|>=1.0,pval<=0.01"
    if s.startswith('topn:'):
        return ('topn', int(s.split(':',1)[1]))
    if s.startswith('thresh:'):
        body = s.split(':',1)[1]
        return ('thresh', body)
    raise ValueError('module must be topn:<N> or thresh:<expr>')

def build_patient_module(deg_df, module_spec):
    kind, val = module_spec
    deg_df = deg_df.copy()
    # Column normalization
    cols = {c.lower(): c for c in deg_df.columns}
    gene_col = cols.get('gene') or list(deg_df.columns)[0]
    deg_df.rename(columns={gene_col:'gene'}, inplace=True)
    lf_col = cols.get('logfc')
    pv_col = cols.get('pval') or cols.get('pvalue')
    padj_col = cols.get('padj') or cols.get('fdr') or cols.get('adj_p')

    if kind == 'topn':
        n = int(val)
        # Split up/down if logFC exists; otherwise just take top by |stat| if available
        if lf_col:
            up = deg_df.sort_values(lf_col, ascending=False).head(n)
            down = deg_df.sort_values(lf_col, ascending=True).head(n)
            mod = pd.concat([up, down])
        else:
            mod = deg_df.head(2*n)
        return set(mod['gene'].astype(str))

    if kind == 'thresh':
        # very simple parser: look for tokens like |logFC|>=1.0 and padj<=0.05 or pval<=0.01
        mask = pd.Series(True, index=deg_df.index)
        if lf_col and ('logfc' in val.lower()):
            m = re.search(r"\|?logfc\|?\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I)
            if m:
                op, thr = m.group(1), float(m.group(2))
                if '|' in val:
                    comp = deg_df[lf_col].abs()
                else:
                    comp = deg_df[lf_col]
                if op == '>=':
                    mask &= comp >= thr
                elif op == '>':
                    mask &= comp > thr
                elif op == '<=':
                    mask &= comp <= thr
                elif op == '<':
                    mask &= comp < thr
        if ('padj' in (padj_col or '').lower()) or ('padj' in val.lower()):
            if padj_col and re.search(r"padj\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I):
                op, thr = re.search(r"padj\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I).groups()
                thr = float(thr)
                if op == '<=':
                    mask &= deg_df[padj_col] <= thr
                elif op == '<':
                    mask &= deg_df[padj_col] < thr
                elif op == '>=':
                    mask &= deg_df[padj_col] >= thr
                elif op == '>':
                    mask &= deg_df[padj_col] > thr
        if ('pval' in (pv_col or '').lower()) or ('pval' in val.lower()):
            if pv_col and re.search(r"pval\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I):
                op, thr = re.search(r"pval\s*([><]=?)\s*([0-9\.]+)", val, flags=re.I).groups()
                thr = float(thr)
                if op == '<=':
                    mask &= deg_df[pv_col] <= thr
                elif op == '<':
                    mask &= deg_df[pv_col] < thr
                elif op == '>=':
                    mask &= deg_df[pv_col] >= thr
                elif op == '>':
                    mask &= deg_df[pv_col] > thr
        mod = deg_df.loc[mask, 'gene'].astype(str)
        return set(mod)

    raise ValueError('Unknown module spec')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reversal-dir', required=True)
    ap.add_argument('--degs-dir', required=True)
    ap.add_argument('--targets', required=True)
    ap.add_argument('--ppi', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--module', default='topn:150', help='topn:N or thresh:|logFC|>=1.0,padj<=0.05')
    ap.add_argument('--n-null', type=int, default=200)
    ap.add_argument('--ppi-min-degree', type=int, default=1)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--tsv', action='store_true', help='write TSV instead of Parquet')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Cache for per-drug BFS distances across patients
    mem = Memory(args.cache_dir, verbose=0) if args.cache_dir else None

    # Load PPI
    print('Loading PPI...')
    G = load_ppi(args.ppi, min_degree=args.ppi_min_degree)
    bins = degree_bins(G)

    # Load drug targets
    print('Loading drug targets...')
    tdf = pd.read_csv(args.targets, sep='\t')
    tdf = tdf[['drug', 'target']].astype(str)
    drug2targets = defaultdict(set)
    for d, g in tdf.itertuples(index=False):
        drug2targets[d].add(g)

    # List patients by reversal files
    rev_files = sorted(glob.glob(os.path.join(args.reversal_dir, 'reversal_scores_*.tsv')))
    module_spec = parse_module_arg(args.module)

    # Optional cached BFS: wrap function
    @mem.cache if mem else (lambda f: f)
    def cached_bfs_sources(drug):
        return bfs_multi_source(G, [t for t in drug2targets.get(drug, []) if t in G])

    for rf in tqdm(rev_files, desc='Patients'):
        pid = re.search(r'reversal_scores_(.+)\.tsv$', os.path.basename(rf)).group(1)
        rev = pd.read_csv(rf, sep='\t')
        if 'drug' not in rev.columns or 'reversal_score' not in rev.columns:
            raise ValueError(f'{rf} must have columns [drug, reversal_score]')
        rev['drug'] = rev['drug'].astype(str)

        # Load DEGs for PID
        deg_path = os.path.join(args.degs_dir, f'degs_{pid}.tsv')
        if not os.path.exists(deg_path):
            # Try alternative naming
            candidates = glob.glob(os.path.join(args.degs_dir, f'*{pid}*.tsv'))
            if not candidates:
                print(f'WARNING: no DEGs found for {pid}; skipping')
                continue
            deg_path = candidates[0]
        degs = pd.read_csv(deg_path, sep='\t')
        module_genes = build_patient_module(degs, module_spec)

        # Precompute weights for weighted overlap
        cols = {c.lower(): c for c in degs.columns}
        lf_col = cols.get('logfc')
        weights = {}
        if lf_col:
            tmp = degs[['gene', lf_col]].copy()
            tmp.columns = ['gene', 'logFC']
            tmp['w'] = tmp['logFC'].abs()
            weights = dict(zip(tmp['gene'].astype(str), tmp['w']))

        recs = []
        for drug, revscore in rev[['drug','reversal_score']].itertuples(index=False):
            targets = [t for t in drug2targets.get(drug, []) if t in G]
            A = module_genes
            B = set(targets)

            # Overlap features
            inter = B.intersection(A)
            union = B.union(A)
            overlap_count = len(inter)
            jaccard = (len(inter) / len(union)) if len(union) > 0 else np.nan
            weighted_overlap = float(np.sum([weights.get(g, 0.0) for g in inter])) if inter else 0.0

            # Proximity (z negative = closer = better)
            # Use cached BFS for this drug's targets
            if mem:
                dist_from_B = cached_bfs_sources(drug)
            else:
                dist_from_B = bfs_multi_source(G, list(B))

            # Observed
            d_obs_vals = [dist_from_B.get(a, np.inf) for a in A]
            finite = [d for d in d_obs_vals if np.isfinite(d)]
            prox_raw = float(np.mean(finite)) if finite else np.nan

            # Null z
            prox_z, _ = compute_proximity_z(G, list(A), list(B), bins, n_null=args.n_null)

            recs.append({
                'drug': drug,
                'reversal_score': revscore,
                'overlap_count': overlap_count,
                'jaccard': jaccard,
                'weighted_overlap': weighted_overlap,
                'prox_z': prox_z,
                'prox_raw': prox_raw,
                'n_targets_hit': len(B),
                'n_deg_module': len(A),
            })

        out_df = pd.DataFrame.from_records(recs)
        out_df = out_df.sort_values('reversal_score', ascending=False)
        out_path = os.path.join(args.out, f'features_{pid}.parquet')
        if args.tsv:
            out_path = os.path.join(args.out, f'features_{pid}.tsv')
            out_df.to_csv(out_path, sep='\t', index=False)
        else:
            out_df.to_parquet(out_path, index=False)
        print(f'Wrote {out_path} ({len(out_df)} rows)')

if __name__ == '__main__':
    main()
