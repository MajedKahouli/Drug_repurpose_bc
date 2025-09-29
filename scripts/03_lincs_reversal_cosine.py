# scripts/03_lincs_reversal_cosine.py
# (v5 — uses Level-5 GCTX: columns = sig_id; robust matching + smoke test + loud logging)

import argparse, glob, traceback, re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from cmapPy.pandasGEXpress.parse_gctx import parse as parse_gctx

# ------------ tiny utils ------------
_SIG_TAIL = re.compile(r":[-+]?\d+(?:\.\d+)?$")  # matches a trailing :<int/float>

def clean_sig_id(s: str) -> str:
    """Drop a trailing ':<number>' if present (seen in Level-5 columns)."""
    return _SIG_TAIL.sub("", str(s))

def pick_one(path_or_glob: str) -> str:
    if any(ch in path_or_glob for ch in ["*", "?", "["]):
        hits = sorted(glob.glob(path_or_glob))
        if not hits:
            raise FileNotFoundError(f"No files matched: {path_or_glob}")
        return hits[0]
    return path_or_glob

# ------------ IO helpers ------------
def load_lm978_geneinfo(gene_info_path: Path) -> pd.DataFrame:
    gi = pd.read_csv(gene_info_path, sep="\t", low_memory=False)
    gl = {c.lower(): c for c in gi.columns}
    sym_col = gl.get("pr_gene_symbol") or gl.get("gene_symbol")
    if not sym_col:
        raise ValueError("No gene symbol column (pr_gene_symbol/gene_symbol)")
    candidates_lc = [c for c in ["rid","pr_id","pr_gene_id","gene_id","feature_id","id"] if c in gl]
    cols = [gl[sym_col]] + ([gl[c] for c in candidates_lc] if candidates_lc else [])
    out = gi[cols].copy().rename(columns={gl[sym_col]:"gene"})
    for c in out.columns:
        out[c] = out[c].astype(str)
    return out  # columns: gene + possible keys that match GCTX row IDs

def load_sig_and_pert(sig_info_path: Path, pert_info_path: Path, cell_lines: str) -> pd.DataFrame:
    """Return catalog indexed by cleaned sig_id with ['pert_iname','cell_id']."""
    si = pd.read_csv(sig_info_path, sep="\t", low_memory=False)
    pi = pd.read_csv(pert_info_path, sep="\t", low_memory=False)

    # keep compounds if present
    if "pert_type" in si.columns:
        si = si[si["pert_type"].astype(str).str.lower().eq("trt_cp")]

    # filter cell line(s)
    if "cell_id" not in si.columns:
        raise ValueError("sig_info missing 'cell_id'")
    keep = {c.strip().upper() for c in str(cell_lines).split(",")}
    si = si[si["cell_id"].astype(str).str.upper().isin(keep)]

    # ensure pert_iname present (merge from pert_info if needed)
    if "pert_iname" not in si.columns:
        if "pert_id" in si.columns and "pert_id" in pi.columns and "pert_iname" in pi.columns:
            si = si.merge(pi[["pert_id","pert_iname"]], on="pert_id", how="left")
        elif "pert_id" in si.columns:
            si["pert_iname"] = si["pert_id"].astype(str)
        else:
            si["pert_iname"] = si.get("sig_id", pd.Series(index=si.index, dtype=str)).astype(str)

    # clean / types
    for col in ["sig_id","pert_iname","cell_id"]:
        if col not in si.columns:
            raise ValueError(f"Missing column after normalization: {col}")
    si["sig_id"]     = si["sig_id"].astype(str).map(clean_sig_id)
    si["pert_iname"] = si["pert_iname"].astype(str)
    si["cell_id"]    = si["cell_id"].astype(str)

    si = si[["sig_id","pert_iname","cell_id"]].dropna().drop_duplicates(subset=["sig_id"])
    si = si.set_index("sig_id")
    return si  # index: cleaned sig_id

# ------------ GCTX (LEVEL-5) loader ------------
def load_level5_gctx_and_align_by_sig(
    gctx_path: Path,
    sig_info_path: Path,
    pert_info_path: Path,
    cell_lines: str
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Load Level-5 GCTX (genes x signatures). GCTX column labels are original
    sig_ids (often with a trailing ':<number>'). We:
      1) build a catalog of *cleaned* sig_ids from sig_info,
      2) keep those GCTX columns whose *cleaned* name is in the catalog,
      3) select by ORIGINAL labels, then rename the kept columns to the CLEANED ids,
      4) return col_meta indexed by the same CLEANED ids.
    """
    # 1) filtered catalog by sig_id (index = cleaned sig_id)
    cat = load_sig_and_pert(sig_info_path, pert_info_path, cell_lines)

    # 2) read Level-5 GCTX
    print("[INFO] Reading Level-5 GCTX (genes x signatures) and aligning by sig_id …")
    g = parse_gctx(str(gctx_path), rid=None, cid=None)
    G_all = g.data_df

    # 3) map original -> cleaned sig_id
    orig_cols  = G_all.columns.astype(str).tolist()
    clean_cols = [clean_sig_id(c) for c in orig_cols]

    # Decide which columns to keep by checking CLEANED names against the catalog
    cat_ids = set(cat.index)  # cleaned
    keep_pairs = [(orig, clean) for orig, clean in zip(orig_cols, clean_cols) if clean in cat_ids]

    overlap_n = len(keep_pairs)
    if overlap_n == 0:
        dbg_dir = Path(gctx_path).parent
        (dbg_dir / "diag_level5_cols_200.txt").write_text(
            "\n".join(clean_cols[:200]), encoding="utf-8"
        )
        (dbg_dir / "diag_cat_sigids_200.txt").write_text(
            "\n".join(list(map(str, cat.index[:200]))), encoding="utf-8"
        )
        raise RuntimeError(
            "No overlap between Level-5 GCTX columns (sig_id) and filtered sig_info.sig_id.\n"
            f"Wrote heads to:\n  {dbg_dir / 'diag_level5_cols_200.txt'}\n  {dbg_dir / 'diag_cat_sigids_200.txt'}"
        )

    keep_orig  = [o for o, _ in keep_pairs]   # as in G_all
    keep_clean = [c for _, c in keep_pairs]   # cleaned, matches cat.index

    # 4) subset by ORIGINAL labels, then rename to CLEANED ids
    G = G_all.loc[:, keep_orig].copy()
    G.columns = pd.Index(keep_clean, name="sig_id")

    # Align col_meta to the same cleaned ids/order
    col_meta = cat.loc[keep_clean, ["pert_iname", "cell_id"]].copy()
    col_meta.index = G.columns  # identical labels/order

    return G, col_meta, overlap_n



# ------------ mapping rows → symbols ------------
def index_looks_like_symbols(idx: pd.Index) -> bool:
    s = pd.Series(idx.astype(str))
    return (s.str.contains(r"[A-Za-z]")).mean() > 0.9

def map_gctx_rows_to_symbols(G: pd.DataFrame, gi: pd.DataFrame) -> pd.DataFrame:
    if index_looks_like_symbols(G.index):
        G.index = pd.Index(G.index.astype(str), name="gene")
        return G
    best_col, best_overlap = None, -1
    gset = set(G.index.astype(str))
    for c in gi.columns:
        if c == "gene":
            continue
        overlap = len(gset & set(gi[c].astype(str)))
        if overlap > best_overlap:
            best_overlap, best_col = overlap, c
    if not best_col or best_overlap == 0:
        raise RuntimeError("Could not map GCTX row IDs to gene_info")
    key2sym = (gi.dropna(subset=[best_col,"gene"])
                 .drop_duplicates(subset=[best_col])
                 .set_index(best_col)["gene"]
                 .to_dict())
    mapped = [key2sym.get(str(r)) for r in G.index]
    keep = pd.Series(mapped, index=G.index).notna()
    G = G.loc[keep, :].copy()
    G.index = pd.Index([key2sym[str(r)] for r in G.index], name="gene")
    G = G[~G.index.duplicated(keep="first")]
    return G

# ------------ aggregation & scoring ------------
def aggregate_signatures_to_drugs(expr_df: pd.DataFrame, col_meta: pd.DataFrame) -> pd.DataFrame:
    """
    expr_df: genes x signatures  (columns must equal col_meta.index)
    col_meta: index = signatures, columns include 'pert_iname'
    Returns: genes x drugs (mean signature per drug)
    """
    # Ensure the indices/columns match perfectly and in order
    if not (expr_df.columns.equals(col_meta.index)):
        # hard align in case something slipped
        col_meta = col_meta.reindex(expr_df.columns)
        assert expr_df.columns.equals(col_meta.index), "expr_df.columns and col_meta.index must match"

    mats = []
    names = []
    # groups maps drug_name -> Index of signature labels (not positions)
    groups = col_meta.groupby("pert_iname").groups
    for drug, sig_labels in groups.items():
        # sig_labels is an Index of column names present in expr_df
        X = expr_df.loc[:, list(sig_labels)]
        v = X.mean(axis=1)
        mats.append(v)
        names.append(drug)

    M = pd.concat(mats, axis=1)
    M.columns = names
    return M


def compute_reversal_for_patient(deg_file: Path, drug_mat: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(deg_file, sep="\t")
    cols = {c.lower(): c for c in d.columns}
    gcol = cols.get("gene","gene")
    lcol = cols.get("logfc","logFC")
    if gcol not in d.columns or lcol not in d.columns:
        raise ValueError(f"{deg_file.name}: need columns 'gene' and 'logFC'")
    x = d.set_index(gcol)[lcol].astype(float)
    v = x.reindex(drug_mat.index).fillna(0.0).values.reshape(-1, 1)
    v = normalize(v, axis=0)                 # (G,1)
    D = normalize(drug_mat.values, axis=0)   # (G,n)
    cos = (v.T @ D).ravel()
    rev = -cos  # higher = better reversal
    return pd.DataFrame({"drug": drug_mat.columns, "reversal_score": rev}).sort_values("reversal_score", ascending=False)

# ------------ main ------------
print(">>> RUNNING 03_lincs_reversal_cosine_v5 (Level-5, writes TSV) <<<")

def main():
    ap = argparse.ArgumentParser(description="Compute per-patient LINCS reversal scores (cosine, LM978).")
    ap.add_argument("--degs-dir", required=True)
    ap.add_argument("--gctx", required=False)          # optional in --smoke-test
    ap.add_argument("--gene-info", required=False)
    ap.add_argument("--sig-info", required=False)
    ap.add_argument("--pert-info", required=False)
    ap.add_argument("--cell-lines", default="MCF7")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Write trivial TSVs per patient (path/write/glob test).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    deg_dir = Path(args.degs_dir)
    deg_files = sorted(deg_dir.glob("DEGs_*.tsv"))
    print(f"[INFO] degs-dir = {deg_dir}")
    print(f"[INFO] found {len(deg_files)} DEG files: {[p.name for p in deg_files]}")
    if not deg_files:
        raise RuntimeError(f"No DEG files found in {args.degs_dir}")

    try:
        if args.smoke_test:
            print("[SMOKE] Writing dummy TSVs (no LINCS)…")
            for f in deg_files:
                pid = f.stem.replace("DEGs_", "")
                out_file = outdir / f"reversal_scores_{pid}.tsv"
                pd.DataFrame(
                    {"drug": ["dummyA", "dummyB"], "reversal_score": [1.0, 0.5]}
                ).to_csv(out_file, sep="\t", index=False)
                print(f"[OK][SMOKE] {pid} -> {out_file}")
            print("[SMOKE] Done.")
            return

        # --- real pipeline (Level-5) ---
        gctx_path = Path(pick_one(args.gctx))
        gi_path   = Path(pick_one(args.gene_info))
        si_path   = Path(pick_one(args.sig_info))
        pi_path   = Path(pick_one(args.pert_info))

        print("[INFO] Loading LM978 gene info …")
        gi = load_lm978_geneinfo(gi_path)

        # Load GCTX and align by sig_id (Level-5)
        G, col_meta, overlap_n = load_level5_gctx_and_align_by_sig(
            gctx_path, si_path, pi_path, args.cell_lines
        )
        print(f"[INFO] signatures kept after intersection: {overlap_n}")
        print(f"[INFO] GCTX matrix: {G.shape[0]} features x {G.shape[1]} signatures")

        # Map row IDs → gene symbols
        print("[INFO] Mapping GCTX row IDs → gene symbols …")
        G.index = G.index.astype(str)
        G = map_gctx_rows_to_symbols(G, gi)
        print(f"[INFO] Mapped features to symbols: {G.shape[0]} genes")

        # Normalize signatures, aggregate → drugs, normalize again
        G = pd.DataFrame(normalize(G.values, axis=0), index=G.index, columns=G.columns)
        print("[INFO] Aggregating signatures → drugs …")
        drug_mat = aggregate_signatures_to_drugs(G, col_meta)
        drug_mat = pd.DataFrame(normalize(drug_mat.values, axis=0), index=drug_mat.index, columns=drug_mat.columns)
        print(f"[INFO] drug_mat: {drug_mat.shape[0]} genes x {drug_mat.shape[1]} drugs")

        # Score each patient
        print("[INFO] Computing per-patient reversal …")
        for f in deg_files:
            pid = f.stem.replace("DEGs_", "")
            out = compute_reversal_for_patient(f, drug_mat)
            out_file = outdir / f"reversal_scores_{pid}.tsv"
            out.to_csv(out_file, sep="\t", index=False)
            print(f"[OK] {pid}: wrote {out.shape[0]} drug scores -> {out_file}")

        print("[DONE] All patients processed.")

    except Exception as e:
        err_file = outdir / "run03_error.log"
        with open(err_file, "w", encoding="utf-8") as fh:
            fh.write("ERROR in 03_lincs_reversal_cosine_v5\n")
            fh.write(str(e) + "\n\n")
            fh.write(traceback.format_exc())
        print(f"[ERROR] {e}")
        print(f"[ERROR] See log: {err_file}")

if __name__ == "__main__":
    main()
