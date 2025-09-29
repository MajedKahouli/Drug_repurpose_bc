# -*- coding: utf-8 -*-
# Collapse Affymetrix GPL96 probes -> genes (max IQR per gene)
# Usage:
#   python scripts/01_collapse_gpl96.py --data-dir "C:\Projects\drug-repurpose-bc\data\geo\gse15852"

import argparse, gzip, io
from pathlib import Path
import pandas as pd
import numpy as np

def read_gpl96(gpl_path: Path) -> pd.DataFrame:
    """Read GEO GPL96 annotation (SOFT .annot.gz) by skipping header lines until the 'ID\t' table header."""
    print(f"[INFO] Reading GPL96 annotation: {gpl_path}")
    if not gpl_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {gpl_path}")

    # Read raw lines and find the tabular header
    import gzip, io
    with gzip.open(gpl_path, "rt", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("ID\t"))
    except StopIteration:
        # Sometimes it's 'ID' separated by spaces; try a looser match
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("ID"))
    table_txt = "".join(lines[start:])

    # Now parse the annotation table
    df = pd.read_csv(io.StringIO(table_txt), sep="\t", dtype=str, low_memory=False)
    # Robustly find ID and Gene Symbol columns
    cols_lower = {c.lower(): c for c in df.columns}
    # common header is exactly "ID"
    id_col = "ID" if "ID" in df.columns else cols_lower.get("id")
    if id_col is None:
        # fallback: first column
        id_col = df.columns[0]
    # Gene Symbol column name varies a bit but contains 'symbol'
    sym_col = next((c for c in df.columns if "symbol" in c.lower()), None)
    if sym_col is None:
        raise RuntimeError("Could not find a 'Gene Symbol' column in GPL96 annotation.")

    m = df[[id_col, sym_col]].copy()
    m.columns = ["probe_id", "gene_symbol"]

    # Clean & filter
    m["gene_symbol"] = m["gene_symbol"].astype(str).str.strip()
    m = m[~m["gene_symbol"].isin(["", "—", "-", "NA", "na", "null"])]
    m = m[~m["probe_id"].astype(str).str.startswith("AFFX-")]      # drop controls
    m["gene_symbol"] = m["gene_symbol"].str.split("///").str[0].str.strip()
    m["gene_symbol"] = m["gene_symbol"].str.replace(r"\s+", "", regex=True)
    m = m[m["gene_symbol"].str.match(r"^[A-Za-z0-9\-\.]+$")]
    m = m.drop_duplicates(subset=["probe_id"])

    print(f"[INFO] GPL96 rows after cleaning: {len(m):,} (kept columns: {id_col!r}, {sym_col!r})")
    return m


def read_series_matrix(series_path: Path) -> pd.DataFrame:
    """Read GEO Series Matrix by slicing between !series_matrix_table_begin/end."""
    print(f"[INFO] Reading series matrix: {series_path}")
    if not series_path.exists():
        raise FileNotFoundError(f"Missing series matrix: {series_path}")

    with series_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Locate the table (your file has these markers)
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "!series_matrix_table_begin")
        end   = next(i for i, l in enumerate(lines) if l.strip() == "!series_matrix_table_end")
    except StopIteration:
        # Fallback: older files – try header search
        try:
            start = next(i for i, l in enumerate(lines) if l.startswith("ID_REF"))
            end = len(lines)
        except StopIteration:
            raise RuntimeError("Could not find table markers or 'ID_REF' in series matrix.")

    # Build table text (skip the begin marker line; next line is the header)
    table_txt = "".join(lines[start + 1 : end])

    # Parse as TSV; normalize header text (strip quotes/whitespace)
    df = pd.read_csv(io.StringIO(table_txt), sep="\t", dtype=str, low_memory=False)
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    # Ensure first column is named ID_REF (some files quote/rename it)
    if df.columns.size == 0:
        raise RuntimeError("Parsed empty table from series matrix.")
    if df.columns[0] != "ID_REF":
        df = df.rename(columns={df.columns[0]: "ID_REF"})

    print(f"[INFO] Probes in matrix: {df.shape[0]:,}; samples: {df.shape[1]-1:,}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder with GSE15852_series_matrix.txt and GPL96.annot.gz")
    ap.add_argument("--series", default="GSE15852_series_matrix.txt")
    ap.add_argument("--gpl",    default="GPL96.annot.gz")
    ap.add_argument("--out-prefix", default="expr_gene_level")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    series_path = data_dir / args.series
    gpl_path = data_dir / args.gpl
    assert series_path.exists(), f"Missing: {series_path}"
    assert gpl_path.exists(),    f"Missing: {gpl_path}"

    print("Reading GPL96 annotation...")
    gpl = read_gpl96(gpl_path)
    print(f"GPL96 rows after cleaning: {len(gpl):,}")

    print("Reading series matrix...")
    expr_probe = read_series_matrix(series_path)
    print(f"Probes in matrix: {expr_probe.shape[0]:,}; samples: {expr_probe.shape[1]-1:,}")

    print("Joining probe->gene...")
    expr = expr_probe.merge(gpl, left_on="ID_REF", right_on="probe_id", how="inner").drop(columns=["probe_id"])
    sample_cols = [c for c in expr.columns if c not in ["ID_REF","gene_symbol"]]
    X = expr[sample_cols].apply(pd.to_numeric, errors="coerce")

    iqr = X.quantile(0.75, axis=1) - X.quantile(0.25, axis=1)
    expr["probe_IQR"] = iqr.values
    expr["probe_mean"] = X.mean(axis=1).values

    idx = expr.groupby("gene_symbol")["probe_IQR"].idxmax()
    expr_sel = expr.loc[idx].copy()
    gene_level = expr_sel.set_index("gene_symbol")[sample_cols].sort_index()

    out_matrix = data_dir / f"{args.out_prefix}.tsv"
    out_log    = data_dir / "probe_selection.tsv"
    gene_level.to_csv(out_matrix, sep="\t")
    expr_sel[["ID_REF","gene_symbol","probe_IQR","probe_mean"]].to_csv(out_log, sep="\t", index=False)

    print(f"Saved: {out_matrix}  (genes={gene_level.shape[0]:,}, samples={gene_level.shape[1]:,})")
    print(f"Saved: {out_log}")
    if not (10_000 <= gene_level.shape[0] <= 12_500):
        print("NOTE: gene count is outside typical GPL96 range; double-check mapping/filters.")

if __name__ == "__main__":
    main()
