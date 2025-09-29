import argparse
import pandas as pd
from cmapPy.pandasGEXpress.parse_gctx import parse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gctx", required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    print("[INFO] Loading GCTX metadata (columns only)…")
    g = parse(args.gctx, rid=None, cid=None)  # loads data + metadata; we’ll only use col metadata
    meta = g.col_metadata_df

    # Save full metadata and a small head preview (easier to upload)
    meta.to_csv(args.out, index=True)
    head_path = args.out.replace(".csv", "_head500.csv")
    meta.head(500).to_csv(head_path, index=True)

    print("[OK] Saved full col metadata to:", args.out, "shape:", meta.shape)
    print("[OK] Saved head(500) to:", head_path)
    print("Columns present:", list(meta.columns))

if __name__ == "__main__":
    main()
