import pandas as pd
from pathlib import Path
from cmapPy.pandasGEXpress.parse_gctx import parse as parse_gctx

gctx = r"C:\Projects\drug-repurpose-bc\data\lincs\GSE92742_Broad_LINCS_Level2_GEX_delta_n49216x978.gctx"
sig  = r"C:\Projects\drug-repurpose-bc\data\lincs\GSE92742_Broad_LINCS_sig_info.txt.gz"
outd = Path(r"C:\Projects\drug-repurpose-bc\data\lincs")
outd.mkdir(parents=True, exist_ok=True)

print("[DIAG] Reading GCTX column IDs …", flush=True)
g = parse_gctx(gctx, rid=None, cid=None)
cids = pd.Index(g.data_df.columns.astype(str))
print(f"[DIAG] GCTX columns: {len(cids)}")

print("[DIAG] Reading sig_info and exploding distil_id …", flush=True)
si = pd.read_csv(sig, sep="\t", low_memory=False)
si = si[si["pert_type"].astype(str).str.lower().eq("trt_cp")]
si = si[si["cell_id"].astype(str).str.upper().eq("MCF7")]
si = si[~si["distil_id"].isna()]
si["distil_id"] = si["distil_id"].astype(str)

# explode on '|'
rows = []
for s in si["distil_id"]:
    rows.extend([x.strip() for x in s.split("|") if x.strip()])
distil = pd.Index(pd.unique(pd.Series(rows, dtype=str)))

print(f"[DIAG] unique exploded distil_id (MCF7, trt_cp): {len(distil)}")

# exact set operations (normalize by stripping whitespace)
cset = set(map(str.strip, cids.tolist()))
dset = set(map(str.strip, distil.tolist()))
both = cset & dset
print(f"[DIAG] EXACT overlap count: {len(both)}")

# save samples for eyeballing
(outd / "diag_gctx_cols_50.txt").write_text("\n".join(list(cids[:50])), encoding="utf-8")
(outd / "diag_distil_50.txt").write_text("\n".join(list(distil[:50])), encoding="utf-8")
(outd / "diag_overlap_50.txt").write_text("\n".join(list(sorted(both))[:50]), encoding="utf-8")

# also show if any prefix clearly matches
starts = ["ASG001_MCF7", "CPC", "DOS", "HOG", "ERG003_VCAP", "MUC.CP"]
for pref in starts:
    c_pref = sum(x.startswith(pref) for x in cids)
    d_pref = sum(x.startswith(pref) for x in distil)
    print(f"[DIAG] prefix {pref!r}: in GCTX={c_pref}, in distil={d_pref}")

print("[DIAG] Wrote:")
print("  - diag_gctx_cols_50.txt")
print("  - diag_distil_50.txt")
print("  - diag_overlap_50.txt")
