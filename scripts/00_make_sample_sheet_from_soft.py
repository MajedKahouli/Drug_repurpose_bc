import argparse, gzip, re
from pathlib import Path
import pandas as pd
from collections import Counter

def parse_soft(soft_path: Path) -> pd.DataFrame:
    samples = []
    cur = None
    with gzip.open(soft_path, "rt", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.startswith("^SAMPLE = GSM"):
                if cur: samples.append(cur)
                cur = {"sample_id": ln.split("=",1)[1].strip()}
            elif ln.startswith("!Sample_title"):
                cur["title"] = ln.split("=",1)[1].strip()
            elif ln.startswith("!Sample_source_name_ch1"):
                cur["source_name"] = ln.split("=",1)[1].strip()
            elif ln.startswith("!Sample_characteristics_ch1"):
                cur.setdefault("characteristics", []).append(ln.split("=",1)[1].strip())
        if cur: samples.append(cur)

    # flatten
    for s in samples:
        s["characteristics"] = "; ".join(s.get("characteristics", []))
    df = pd.DataFrame(samples)

    # infer tumor/normal from any text available
    def infer(t: str) -> str:
        t = (t or "").lower()
        if re.search(r"\b(adjacent normal|normal|benign|non[- ]tumou?r)\b", t): return "normal"
        if re.search(r"\b(tumou?r|carcinoma|cancer|malignan)\b", t): return "tumor"
        return "unknown"

    combo = (df["title"].fillna("") + " " +
             df["source_name"].fillna("") + " " +
             df["characteristics"].fillna(""))
    df["condition"] = [infer(x) for x in combo]

    # patient_id heuristic from title (pairs appear twice with tumor/normal tokens removed)
    def base_id(s: str) -> str:
        s = (s or "").lower()
        s = re.sub(r"\b(adjacent )?normal\b", "", s)
        s = re.sub(r"\btumou?r\b", "", s)
        s = re.sub(r"\bbreast\b", "", s)
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    base = df["title"].fillna("").apply(base_id).tolist()
    cnt = Counter(base)
    pid_map, pid_seq, pids = {}, 1, []
    for b in base:
        if b and cnt[b] == 2:
            if b not in pid_map:
                pid_map[b] = f"P{pid_seq:02d}"; pid_seq += 1
            pids.append(pid_map[b])
        else:
            pids.append("")
    df["patient_id"] = pids

    return df[["sample_id","title","source_name","characteristics","condition","patient_id"]]

def main():
    ap = argparse.ArgumentParser(description="Create sample_sheet.csv from GEO family SOFT.")
    ap.add_argument("--soft", required=True, help="Path to GSE####_family.soft.gz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = parse_soft(Path(args.soft))
    print(f"[INFO] Samples: {len(df)}")
    print(df["condition"].value_counts(dropna=False))
    unk = (df["condition"] == "unknown").sum()
    if unk:
        print(f"[WARN] {unk} samples with 'unknown' condition — please edit the CSV if needed.")
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out}")

if __name__ == "__main__":
    main()
