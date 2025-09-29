import sys, ast, pandas as pd

raw_path = r"C:\Projects\drug-repurpose-bc\work\scores\MCF7_scores_raw.txt"
out_path = r"C:\Projects\drug-repurpose-bc\work\scores\MCF7_scores.tsv"

s = open(raw_path, 'r', encoding='utf-8', errors='ignore').read().strip()

# The script seems to print a Python set/list of strings like 'SIG:BRD:score'
try:
    data = ast.literal_eval(s)  # set or list
    if isinstance(data, (set, tuple)):
        data = list(data)
except Exception:
    # Fallback: split by comma
    data = [x.strip() for x in s.strip("{}[]").split(",") if x.strip()]
    data = [x.strip("'\"") for x in data]

rows = []
for item in data:
    # split on the LAST colon => left is signature id, right is score
    sig, score = item.rsplit(":", 1)
    rows.append({"signature": sig, "score": float(score)})

pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
print("wrote", out_path, "n=", len(rows))
