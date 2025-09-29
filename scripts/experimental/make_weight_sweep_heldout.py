# save as make_weight_sweep_heldout.py and run in the parent folder that contains
# directories like: rankings_w651520_metrics, rankings_w701020_metrics, etc.

import os, re, glob
import pandas as pd
import matplotlib.pyplot as plt

# ---- config: adjust if your naming/columns differ ----
GLOB_PATTERNS = [
    "rankings_w*_metrics/metrics_summary.csv",
    "rankings_w*/metrics_summary.csv",         # fallback, if no _metrics suffix
]
# Columns expected in metrics_summary.csv (held-out across folds)
COL_TOP25 = "top25_any_hit_mean"              # e.g., mean fraction of patients
COL_AUROC_MED = "auroc_median"                # median across patients
OUT1 = "fig_weight_sweep_top25_HELDOUT.png"
OUT2 = "fig_weight_sweep_auroc_HELDOUT.png"

rows = []
for pat in GLOB_PATTERNS:
    for csv in glob.glob(pat):
        m = re.search(r"w(\d{2})(\d{2})(\d{2})", csv.replace("\\", "/"))
        if not m:
            continue
        wr = int(m.group(1))/100.0
        wp = int(m.group(2))/100.0
        wo = int(m.group(3))/100.0
        df = pd.read_csv(csv)
        # robustly pick the single summary row
        s = df.iloc[0]
        rows.append({
            "wr": wr, "wp": wp, "wo": wo,
            "top25": float(s[COL_TOP25]),
            "auroc_med": float(s[COL_AUROC_MED]),
            "src": csv
        })

data = pd.DataFrame(rows)
if data.empty:
    raise SystemExit("No metrics found. Check GLOB_PATTERNS and folder names.")

# Sort for clean plotting
data = data.sort_values(["wp","wr"])

# -------- Top-25 any-hit (HELD-OUT) --------
plt.figure(figsize=(8,4.3), dpi=180)
for wp, sub in data.groupby("wp"):
    plt.plot(sub["wr"], sub["top25"], marker="o", label=f"wp={wp:.2f}")
# highlight (0.60, 0.20, 0.20) if present
p060 = data[(data.wr==0.60) & (data.wp==0.20)]
if not p060.empty:
    x, y = p060.iloc[0][["wr","top25"]]
    plt.scatter([x],[y], s=140, facecolors='none', edgecolors='k', linewidths=2)
    plt.annotate("(0.60,0.20,0.20)", (x,y), xytext=(6,6), textcoords="offset points", fontsize=9)
plt.ylim(0,1)
plt.xlabel("Weight on reversal (wr)")
plt.ylabel("Top-25 any-hit (held-out mean)")
plt.title("Held-out Top-25 any-hit over weight grid; wo = 1 - wr - wp")
plt.grid(True, ls="--", alpha=0.4)
plt.legend(title="Weight on proximity (wp)", ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(OUT1)

# -------- AUROC (median, HELD-OUT) --------
plt.figure(figsize=(8,4.3), dpi=180)
for wp, sub in data.groupby("wp"):
    plt.plot(sub["wr"], sub["auroc_med"], marker="o", label=f"wp={wp:.2f}")
p060 = data[(data.wr==0.60) & (data.wp==0.20)]
if not p060.empty:
    x, y = p060.iloc[0][["wr","auroc_med"]]
    plt.scatter([x],[y], s=140, facecolors='none', edgecolors='k', linewidths=2)
    plt.annotate("(0.60,0.20,0.20)", (x,y), xytext=(6,6), textcoords="offset points", fontsize=9)
plt.xlabel("Weight on reversal (wr)")
plt.ylabel("AUROC (held-out median)")
plt.title("Held-out AUROC over weight grid; wo = 1 - wr - wp")
plt.grid(True, ls="--", alpha=0.4)
plt.legend(title="Weight on proximity (wp)", ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(OUT2)

print("Wrote:", OUT1, "and", OUT2)
