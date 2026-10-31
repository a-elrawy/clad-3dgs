"""Plot held-out PSNR changes from the recorded 30k Train result."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
ROWS = ROOT / "results" / "held-out-views.csv"
OUTPUT = Path(__file__).with_name("psnr-by-view.png")

with ROWS.open(newline="") as handle:
    views = [
        (row["camera"], float(row["psnr_delta_db"]))
        for row in csv.DictReader(handle)
        if row["endpoint"] == "30000" and row["split"] == "heldout"
    ]

if len(views) != 38:
    raise ValueError(f"Expected 38 held-out views, found {len(views)}")

views.sort(key=lambda view: (view[1], view[0]))
names, changes = zip(*views)
colors = ["#bd4b43" if change < 0 else "#237d75" for change in changes]

fig, ax = plt.subplots(figsize=(10, 11))
fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.07)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.barh(range(len(views)), changes, height=0.76, color=colors)
ax.axvline(0, color="#263746", linewidth=1)
ax.set_yticks(range(len(views)), names, fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-2.15, 2.65)
ax.set_xlabel("Hybrid Clad − unchanged native PSNR (dB)", fontsize=11)
ax.grid(axis="x", color="#dde3e7", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)

mean_change = sum(changes) / len(changes)
fig.text(0.16, 0.96, "Held-out PSNR change across 38 views", ha="left",
         fontsize=16, weight="bold", color="#172c3c")
fig.text(0.16, 0.93,
         f"Tanks and Temples Train · 30k steps · seed 42 · mean {mean_change:+.3f} dB",
         ha="left", fontsize=10, color="#506171")
fig.savefig(OUTPUT, dpi=160, facecolor="white")
plt.close(fig)
