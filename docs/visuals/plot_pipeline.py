"""Draw the measured hybrid training pipeline at a high level."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).with_name("hybrid-pipeline.png")

fig, ax = plt.subplots(figsize=(12, 5.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.6)
ax.axis("off")


def box(x, y, width, height, label, color, fontsize=12):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.12,rounding_size=0.15",
        linewidth=1.5, edgecolor=color, facecolor="white"))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
            fontsize=fontsize, color="#172c3c", linespacing=1.35)


def arrow(start, end, color="#506171", style="-"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.1,
        color=color, linestyle=style, connectionstyle="arc3"))


fig.text(0.06, 0.95, "Hybrid 3DGS training: where Clad runs", fontsize=18,
         weight="bold", color="#172c3c")
fig.text(0.06, 0.90,
         "Selected CUDA adjoints are generated; rasterization and geometry backward remain native",
         fontsize=10.5, color="#506171")

box(0.5, 3.8, 2.3, 0.8, "Gaussian\nparameters", "#748696")
box(3.65, 3.8, 3.6, 0.8, "Native forward\nrasterization", "#748696")
box(8.1, 3.8, 2.7, 0.8, "Rendered image\nand loss", "#748696")
box(8.1, 1.55, 2.7, 1.0, "Native pixel +\ngeometry backward", "#748696")
box(3.65, 1.55, 3.6, 1.0,
    "Clad-generated CUDA adjoints\nSH, view direction, activations", "#16897e",
    fontsize=10.5)
box(0.5, 1.55, 2.3, 1.0, "Dense Adam\nall parameters", "#be7a20")

arrow((2.95, 4.2), (3.5, 4.2))
arrow((7.4, 4.2), (7.95, 4.2))
arrow((9.45, 3.65), (9.45, 2.7))
arrow((7.95, 2.05), (7.4, 2.05), "#16897e")
arrow((3.5, 2.05), (2.95, 2.05), "#16897e")

# The native geometry derivatives also reach the optimizer without Clad.
ax.plot([9.45, 9.45, 1.65], [1.4, 0.8, 0.8], color="#748696",
        linewidth=1.8, linestyle="--")
arrow((1.65, 0.8), (1.65, 1.4), "#748696", "--")
ax.text(5.45, 0.94, "Native geometry gradients", ha="center", fontsize=10,
        color="#506171", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

fig.text(0.06, 0.035,
         "Direct input reads, warp accumulation, and optimizer fusion are explicit integration choices.",
         fontsize=9.5, color="#506171")
fig.savefig(OUTPUT, dpi=160, facecolor="white")
plt.close(fig)
