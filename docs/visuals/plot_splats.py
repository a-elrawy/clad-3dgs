"""Draw a schematic of differentiable 3D Gaussian splatting."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


OUTPUT = Path(__file__).with_name("differentiable-splats.png")

fig, ax = plt.subplots(figsize=(12, 5.4))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.4)
ax.axis("off")

ink = "#172c3c"
muted = "#506171"
colors = ("#e9954b", "#4a9a91", "#7289c8")


def panel(x, title):
    ax.add_patch(FancyBboxPatch(
        (x, 1.0), 3.25, 3.2, boxstyle="round,pad=0.1,rounding_size=0.14",
        facecolor="#f8fafb", edgecolor="#d4dde3", linewidth=1.3))
    ax.text(x + 0.15, 3.88, title, ha="left", fontsize=12,
            weight="bold", color=ink)


def arrow(start, end, color=muted):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=18, linewidth=2.0, color=color))


fig.text(0.06, 0.96, "3D Gaussian Splatting: scene parameters to gradients",
         fontsize=18, weight="bold", color=ink)
fig.text(0.06, 0.90, "Schematic of projection, blending, and the training signal",
         fontsize=10.5, color=muted)

panel(0.35, "1  Scene parameters")
panel(4.38, "2  Projected splats")
panel(8.41, "3  Pixel and gradient")

# A simplified 3D scene: anisotropic Gaussian volumes viewed by one camera.
ax.plot([0.75, 1.18, 0.75, 0.75], [2.10, 2.34, 2.58, 2.10],
        color=ink, linewidth=2)
ax.plot([1.18, 1.42], [2.34, 2.34], color=ink, linewidth=2)
ax.text(0.68, 1.69, "camera", fontsize=9, color=muted)
for x, y, width, height, angle, color in (
    (2.0, 2.65, 0.78, 1.18, 28, colors[0]),
    (2.65, 2.18, 1.05, 0.63, -28, colors[1]),
    (2.80, 2.97, 0.65, 0.96, 38, colors[2]),
):
    ax.add_patch(Ellipse((x, y), width, height, angle=angle,
                         facecolor=color, edgecolor=color, alpha=0.63))
ax.text(0.64, 1.20, "position · scale · rotation\nopacity · SH color",
        fontsize=9.6, color=muted, linespacing=1.5)

# Image plane with one pixel receiving several overlapping splats.
for x in [4.86 + 0.33 * i for i in range(8)]:
    ax.plot([x, x], [1.57, 3.36], color="#e0e6ea", linewidth=0.7)
for y in [1.57 + 0.30 * i for i in range(7)]:
    ax.plot([4.86, 7.17], [y, y], color="#e0e6ea", linewidth=0.7)
for x, y, width, height, angle, color in (
    (5.85, 2.60, 1.35, 0.66, 22, colors[0]),
    (6.20, 2.42, 1.17, 0.95, -33, colors[1]),
    (6.35, 2.81, 0.86, 0.60, 37, colors[2]),
):
    ax.add_patch(Ellipse((x, y), width, height, angle=angle,
                         facecolor=color, edgecolor="none", alpha=0.43))
ax.add_patch(Rectangle((5.85, 2.47), 0.33, 0.30, fill=False,
                       edgecolor=ink, linewidth=2.4))
ax.text(4.68, 1.20, "Overlapping ellipses contribute\nto the highlighted pixel",
        fontsize=9.6, color=muted, linespacing=1.5)

# The image loss produces a gradient that flows back to Gaussian parameters.
ax.text(8.74, 3.18, "front-to-back blending", fontsize=10.5, color=ink)
for i, color in enumerate(colors):
    ax.add_patch(Rectangle((8.77 + i * 0.28, 2.60), 0.50, 0.32,
                           facecolor=color, alpha=0.8, edgecolor="white"))
arrow((9.85, 2.75), (10.37, 2.75))
ax.add_patch(Rectangle((10.45, 2.48), 0.52, 0.52,
                       facecolor="#888d91", edgecolor=ink, linewidth=1.2))
ax.text(10.27, 2.12, "rendered pixel", fontsize=9.4, color=muted)
ax.text(8.74, 1.65, "image loss  →  gradients", fontsize=11,
        weight="bold", color="#16897e")
ax.text(8.74, 1.22, "update Gaussian parameters", fontsize=9.6, color=muted)

arrow((3.72, 2.59), (4.17, 2.59))
arrow((7.75, 2.59), (8.20, 2.59))

fig.savefig(OUTPUT, dpi=160, facecolor="white")
plt.close(fig)
