"""Rysunki: dlaczego gruba róża wiatrów zawyża zysk z optymalizacji (EN, do prezentacji).

  rose_aliasing_sketch.png  — 12 kierunków co 30° vs 72 co 5°: turbina „między” kierunkami
  rose_resolution_gain.png  — zysk tego samego layoutu GA wg rozdzielczości róży (test 21.11)

Uruchomienie:  python scripts/rose_resolution_figures.py <katalog_wyjściowy>
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, BODY, MUTED, GRID = "#14213D", "#3F4B5C", "#6B7785", "#DDE4E8"
TEAL, TEAL_DARK, ORANGE = "#2A9DA8", "#136F86", "#D9822B"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": BODY,
                     "xtick.color": MUTED, "ytick.color": MUTED})

# wynik testu 21.11 (outputs/m2_test, layout GA seed 0, model GCH, wiatr IMGW)
RESULTS = [("30°", 1.53), ("15°", 0.26), ("10°", 0.07), ("5°", -0.24)]


def sketch(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    target_deg = 15.0  # druga turbina leży 15° od kierunku wiatru
    for ax, step, title in ((axes[0], 30, "Coarse rose: wind from 12 directions (every 30°)"),
                            (axes[1], 5, "Fine rose: wind from 72 directions (every 5°)")):
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        hit = False
        for d in np.arange(0, 360, step):
            t = np.radians(90 - d)  # kierunek przepływu narysowany jako promień
            ex, ey = np.cos(t), np.sin(t)
            # ślad za turbiną centralną: wąski stożek wzdłuż kierunku
            half = np.radians(3.0)
            pts = np.array([[0, 0], [np.cos(t - half), np.sin(t - half)], [np.cos(t + half), np.sin(t + half)]])
            aligned = abs(((d - target_deg + 180) % 360) - 180) < 3.0
            hit |= aligned
            ax.fill(pts[:, 0], pts[:, 1], color=ORANGE if aligned else TEAL, alpha=0.55 if aligned else 0.18, lw=0)
        ax.scatter([0], [0], s=260, color=TEAL_DARK, zorder=5, edgecolors="white", linewidths=2)
        tt = np.radians(90 - target_deg)
        ax.scatter([0.8 * np.cos(tt)], [0.8 * np.sin(tt)], s=260, color=ORANGE if hit else INK, zorder=6,
                   edgecolors="white", linewidths=2)
        ax.text(0, -1.18, "turbine B is in a wake: the model sees the loss" if hit
                else "turbine B sits between directions: no wake, loss hidden",
                ha="center", fontsize=13, color=ORANGE if hit else INK)
        ax.text(0.06, -0.12, "A", fontsize=14, color=TEAL_DARK, ha="left")
        ax.text(0.8 * np.cos(tt) + 0.07, 0.8 * np.sin(tt), "B", fontsize=14, color=INK, va="center")
        ax.set_title(title, fontsize=14, color=INK)
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)


def gain_chart(out):
    labs = [r[0] for r in RESULTS]; vals = [r[1] for r in RESULTS]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    colors = [TEAL if v >= 0 else ORANGE for v in vals]
    bars = ax.bar(labs, vals, width=0.55, color=colors, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.06 if v >= 0 else -0.06), f"{v:+.2f}%",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=14, color=INK, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlabel("Wind rose direction step used to evaluate the layout")
    ax.set_ylabel("AEP gain vs baseline [%]")
    ax.grid(True, axis="y", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_title("Same GA layout, different wind roses: the gain disappears", fontsize=13, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    o = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/figures")
    o.mkdir(parents=True, exist_ok=True)
    sketch(o / "rose_aliasing_sketch.png")
    gain_chart(o / "rose_resolution_gain.png")
    print("ok")
