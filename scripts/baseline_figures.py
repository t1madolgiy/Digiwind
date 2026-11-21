"""Rysunki wyników bazowych Baltica 2 (do prezentacji, po angielsku).

  1. wind_rose_imgw.png      — udział energii wiatru (ws³) w sektorach 10°
  2. turbine_wake_loss.png   — straty wake każdej turbiny (GCH)
  3. wake_models_aep.png     — AEP dla 3 modeli wake

Uruchomienie:  python scripts/baseline_figures.py <katalog_wyjściowy>
"""
import sys, logging, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from src.wind_data import WindDataLoader
from src.farm_model import FarmModel
from src.site_baltica2 import baltica2_site, rotated_grid_layout

INK, BODY, MUTED, GRID = "#14213D", "#3F4B5C", "#6B7785", "#DDE4E8"
TEAL, TEAL_DARK = "#2A9DA8", "#136F86"
SEQ = LinearSegmentedColormap.from_list("seq", ["#E3F3F4", "#7CCBD0", "#2A9DA8", "#136F86", "#0E3F57"])
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": BODY,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})


def wind_rose(L, out):
    wd, ws = L.wind_directions, L.wind_speeds
    sec = (np.floor(((wd + 5) % 360) / 10) * 10).astype(int)
    e = np.bincount(sec // 10, weights=ws ** 3, minlength=36)
    e = e / e.sum() * 100
    theta = np.radians(np.arange(0, 360, 10))
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.bar(theta, e, width=np.radians(9), color=SEQ(0.35 + 0.65 * e / e.max()), edgecolor="white", linewidth=1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=14, color=BODY)
    ax.set_yticks([3, 6, 9]); ax.set_yticklabels(["3%", "6%", "9%"], fontsize=10, color=MUTED)
    ax.set_rlabel_position(135)
    ax.grid(color=GRID, linewidth=1)
    ax.spines["polar"].set_color(GRID)
    top = int(np.argmax(e)) * 10
    ax.set_title(f"Share of wind energy by direction (strongest: {top}°)", fontsize=13, pad=18, color=INK)
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)
    return e


def turbine_losses(L, site, x, y, out):
    rose = L.to_wind_rose(wd_step=10.0, ws_step=2.0)
    f = FarmModel("gch", "iea_15MW"); f.set_layout_custom(x, y); f.set_wind_shear(L.wind_shear)
    f.set_wind_data(rose); f.run()
    aep_t = f.get_aep_per_turbine_gwh()
    one = FarmModel("gch", "iea_15MW"); one.set_layout_custom(np.array([0.0]), np.array([0.0]))
    one.set_wind_shear(L.wind_shear); one.set_wind_data(rose); one.run()
    free = one.get_aep_gwh()
    loss = (1 - aep_t / free) * 100
    px, py = site.polygon.exterior.xy
    k = 1e-3
    fig, ax = plt.subplots(figsize=(10, 6.6), dpi=300)
    ax.fill(np.array(px) * k, np.array(py) * k, color="#F3F6F7", zorder=1)
    ax.plot(np.array(px) * k, np.array(py) * k, color=TEAL_DARK, lw=1.4, zorder=2)
    sc = ax.scatter(x * k, y * k, c=loss, cmap=SEQ, s=70, edgecolors="white", linewidths=1.2, zorder=3,
                    vmin=0, vmax=np.ceil(loss.max()))
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Wake loss per turbine [%]", color=BODY); cb.outline.set_edgecolor(GRID)
    worst, best = int(np.argmax(loss)), int(np.argmin(loss))
    for i, lab in ((worst, f"{loss[worst]:.0f}%"), (best, f"{loss[best]:.0f}%")):
        ax.annotate(lab, (x[i] * k, y[i] * k), xytext=(8, 8), textcoords="offset points", fontsize=11, color=INK)
    ax.set_aspect("equal"); ax.set_xlim(-1, 21); ax.set_ylim(-1, 12.8)
    ax.set_xlabel("x [km] (east)"); ax.set_ylabel("y [km] (north)")
    ax.grid(True, color=GRID, alpha=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(f"Wake loss of each turbine (GCH) — farm average {100 - aep_t.sum() / (free * len(x)) * 100:.1f}%",
                 fontsize=13, color=INK)
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)
    return loss


def wake_models(L, x, y, out):
    rose = L.to_wind_rose(wd_step=10.0, ws_step=2.0)
    names = {"gch": "GCH", "empirical_gauss": "Empirical Gauss", "turbopark": "TurbOPark"}
    res = []
    for key, lab in names.items():
        f = FarmModel(key, "iea_15MW"); f.set_layout_custom(x, y); f.set_wind_shear(L.wind_shear)
        f.set_wind_data(rose); f.run()
        res.append((lab, f.get_aep_gwh(), f.get_wake_losses_percent()))
    gross = res[0][1] / (1 - res[0][2] / 100)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
    labs = [r[0] for r in res][::-1]; vals = [r[1] for r in res][::-1]; wl = [r[2] for r in res][::-1]
    ax.barh(labs, vals, height=0.55, color=TEAL, edgecolor="white")
    ax.axvline(gross, color=MUTED, lw=1.5, ls="--")
    ax.text(gross, 2.45, f"no wake {gross:.0f}", ha="right", va="bottom", fontsize=10, color=MUTED)
    for i, (v, w) in enumerate(zip(vals, wl)):
        ax.text(v - 80, i, f"{v:.0f} GWh  (−{w:.0f}%)", ha="right", va="center", fontsize=12, color="white",
                fontweight="bold")
    ax.set_xlim(0, gross * 1.04)
    ax.set_xlabel("AEP [GWh/year]")
    ax.tick_params(axis="y", labelsize=12, colors=INK)
    ax.grid(True, axis="x", color=GRID); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Baseline AEP by wake model (107 × IEA 15 MW)", fontsize=13, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)
    return res, gross


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    site = baltica2_site()
    x, y, _ = rotated_grid_layout(site, n_turbines=107, D=240.0)
    L = WindDataLoader.from_imgw()
    e = wind_rose(L, out / "wind_rose_imgw.png")
    print("strongest sector:", int(np.argmax(e)) * 10, "share", round(e.max(), 1), "%;",
          "WSW-W 230-280 share:", round(e[23:29].sum(), 1), "%")
    loss = turbine_losses(L, site, x, y, out / "turbine_wake_loss.png")
    print(f"turbine loss: min {loss.min():.1f}% max {loss.max():.1f}% mean {loss.mean():.1f}%")
    res, gross = wake_models(L, x, y, out / "wake_models_aep.png")
    print("gross", round(gross), res)


if __name__ == "__main__":
    main()
