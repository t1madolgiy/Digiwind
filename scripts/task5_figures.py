"""Wykresy dla Task 5 z mocy farmy co 3 h (layout bazowy, IMGW 1991–2010). Po angielsku.

  t5_monthly_energy.png   — energia w miesiącu: średnia z 20 lat + najsłabszy/najlepszy rok
  t5_annual_energy.png    — energia każdego roku 1991–2010
  t5_power_duration.png   — krzywa czasu trwania mocy farmy
  t5_daily_profile.png    — średnia moc o danej godzinie: zima vs lato

Uruchomienie:  python scripts/task5_figures.py <farm_power_3h_1991_2010.csv.gz> <katalog_wyjściowy>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, BODY, MUTED, GRID = "#1A1A1A", "#333333", "#777777", "#E2E2E2"
BLUE, LIGHT, ORANGE = "#2B5C8A", "#AFC4DA", "#E07B24"
CAP = 1605.0
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": BODY,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                     "font.size": 12})


def style(ax, title):
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID); ax.set_axisbelow(True)


def main():
    df = pd.read_csv(sys.argv[1], parse_dates=["time"])
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    df["year"], df["month"], df["hour"] = df.time.dt.year, df.time.dt.month, df.time.dt.hour
    df["e_gwh"] = df.farm_power_mw * 3.0 / 1e3

    # 1. miesiące
    m = df.groupby(["year", "month"]).e_gwh.sum().unstack()
    mean, lo, hi = m.mean(), m.min(), m.max()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    xs = np.arange(1, 13)
    ax.bar(xs, mean, width=0.62, color=BLUE, label="20-year average")
    ax.vlines(xs, lo, hi, color=INK, lw=1.4, label="weakest – strongest year")
    ax.scatter(xs, lo, marker="_", s=120, color=INK); ax.scatter(xs, hi, marker="_", s=120, color=INK)
    ax.set_xticks(xs); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("Energy per month [GWh]")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(0, hi.max() * 1.15)
    ratio = mean[[12, 1, 2]].sum() / mean[[6, 7, 8]].sum()
    style(ax, f"Winter gives about {100 * (ratio - 1):.0f}% more energy than summer")
    fig.tight_layout(); fig.savefig(out / "t5_monthly_energy.png", facecolor="white"); plt.close(fig)

    # 2. lata
    y = df.groupby("year").e_gwh.sum()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cols = [ORANGE if v in (y.min(), y.max()) else BLUE for v in y]
    ax.bar(y.index, y.values, width=0.7, color=cols)
    ax.axhline(y.mean(), color=INK, lw=1.2, ls="--")
    for yr in (y.idxmin(), y.idxmax()):
        ax.text(yr, y[yr] + 60, f"{y[yr]:.0f}", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel("Energy per year [GWh]")
    ax.set_ylim(0, y.max() * 1.12)
    ax.set_xticks(y.index[::2])
    style(ax, f"From {y.min():.0f} to {y.max():.0f} GWh a year (dashed line: average {y.mean():.0f})")
    fig.tight_layout(); fig.savefig(out / "t5_annual_energy.png", facecolor="white"); plt.close(fig)

    # 3. krzywa czasu trwania
    p = np.sort(df.farm_power_mw.values)[::-1]
    t = np.arange(1, len(p) + 1) / len(p) * 100
    full = (df.farm_power_mw >= 0.98 * CAP).mean() * 100
    zero = (df.farm_power_mw <= 0.01 * CAP).mean() * 100
    half = (df.farm_power_mw >= 0.5 * CAP).mean() * 100
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.fill_between(t, p, color=LIGHT, lw=0)
    ax.plot(t, p, color=BLUE, lw=2)
    ax.axhline(CAP, color=MUTED, lw=1, ls="--")
    ax.text(99, CAP + 25, "installed 1 605 MW", ha="right", fontsize=11, color=MUTED)
    # punkty na krzywej + napisy w pustym polu NAD krzywą (bez przecinania jej)
    pts = [(full, CAP, f"full power {full:.0f}% of the time", (full + 4, CAP * 0.93)),
           (half, CAP * 0.5, f"more than half power {half:.0f}% of the time", (half + 3, CAP * 0.62)),
           (100 - zero, 0.0, f"almost no power {zero:.0f}% of the time", (100 - zero - 20, CAP * 0.34))]
    for px_, py_, txt, (tx, ty) in pts:
        ax.scatter([px_], [py_], s=60, color=ORANGE, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(txt, (px_, py_), xytext=(tx, ty), fontsize=12, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1, shrinkA=2, shrinkB=5))
    ax.set_xlim(0, 100); ax.set_ylim(0, CAP * 1.1)
    ax.set_xlabel("Share of time [%]"); ax.set_ylabel("Farm power [MW]")
    style(ax, "How often the farm makes how much power")
    fig.tight_layout(); fig.savefig(out / "t5_power_duration.png", facecolor="white"); plt.close(fig)

    # 4. profil dzienny
    win = df[df.month.isin([12, 1, 2])].groupby("hour").farm_power_mw.mean()
    sum_ = df[df.month.isin([6, 7, 8])].groupby("hour").farm_power_mw.mean()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    ax.plot(win.index, win.values, color=BLUE, lw=2.5, marker="o", ms=7)
    ax.plot(sum_.index, sum_.values, color=ORANGE, lw=2.5, marker="o", ms=7)
    ax.text(21.3, win.values[-1], "winter", color=BLUE, va="center", fontsize=13, fontweight="bold")
    ax.text(21.3, sum_.values[-1], "summer", color=ORANGE, va="center", fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, CAP * 0.85)
    ax.set_xlabel("Time of day (UTC)"); ax.set_ylabel("Average farm power [MW]")
    style(ax, "Power over the day: winter vs summer")
    fig.tight_layout(); fig.savefig(out / "t5_daily_profile.png", facecolor="white"); plt.close(fig)

    w_s = mean[[12, 1, 2]].sum() / mean[[6, 7, 8]].sum()
    print(f"winter/summer energy ratio {w_s:.2f}; years {y.min():.0f}-{y.max():.0f} (mean {y.mean():.0f}); "
          f"full {full:.1f}% half {half:.1f}% zero {zero:.1f}%; "
          f"summer day range {sum_.min():.0f}-{sum_.max():.0f} MW, winter {win.min():.0f}-{win.max():.0f} MW")


if __name__ == "__main__":
    main()
