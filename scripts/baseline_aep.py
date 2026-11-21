"""AEP layoutu bazowego Baltica 2 (107 × IEA 15 MW) na wietrze IMGW 1991–2010, 150 m.

Liczy:
  1. wpływ wyboru punktów IMGW (3 zestawy),
  2. trzy modele wake (GCH, TurbOPark, Empirical Gauss),
  3. wrażliwość na TI 0.04–0.10,
  4. wpływ wykładnika α (z danych vs domyślne 0.12 w FLORIS).
Róża 10° × 2 m/s (od 0 m/s). Wynik: outputs/exports/baseline_aep_imgw.csv

Uruchomienie (z katalogu repo):  python scripts/baseline_aep.py
"""
import sys, time, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from src.wind_data import WindDataLoader
from src.farm_model import FarmModel
from src.site_baltica2 import baltica2_site, rotated_grid_layout

POINT_SETS = {
    "dookoła farmy (P47,P62,P63,P35)": ("P47", "P62", "P63", "P35"),
    "południe/zachód (P47,P33,P20,P35)": ("P47", "P33", "P20", "P35"),
    "6 najbliższych": ("P35", "P62", "P47", "P63", "P33", "P61"),
}
DEFAULT_SET = "dookoła farmy (P47,P62,P63,P35)"


def run(loader, wake="gch", ti=0.06, shear="data", x=None, y=None):
    if ti != 0.06:
        loader = WindDataLoader(loader.wind_speeds, loader.wind_directions,
                                np.full(len(loader.wind_speeds), ti), loader.timestamps)
        loader.wind_shear = None
    rose = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)
    f = FarmModel(wake, "iea_15MW")
    f.set_layout_custom(x, y, name="baltica2_base")
    if shear == "data":
        f.set_wind_shear(ALPHA)
    elif shear is not None:
        f.set_wind_shear(shear)
    f.set_wind_data(rose)
    t = time.perf_counter()
    f.run()
    aep = f.get_aep_gwh()
    cap = f.turbine_info["rated_power"] * f.n_turbines
    return {
        "aep_gwh": aep,
        "capacity_factor_pct": aep / (cap * 8.76) * 100,
        "wake_losses_pct": f.get_wake_losses_percent(),
        "runtime_s": time.perf_counter() - t,
    }


def main():
    global ALPHA
    site = baltica2_site()
    x, y, info = rotated_grid_layout(site, n_turbines=107, D=240.0)
    rows = []

    print("== 1. Wybór punktów IMGW (GCH, TI 0.06, α z danych)")
    loaders = {}
    for name, pts in POINT_SETS.items():
        L = WindDataLoader.from_imgw(points=pts)
        loaders[name] = L
        ALPHA = L.wind_shear
        r = run(L, x=x, y=y)
        s = L.summary()
        print(f"  {name:36s} ws={s['wind_speed']['mean']:.2f} m/s  α={L.wind_shear:.3f}  "
              f"AEP={r['aep_gwh']:.0f} GWh  CF={r['capacity_factor_pct']:.1f}%  wake={r['wake_losses_pct']:.1f}%",
              flush=True)
        rows.append({"study": "points", "case": name, "wake": "gch", "ti": 0.06, "alpha": L.wind_shear,
                     "mean_ws": s["wind_speed"]["mean"], **r})

    L = loaders[DEFAULT_SET]
    ALPHA = L.wind_shear
    print(f"\nDomyślny zestaw: {DEFAULT_SET}; wagi {L.source_info['weights']}")

    print("\n== 2. Modele wake (TI 0.06)")
    for wake in ("gch", "turbopark", "empirical_gauss"):
        r = run(L, wake=wake, x=x, y=y)
        print(f"  {wake:16s} AEP={r['aep_gwh']:.0f} GWh  CF={r['capacity_factor_pct']:.1f}%  "
              f"wake={r['wake_losses_pct']:.1f}%  ({r['runtime_s']:.0f}s)", flush=True)
        rows.append({"study": "wake_model", "case": wake, "wake": wake, "ti": 0.06, "alpha": ALPHA, **r})

    print("\n== 3. Wrażliwość na TI (GCH)")
    for ti in (0.04, 0.06, 0.08, 0.10):
        r = run(L, ti=ti, x=x, y=y)
        print(f"  TI={ti:.2f}  AEP={r['aep_gwh']:.0f} GWh  wake={r['wake_losses_pct']:.1f}%", flush=True)
        rows.append({"study": "ti", "case": f"TI={ti}", "wake": "gch", "ti": ti, "alpha": ALPHA, **r})

    print("\n== 4. Wykładnik α (GCH)")
    for label, sh in ((f"z danych ({ALPHA:.3f})", "data"), ("domyślne FLORIS 0.12", 0.12)):
        r = run(L, shear=sh, x=x, y=y)
        print(f"  {label:22s} AEP={r['aep_gwh']:.0f} GWh", flush=True)
        rows.append({"study": "alpha", "case": label, "wake": "gch", "ti": 0.06,
                     "alpha": ALPHA if sh == "data" else sh, **r})

    out = ROOT / "outputs" / "exports" / "baseline_aep_imgw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(3).to_csv(out, index=False)
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()
