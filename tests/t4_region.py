"""Ograniczenie wielokątem (Region): 7 algorytmów × 107 turbin w granicy Baltica 2.
Każdy wynik musi leżeć w wielokącie i spełniać min. odstęp.
Uruchomienie: python tests/t4_region.py <ścieżka_do_repo>   (kilka minut)
"""
import sys, logging, warnings, time
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
logging.disable(logging.WARNING)

from floris import TimeSeries
from src.farm_model import FarmModel
from src.algorithms import ALGORITHMS, Region, is_feasible
from src.algorithms.base import random_feasible_layout
from src.site_baltica2 import baltica2_site, rotated_grid_layout


def main():
    D = 240.0
    site = baltica2_site()
    region = Region(site.polygon, margin=120.0)
    x0, y0, _ = rotated_grid_layout(site, n_turbines=107, D=D)
    md = 4.0 * D
    print("baseline feasible:", is_feasible(x0, y0, region, md))

    rx, ry = random_feasible_layout(107, region, md, seed=1)
    print("random layout feasible:", is_feasible(rx, ry, region, md),
          " inside:", region.contains(rx, ry).all())

    # punkt w wcięciu granicy → rzutowanie musi przenieść go do środka
    px, py = region.project(np.array([9800.0]), np.array([4000.0]))
    print("projected notch point inside:", region.contains(px, py).all())

    # jeden warunek wiatrowy — test sprawdza ograniczenia, nie jakość optimum
    rose = TimeSeries(wind_directions=np.array([260.0]), wind_speeds=np.array([9.0]),
                      turbulence_intensities=np.array([0.06]))
    ok_all = True
    keys = [k for k in ALGORITHMS if k != "scipy"] + ["scipy"]  # SLSQP najwolniejszy — na końcu
    for key in keys:
        cls = ALGORITHMS[key]
        f = FarmModel("gch", "iea_15MW"); f.set_layout_custom(x0, y0, name="b2"); f.set_wind_data(rose)
        params = {"maxiter": 2} if key == "scipy" else {}
        t = time.perf_counter()
        r = cls().run(f, region, md, 40, seed=0, **params)
        inside = region.contains(r.final_x, r.final_y).all()
        ok_all &= bool(r.feasible) and bool(inside) and r.error is None
        print(f"{key:24s} {r.initial_aep:8.1f} -> {r.final_aep:8.1f} GWh ({r.improvement_pct:+.2f}%)  "
              f"minDist={r.min_spacing_m / D:.2f}D inside={inside} feasible={r.feasible} "
              f"t={time.perf_counter() - t:.0f}s err={r.error}", flush=True)
    print("ALL OK" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
