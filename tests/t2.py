"""GA vs SLSQP: wąski bin vs pełna róża, ograniczenia wyniku (#5),
różnorodność losowych layoutów dla 107 turbin (#6).
Uruchomienie: python tests/t2.py <ścieżka_do_repo>   (ok. 5 min)
"""
import sys, logging, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
logging.disable(logging.WARNING)

from scipy.spatial.distance import pdist
from floris import TimeSeries
from src.farm_model import FarmModel
from src.wind_data import WindDataLoader
from src.algorithms import ALGORITHMS, bounds_from_layout
from src.algorithms.base import random_feasible_layout, is_feasible


def main():
    D = 240.0

    def mk(sp, wind):
        f = FarmModel("gch", "iea_15MW"); f.set_layout_grid(3, 3, sp); f.set_wind_data(wind); return f

    narrow = TimeSeries(wind_directions=np.array([270.]), wind_speeds=np.array([9.]),
                        turbulence_intensities=np.array([.06]))
    L = WindDataLoader(); L.generate_mock_data(); rose = L.to_wind_rose()

    for sp in (7.0, 4.0):
        for key in ("scipy", "genetic"):
            f = mk(sp, narrow); b = bounds_from_layout(f.layout_x, f.layout_y, 3 * D); md = 3 * D
            r = ALGORITHMS[key]().run(f, b, md, 100, seed=42)
            g = mk(sp, rose); g.run(); a0 = g.get_aep_gwh()
            g.set_layout_custom(r.final_x, r.final_y); g.run(); a1 = g.get_aep_gwh()
            print(f"start {sp}D {key:8s} narrow: {r.initial_aep:.1f}->{r.final_aep:.1f} "
                  f"({r.improvement_pct:+.1f}%) t={r.elapsed_s:.1f}s evals={r.n_evaluations} | "
                  f"minDist={r.min_spacing_m / D:.2f}D feasible={r.feasible} | "
                  f"FULL ROSE: {a0:.1f}->{a1:.1f} ({100 * (a1 / a0 - 1):+.2f}%)")

    b = (0, 0, 25000, 8000); md = 3 * D
    lays = [random_feasible_layout(107, b, md, seed=s) for s in range(5)]
    print("107T: identical across seeds:", all(np.allclose(lays[0][0], l[0]) for l in lays[1:]),
          " feasible:", [is_feasible(x, y, b, md) for x, y in lays],
          " minDist[D]:", [round(pdist(np.column_stack([x, y])).min() / D, 2) for x, y in lays])


if __name__ == "__main__":
    main()
