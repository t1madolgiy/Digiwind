"""Benchmark M2: GA vs SLSQP vs hybryda GA → SLSQP, Baltica 2, 107 × IEA 15 MW.

Ustalenia z kick-offu: pełna róża wiatrów, 5 seedów na metodę, ten sam budżet
ewaluacji AEP. Róże ZMIENIONE względem kick-offu (30°/10°) po teście 21.11:
layout z GA dawał +1,53% na róży 30° × 3 m/s, +0,07% na 10° × 4 m/s i −0,24%
na 5° × 4 m/s — grube kierunki pozwalają „ustawić turbiny między sektorami”.
Dlatego: szukanie na 5° × 4 m/s, ocena końcowa na 2° × 1 m/s (parametry --search-rose, --eval-rose).
Wiatr: IMGW 150 m 1991–2010 (P47, P62, P63, P35). Model wake: GCH.
Ograniczenia: granica Baltica 2 (raport EIA) z marginesem 120 m, odstęp ≥ min_dist_D × D.
Start: layout bazowy (siatka 4,7 D obrócona do 260°).
Seedy: GA i hybryda — losowa populacja (perturbacje bazy); SLSQP — seed 0 startuje
dokładnie z bazy, seedy 1–4 z bazy przesuniętej losowo o 0,25 D.

Wyniki dopisywane po każdym przebiegu:
  <out>/results.csv        — tabela (metoda, seed, czas, ewaluacje, AEP, zysk, odstęp)
  <out>/layouts/<metoda>_s<seed>.csv — końcowe pozycje turbin
  <out>/log.txt

Uruchomienie (z katalogu repo):
  python scripts/benchmark_m2.py --budget 2000 --seeds 5          (pełny, kilka godzin)
  python scripts/benchmark_m2.py --budget 300 --seeds 1 --out outputs/m2_test   (test)
"""
import sys, time, argparse, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from scipy.spatial.distance import pdist
from src.wind_data import WindDataLoader
from src.farm_model import FarmModel
from src.site_baltica2 import baltica2_site, rotated_grid_layout, layout_table
from src.algorithms import Region, is_feasible
from src.algorithms.genetic import GeneticAlgorithm
from src.algorithms.slsqp_parallel import SLSQPParallel
from src.algorithms.hybrid import HybridGASLSQP
from src.algorithms.parallel import ParallelEvaluator

D = 240.0
GA_PARAMS = dict(pop_size=24, mutation_rate=0.1, mutation_sigma_D=0.5, n_elite=2,
                 init_mode="perturb", init_sigma_D=0.5)


def log(out, msg):
    print(msg, flush=True)
    with open(out / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--methods", default="ga,slsqp,hybrid")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-dist-D", type=float, default=4.0)
    ap.add_argument("--wake", default="gch")
    ap.add_argument("--search-rose", default="5,4", help="wd_step,ws_step róży do szukania")
    ap.add_argument("--eval-rose", default="2,1", help="wd_step,ws_step róży do oceny końcowej")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m2_benchmark"))
    a = ap.parse_args()
    sw, ss = (float(v) for v in a.search_rose.split(","))
    ew, es = (float(v) for v in a.eval_rose.split(","))

    out = Path(a.out); (out / "layouts").mkdir(parents=True, exist_ok=True)
    site = baltica2_site()
    region = Region(site.polygon, margin=120.0)
    x0, y0, _ = rotated_grid_layout(site, n_turbines=107, D=D)
    md = a.min_dist_D * D
    L = WindDataLoader.from_imgw()
    rose_search = L.to_wind_rose(wd_step=sw, ws_step=ss)
    rose_eval = L.to_wind_rose(wd_step=ew, ws_step=es)
    log(out, f"=== M2 benchmark {time.strftime('%Y-%m-%d %H:%M')} | budget {a.budget} | seeds {a.seeds} | "
             f"min dist {a.min_dist_D} D | wake {a.wake} | workers {a.workers} | "
             f"search rose {sw:g}x{ss:g} | eval rose {ew:g}x{es:g}")
    log(out, f"baseline feasible: {is_feasible(x0, y0, region, md)}")

    methods = {"ga": GeneticAlgorithm, "slsqp": SLSQPParallel, "hybrid": HybridGASLSQP}
    runs = []
    with ParallelEvaluator(a.wake, "iea_15MW", rose_search, L.wind_shear, n_workers=a.workers) as ev:
        base_search = ev.one(x0, y0)
        log(out, f"baseline AEP (search rose 30x3): {base_search:.1f} GWh")
        for m in a.methods.split(","):
            for s in range(a.seeds):
                farm = FarmModel(a.wake, "iea_15MW")
                farm.set_layout_custom(x0, y0, name="baseline")
                params = {"evaluator": ev}
                if m in ("ga", "hybrid"):
                    params.update(GA_PARAMS)
                if m == "slsqp":
                    params["start_jitter_D"] = 0.0 if s == 0 else 0.25
                t = time.perf_counter()
                r = methods[m]().run(farm, region, md, a.budget, seed=s, **params)
                dt = time.perf_counter() - t
                row = dict(method=m, seed=s, budget=a.budget, n_evals=r.n_evaluations, time_s=round(dt, 1),
                           search_base=base_search, search_final=r.final_aep,
                           search_gain_pct=100 * (r.final_aep / base_search - 1),
                           min_spacing_D=pdist(np.column_stack([r.final_x, r.final_y])).min() / D,
                           feasible=bool(r.feasible), note=str(r.extra.get("message", r.extra.get("slsqp_message", ""))))
                runs.append((row, r.final_x, r.final_y))
                layout_table(site, r.final_x, r.final_y, "IEA-15-240-RWT").to_csv(
                    out / "layouts" / f"{m}_s{s}.csv", index=False)
                pd.DataFrame([x[0] for x in runs]).to_csv(out / "results.csv", index=False)
                log(out, f"{m:7s} seed {s}: {base_search:.1f} -> {r.final_aep:.1f} GWh "
                         f"({row['search_gain_pct']:+.2f}%) | {r.n_evaluations} evals | {dt / 60:.1f} min | "
                         f"min {row['min_spacing_D']:.2f} D | feasible {r.feasible}")

    # Ocena końcowa wszystkich wyników na dokładnej róży 10° × 2 m/s
    log(out, "evaluating on fine rose 10x2 ...")
    with ParallelEvaluator(a.wake, "iea_15MW", rose_eval, L.wind_shear, n_workers=a.workers) as ev2:
        vals = ev2.map([(x0, y0)] + [(fx, fy) for _, fx, fy in runs])
    base_fine = vals[0]
    for (row, _, _), v in zip(runs, vals[1:]):
        row["fine_base"] = base_fine
        row["fine_final"] = v
        row["fine_gain_pct"] = 100 * (v / base_fine - 1)
    df = pd.DataFrame([x[0] for x in runs])
    df.round(4).to_csv(out / "results.csv", index=False)
    log(out, f"baseline AEP (fine rose 10x2): {base_fine:.1f} GWh")
    summ = df.groupby("method").agg(
        fine_gain_mean=("fine_gain_pct", "mean"), fine_gain_std=("fine_gain_pct", "std"),
        fine_gain_best=("fine_gain_pct", "max"), time_min=("time_s", lambda s: s.mean() / 60),
        all_feasible=("feasible", "all"))
    log(out, "\n" + summ.round(3).to_string())
    log(out, "done")


if __name__ == "__main__":
    main()
