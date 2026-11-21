"""Szeregi czasowe dla Task 5 i Task 6 (layout bazowy Baltica 2, wiatr IMGW 1991–2010, 3 h).

Liczy każdy z 58 400 kroków IMGW przez FLORIS (GCH, IEA 15 MW, α z IMGW, TI 0,06):
prędkość wiatru na każdej turbinie (po śladach innych turbin), moc każdej turbiny
oraz moc „bez śladów” (ta sama farma, gdyby turbiny się nie zasłaniały).

Wyniki (<out>/Task5, <out>/Task6):
  Task5/aep_timeseries.csv          — rok, miesiące, sezony: energia [GWh/rok], CF, straty wake
  Task5/aep_by_year.csv             — energia każdego roku 1991–2010
  Task5/farm_power_3h_1991_2010.csv.gz — moc farmy co 3 h [MW] + wiatr swobodny
  Task6/wind_speed_per_turbine_3h_1991_2010.csv.gz — v [m/s] dla T001…T107 co 3 h
  Task6/wind_speed_per_turbine_3h_2010.csv           — to samo, tylko rok 2010 (lżejszy plik)

Uruchomienie:  python scripts/export_timeseries_t5_t6.py <katalog_docelowy>
"""
import os, sys, time, logging, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

WAKE, TURBINE, TI, CHUNK = "gch", "iea_15MW", 0.06, 50
_W = {}


def _init(x, y, shear):
    import logging, warnings
    warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    from src.farm_model import FarmModel
    f = FarmModel(WAKE, TURBINE); f.set_layout_custom(x, y); f.set_wind_shear(shear)
    one = FarmModel(WAKE, TURBINE); one.set_layout_custom(np.array([0.0]), np.array([0.0])); one.set_wind_shear(shear)
    _W["farm"], _W["one"] = f.fmodel, one.fmodel


def _run(args):
    wd, ws = args
    ti = np.full(len(wd), TI)
    fm = _W["farm"]
    fm.set(wind_directions=wd, wind_speeds=ws, turbulence_intensities=ti)
    fm.run()
    v = fm.turbine_average_velocities.astype(np.float32)
    p = (fm.get_turbine_powers() / 1e6).astype(np.float32)      # MW
    o = _W["one"]
    o.set(wind_directions=wd, wind_speeds=ws, turbulence_intensities=ti)
    o.run()
    p_free = (o.get_turbine_powers()[:, 0] / 1e6).astype(np.float32)  # MW, jedna turbina bez śladu
    return v, p, p_free


def main():
    from src.wind_data import WindDataLoader
    from src.site_baltica2 import baltica2_site, rotated_grid_layout

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "exports" / "Task2_pliki"
    (out / "Task5").mkdir(parents=True, exist_ok=True); (out / "Task6").mkdir(parents=True, exist_ok=True)

    site = baltica2_site()
    x, y, _ = rotated_grid_layout(site, n_turbines=107, D=240.0)
    L = WindDataLoader.from_imgw()
    wd, ws, ts = L.wind_directions, L.wind_speeds, L.timestamps
    n = len(ws)
    chunks = [(wd[i:i + CHUNK], ws[i:i + CHUNK]) for i in range(0, n, CHUNK)]

    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=16, initializer=_init, initargs=(x, y, L.wind_shear)) as ex:
        res = list(ex.map(_run, chunks, chunksize=4))
    V = np.vstack([r[0] for r in res]); P = np.vstack([r[1] for r in res]); Pf = np.concatenate([r[2] for r in res])
    print(f"FLORIS: {n} steps x {V.shape[1]} turbines in {time.perf_counter() - t0:.0f} s", flush=True)

    farm = P.sum(axis=1).astype(float)          # MW
    gross = Pf.astype(float) * len(x)           # MW, bez śladów
    cap = 15.0 * len(x)                         # MW
    years = ts.year.nunique()
    df = pd.DataFrame({"time": ts, "farm_power_mw": farm, "farm_power_no_wake_mw": gross})

    # --- Task 5: rok / miesiące / sezony (średni rok z 20 lat) ---
    def period_row(period, label, mask):
        e = farm[mask].sum() * 3.0 / 1e3 / years      # GWh na „średni rok”
        eg = gross[mask].sum() * 3.0 / 1e3 / years
        hours = mask.sum() * 3.0 / years
        return {"period": period, "period_label": label, "aep_gwh": round(e, 2),
                "capacity_factor_pct": round(100 * e / (cap * hours / 1e3), 2),
                "wake_losses_pct": round(100 * (1 - e / eg), 2),
                "n_turbines": len(x), "turbine_type": "IEA-15-240-RWT", "wake_model": WAKE.upper(),
                "total_capacity_mw": cap}

    allm = np.ones(n, bool)
    rows = [period_row("annual", "Year", allm)]
    names = ["January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December"]
    for m in range(1, 13):
        rows.append(period_row(f"month_{m:02d}", names[m - 1], np.asarray(ts.month == m)))
    for code, label, months in (("DJF", "Winter (Dec-Feb)", (12, 1, 2)), ("MAM", "Spring (Mar-May)", (3, 4, 5)),
                                ("JJA", "Summer (Jun-Aug)", (6, 7, 8)), ("SON", "Autumn (Sep-Nov)", (9, 10, 11))):
        rows.append(period_row(f"season_{code}", label, np.asarray(ts.month.isin(months))))
    t5 = pd.DataFrame(rows)
    t5.to_csv(out / "Task5" / "aep_timeseries.csv", index=False)

    by_year = df.groupby(ts.year).agg(e=("farm_power_mw", "sum"), g=("farm_power_no_wake_mw", "sum"))
    by_year = pd.DataFrame({"year": by_year.index, "aep_gwh": (by_year.e * 3 / 1e3).round(1),
                            "wake_losses_pct": (100 * (1 - by_year.e / by_year.g)).round(2)})
    by_year.to_csv(out / "Task5" / "aep_by_year.csv", index=False)

    p3 = pd.DataFrame({"time": ts.strftime("%Y-%m-%d %H:%M"), "wind_direction_deg": np.round(wd, 1),
                       "wind_speed_free_ms": np.round(ws, 2), "farm_power_mw": np.round(farm, 1)})
    p3.to_csv(out / "Task5" / "farm_power_3h_1991_2010.csv.gz", index=False, compression="gzip")

    # --- Task 6: prędkość na każdej turbinie, szeroki format ---
    cols = {f"T{i + 1:03d}": np.round(V[:, i], 2) for i in range(V.shape[1])}
    t6 = pd.DataFrame({"time": ts.strftime("%Y-%m-%d %H:%M"), "wind_direction_deg": np.round(wd, 1),
                       "wind_speed_free_ms": np.round(ws, 2), **cols})
    t6.to_csv(out / "Task6" / "wind_speed_per_turbine_3h_1991_2010.csv.gz", index=False, compression="gzip")
    t6[np.asarray(ts.year == 2010)].to_csv(out / "Task6" / "wind_speed_per_turbine_3h_2010.csv", index=False)

    a = t5.iloc[0]
    print(f"Annual (time series): {a.aep_gwh:.0f} GWh, CF {a.capacity_factor_pct:.1f}%, wake {a.wake_losses_pct:.1f}%")
    print(f"Sum of months: {t5[t5.period.str.startswith('month')].aep_gwh.sum():.0f} GWh, "
          f"sum of seasons: {t5[t5.period.str.startswith('season')].aep_gwh.sum():.0f} GWh")
    print(f"Years: min {by_year.aep_gwh.min():.0f} ({by_year.year[by_year.aep_gwh.idxmin()]}), "
          f"max {by_year.aep_gwh.max():.0f} ({by_year.year[by_year.aep_gwh.idxmax()]})")
    print(f"Mean wind at turbines {V.mean():.2f} m/s vs free {ws.mean():.2f} m/s")
    print("written to", out)


if __name__ == "__main__":
    main()
