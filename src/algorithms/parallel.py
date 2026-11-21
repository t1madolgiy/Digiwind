"""
Równoległa ewaluacja AEP wielu layoutów (procesy, wszystkie rdzenie).
====================================================================
Każdy proces-pracownik buduje własny FarmModel raz (model wake, turbina,
róża wiatrów, wind shear) i liczy AEP dla przysłanych layoutów. Algorytmy
(GA, SLSQP z gradientem różnicowym) wysyłają całe paczki layoutów naraz.

UWAGA (Windows): używać tylko ze skryptu z `if __name__ == "__main__":`
(procesy są startowane metodą spawn). Nie w Jupyterze ani w Streamlit.

Użycie:
    >>> with ParallelEvaluator("gch", "iea_15MW", rose, wind_shear=0.059) as ev:
    ...     aeps = ev.map([(x1, y1), (x2, y2), ...])   # GWh
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_FARM = None


def _init_worker(cfg: dict) -> None:
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from floris import WindRose
    from src.farm_model import FarmModel

    global _FARM, _CHUNKS
    _FARM = FarmModel(cfg["wake"], cfg["turbine"])
    if cfg["wind_shear"] is not None:
        _FARM.set_wind_shear(cfg["wind_shear"])
    rose = WindRose(
        wind_directions=cfg["wd"], wind_speeds=cfg["ws"],
        freq_table=cfg["freq"], ti_table=cfg["ti"],
    )
    # Róża → lista warunków z niezerową częstością, pocięta na kawałki.
    # FLORIS liczy duże tablice (warunki × turbiny × turbiny) wolniej niż
    # proporcjonalnie (pamięć podręczna CPU) — kawałki po ~100 warunków są
    # kilka razy szybsze. AEP = Σ P(warunek) · częstość · 8760 h, jak w WindRose.
    WD, WS = np.meshgrid(rose.wind_directions, rose.wind_speeds, indexing="ij")
    TI = np.broadcast_to(np.asarray(rose.ti_table, float), WD.shape)
    F = np.asarray(rose.freq_table, float)
    m = F > 0
    wd, ws, ti, fr = WD[m], WS[m], TI[m], F[m]
    fr = fr / fr.sum()
    step = int(cfg.get("chunk") or len(wd))
    _CHUNKS = [(wd[i:i + step], ws[i:i + step], ti[i:i + step], fr[i:i + step])
               for i in range(0, len(wd), step)]


def _eval_one(xy) -> float:
    x, y = xy
    try:
        fm = _FARM.fmodel
        total = 0.0
        for wd, ws, ti, fr in _CHUNKS:
            fm.set(layout_x=np.asarray(x, float), layout_y=np.asarray(y, float),
                   wind_directions=wd, wind_speeds=ws, turbulence_intensities=ti)
            fm.run()
            total += float(np.nansum(fm.get_farm_power() * fr))
        a = total * 8760.0 / 1e9
        return a if np.isfinite(a) else 0.0
    except Exception:
        return 0.0


class ParallelEvaluator:
    """Pula procesów liczących AEP; licznik `n_evals` i czas `busy_s`."""

    def __init__(self, wake: str, turbine: str, wind_rose, wind_shear=None,
                 n_workers: int | None = None, chunk: int | None = 50):
        # jeden wątek BLAS na proces — inaczej procesy walczą o rdzenie
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = "1"
        ti = wind_rose.ti_table
        cfg = {
            "wake": wake, "turbine": turbine,
            "wind_shear": None if wind_shear is None else float(wind_shear),
            "wd": np.asarray(wind_rose.wind_directions, float),
            "ws": np.asarray(wind_rose.wind_speeds, float),
            "freq": np.asarray(wind_rose.freq_table, float),
            "ti": ti if np.isscalar(ti) else np.asarray(ti, float),
            "chunk": chunk,
        }
        self.n_workers = n_workers or max(1, (os.cpu_count() or 2) - 2)
        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers, initializer=_init_worker, initargs=(cfg,)
        )
        self.n_evals = 0
        self.busy_s = 0.0

    def map(self, layouts) -> np.ndarray:
        layouts = list(layouts)
        t = time.perf_counter()
        chunk = max(1, len(layouts) // (self.n_workers * 4))
        res = np.fromiter(self._pool.map(_eval_one, layouts, chunksize=chunk), float, len(layouts))
        self.n_evals += len(layouts)
        self.busy_s += time.perf_counter() - t
        return res

    def one(self, x, y) -> float:
        return float(self.map([(x, y)])[0])

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
