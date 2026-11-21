"""
SLSQP z równoległym gradientem i dokładnymi ograniczeniami.
===========================================================
Własna wersja zamiast FLORIS LayoutOptimizationScipy, bo dla 107 turbin:
  - gradient różnicowy = 214 dodatkowych ewaluacji AEP na iterację —
    tu liczone naraz (ParallelEvaluator), nie po kolei,
  - ograniczenia podane z dokładnymi pochodnymi:
      odstęp:  (d_ij² − d_min²) / d_min² ≥ 0   dla wszystkich par,
      granica: odległość ze znakiem od brzegu wielokąta ≥ 0 (Region),
  - twardy budżet ewaluacji (jak w innych algorytmach),
  - wynik = najlepszy layout SPEŁNIAJĄCY ograniczenia spośród policzonych.

Params:
    evaluator       ParallelEvaluator (zalecany dla dużych farm)
    fd_step_m       krok różnicy skończonej [m] (domyślnie 5)
    maxiter         maks. iteracji SLSQP (domyślnie 200; zwykle kończy budżet)
    start_jitter_D  losowe przesunięcie startu × D (dla porównań na seedach; 0 = start dokładnie z layoutu)
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize
from shapely import contains_xy, points, line_locate_point, line_interpolate_point, get_coordinates
from shapely.geometry import box

from .base import (
    LayoutAlgorithm, AlgorithmResult, evaluate_aep, is_feasible, repair_min_dist, Region,
)

_S = 1000.0  # skala zmiennych: km


class _BudgetExhausted(Exception):
    pass


class SLSQPParallel(LayoutAlgorithm):
    name = "SLSQP (parallel gradient)"
    description = "SLSQP z gradientem różnicowym liczonym równolegle i dokładnymi ograniczeniami."
    PARAMS = [
        {"key": "maxiter", "label": "Max iteracji", "type": "int", "min": 5, "max": 500, "default": 200, "step": 5},
        {"key": "fd_step_m", "label": "Krok gradientu [m]", "type": "float", "min": 0.5, "max": 50.0, "default": 5.0, "step": 0.5},
    ]

    def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params) -> AlgorithmResult:
        evaluator = params.get("evaluator")
        h = float(params.get("fd_step_m", 5.0)) / _S
        maxiter = int(params.get("maxiter", 200))
        jitter_D = float(params.get("start_jitter_D", 0.0))

        region = bounds_rect if isinstance(bounds_rect, Region) else Region(box(*bounds_rect))
        ring = region._inner.exterior if not region._inner.is_empty else region.polygon.exterior
        n = farm.n_turbines
        D = farm.D
        md2 = float(min_dist) ** 2
        iu, ju = np.triu_indices(n, 1)

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        x0, y0 = init_x.copy(), init_y.copy()
        if jitter_D > 0:
            rng = np.random.default_rng(seed)
            x0 = x0 + rng.normal(0, jitter_D * D, n)
            y0 = y0 + rng.normal(0, jitter_D * D, n)
            x0, y0 = repair_min_dist(x0, y0, min_dist, bounds_rect)

        t0 = time.perf_counter()
        state = {"n": 0, "best": -np.inf, "bx": x0.copy(), "by": y0.copy(), "hist": [], "cache": {}}

        def many(layouts):
            if state["n"] + len(layouts) > eval_budget:
                raise _BudgetExhausted
            if evaluator is not None:
                a = evaluator.map(layouts)
            else:
                a = np.array([evaluate_aep(farm, xx, yy) for xx, yy in layouts])
            state["n"] += len(layouts)
            for (xx, yy), v in zip(layouts, a):
                if v > state["best"] and is_feasible(xx, yy, bounds_rect, min_dist):
                    state["best"], state["bx"], state["by"] = float(v), xx.copy(), yy.copy()
            state["hist"].append(state["best"])
            return a

        def split(z):
            return z[:n] * _S, z[n:] * _S

        def aep_at(z):
            key = z.tobytes()
            if key not in state["cache"]:
                state["cache"][key] = float(many([split(z)])[0])
            return state["cache"][key]

        z0 = np.concatenate([x0, y0]) / _S
        a0 = aep_at(z0)
        init_aep = a0 if jitter_D == 0 else None
        scale = a0 if a0 > 0 else 1.0

        def obj(z):
            return -aep_at(z) / scale

        def grad(z):
            base = aep_at(z)
            pert = []
            for k in range(2 * n):
                zk = z.copy(); zk[k] += h
                pert.append(split(zk))
            vals = many(pert)
            return -(vals - base) / h / scale

        def spacing(z):
            x, y = split(z)
            return ((x[iu] - x[ju]) ** 2 + (y[iu] - y[ju]) ** 2 - md2) / md2

        def spacing_jac(z):
            x, y = split(z)
            dx, dy = x[iu] - x[ju], y[iu] - y[ju]
            J = np.zeros((len(iu), 2 * n))
            r = np.arange(len(iu))
            c = 2.0 * _S / md2
            J[r, iu] = c * dx; J[r, ju] = -c * dx
            J[r, n + iu] = c * dy; J[r, n + ju] = -c * dy
            return J

        def _nearest(z):
            x, y = split(z)
            p = points(x, y)
            q = get_coordinates(line_interpolate_point(ring, line_locate_point(ring, p)))
            v = np.column_stack([x, y]) - q
            d = np.hypot(v[:, 0], v[:, 1])
            sign = np.where(contains_xy(region._inner if not region._inner.is_empty else region.polygon, x, y), 1.0, -1.0)
            return v, d, sign

        def boundary(z):
            _, d, sign = _nearest(z)
            return sign * d / _S

        def boundary_jac(z):
            v, d, sign = _nearest(z)
            u = v / np.maximum(d, 1e-9)[:, None] * sign[:, None]
            J = np.zeros((n, 2 * n))
            J[np.arange(n), np.arange(n)] = u[:, 0]
            J[np.arange(n), n + np.arange(n)] = u[:, 1]
            return J

        cons = [
            {"type": "ineq", "fun": spacing, "jac": spacing_jac},
            {"type": "ineq", "fun": boundary, "jac": boundary_jac},
        ]
        message, nit = "budget exhausted", -1
        try:
            res = minimize(obj, z0, jac=grad, method="SLSQP", constraints=cons,
                           options={"maxiter": maxiter, "ftol": 1e-9})
            message, nit = str(res.message), int(res.nit)
        except _BudgetExhausted:
            pass

        best_x, best_y = state["bx"], state["by"]
        best_aep = state["best"] if np.isfinite(state["best"]) else a0
        farm.set_layout_custom(best_x, best_y, name="_slsqp_best")
        return AlgorithmResult(
            name=self.name,
            initial_x=x0, initial_y=y0,
            initial_aep=a0 if init_aep is None else init_aep,
            final_x=best_x, final_y=best_y, final_aep=best_aep,
            elapsed_s=time.perf_counter() - t0,
            n_evaluations=state["n"],
            history=state["hist"],
            extra={"nit": nit, "message": message, "fd_step_m": h * _S, "start_jitter_D": jitter_D},
        )
