"""
Scipy gradient-based — wrapper na FLORIS LayoutOptimizationScipy.
=================================================================
Szybki, ale lokalny optimizer. Dobry baseline dla porównań.
"""
from __future__ import annotations

import time
import numpy as np

from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep


class ScipyGradient(LayoutAlgorithm):
    name = "Scipy (gradient)"
    description = "FLORIS LayoutOptimizationScipy — szybki, lokalny."
    PARAMS = [
        {"key": "maxiter", "label": "Max iteracji", "type": "int", "min": 10, "max": 200, "default": 50, "step": 10},
    ]

    def run(
        self,
        farm,
        bounds_rect,
        min_dist,
        eval_budget,
        seed=42,
        **params,
    ) -> AlgorithmResult:
        from floris.optimization.layout_optimization.layout_optimization_scipy import (
            LayoutOptimizationScipy,
        )

        maxiter = int(params.get("maxiter", min(eval_budget, 50)))

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        x_min, y_min, x_max, y_max = bounds_rect
        boundaries = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

        t0 = time.perf_counter()
        try:
            opt = LayoutOptimizationScipy(
                farm.fmodel,
                boundaries,
                min_dist=min_dist,
                optOptions={"maxiter": maxiter, "disp": False},
            )
            sol = opt.optimize()
            opt_x = np.array(sol[0])
            opt_y = np.array(sol[1])
            final_aep = evaluate_aep(farm, opt_x, opt_y)
            elapsed = time.perf_counter() - t0

            return AlgorithmResult(
                name=self.name,
                initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
                final_x=opt_x, final_y=opt_y, final_aep=final_aep,
                elapsed_s=elapsed,
                n_evaluations=maxiter,
                history=[init_aep, final_aep],
                extra={"maxiter": maxiter},
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return AlgorithmResult(
                name=self.name,
                initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
                final_x=init_x, final_y=init_y, final_aep=init_aep,
                elapsed_s=elapsed,
                n_evaluations=0,
                error=str(e),
            )
