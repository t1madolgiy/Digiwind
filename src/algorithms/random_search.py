"""
Random Search — własna implementacja losowych prób w boundaries.
=================================================================
W każdej iteracji losuje nowy layout w boundaries (z spełnieniem min_dist).
Akceptuje gdy AEP wyższe. Świetny baseline / sanity-check.
"""
from __future__ import annotations

import time
import numpy as np

from .base import (
    LayoutAlgorithm, AlgorithmResult, evaluate_aep,
    random_feasible_layout, repair_min_dist,
)


class RandomSearch(LayoutAlgorithm):
    name = "Random Search"
    description = "Losowe próby w boundaries, akceptacja jeśli lepiej."

    def run(
        self,
        farm,
        bounds_rect,
        min_dist,
        eval_budget,
        seed=42,
        **params,
    ) -> AlgorithmResult:
        n = farm.n_turbines
        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        best_x, best_y = init_x.copy(), init_y.copy()
        best_aep = init_aep
        history = [best_aep]
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        n_evals = 1  # initial eval already counted

        for i in range(eval_budget - 1):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            x, y = random_feasible_layout(n, bounds_rect, min_dist, seed=sub_seed, max_attempts=200)
            if x is None:
                continue
            # repair na wszelki wypadek
            x, y = repair_min_dist(x, y, min_dist, bounds_rect, max_passes=3)
            aep = evaluate_aep(farm, x, y)
            n_evals += 1

            if aep > best_aep:
                best_aep = aep
                best_x, best_y = x.copy(), y.copy()

            history.append(best_aep)

        elapsed = time.perf_counter() - t0
        # przywróć najlepszy layout
        evaluate_aep(farm, best_x, best_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=best_x, final_y=best_y, final_aep=best_aep,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={"eval_budget": eval_budget},
        )
