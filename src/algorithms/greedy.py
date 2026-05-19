"""
Greedy Relocate — heurystyka turbina-po-turbinie.
=====================================================
Dla każdej turbiny po kolei:
    - Próbuje N losowych pozycji w lokalnym otoczeniu (gaussian, sigma = radius_D * D)
    - Zostawia tę która daje najwyższe AEP (jeżeli lepsze niż obecne)
Powtarza n_passes razy lub do braku poprawy.

Świetny baseline porównawczy — często zaskakująco dobry przy małych farmach.
"""
from __future__ import annotations

import time
import numpy as np

from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep, repair_min_dist


class GreedyRelocate(LayoutAlgorithm):
    name = "Greedy Relocate"
    description = "Po kolei każda turbina szuka najlepszej lokalnej pozycji."

    def run(
        self,
        farm,
        bounds_rect,
        min_dist,
        eval_budget,
        seed=42,
        **params,
    ) -> AlgorithmResult:
        radius_D = float(params.get("radius_D", 2.0))  # lokalny radius eksploracji
        n_trials_per_turbine = int(params.get("n_trials", 8))

        n = farm.n_turbines
        D = farm.D
        sigma = radius_D * D
        x_min, y_min, x_max, y_max = bounds_rect

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        cur_x, cur_y = init_x.copy(), init_y.copy()
        cur_aep = init_aep
        history = [cur_aep]

        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        n_evals = 1

        # ile pełnych passes się zmieści w budżecie?
        evals_per_pass = n * n_trials_per_turbine
        n_passes = max(1, (eval_budget - 1) // evals_per_pass)

        for pass_idx in range(n_passes):
            improved = False
            for i in rng.permutation(n):
                best_x_i, best_y_i = cur_x[i], cur_y[i]
                best_local_aep = cur_aep

                for _ in range(n_trials_per_turbine):
                    if n_evals >= eval_budget:
                        break
                    cand_x = cur_x.copy()
                    cand_y = cur_y.copy()
                    cand_x[i] = cur_x[i] + rng.normal(0, sigma)
                    cand_y[i] = cur_y[i] + rng.normal(0, sigma)
                    cand_x[i] = float(np.clip(cand_x[i], x_min, x_max))
                    cand_y[i] = float(np.clip(cand_y[i], y_min, y_max))
                    cand_x, cand_y = repair_min_dist(cand_x, cand_y, min_dist, bounds_rect, max_passes=3)
                    aep = evaluate_aep(farm, cand_x, cand_y)
                    n_evals += 1

                    if aep > best_local_aep:
                        best_local_aep = aep
                        best_x_i = cand_x[i]
                        best_y_i = cand_y[i]
                        improved = True

                cur_x[i] = best_x_i
                cur_y[i] = best_y_i
                cur_aep = best_local_aep
                history.append(cur_aep)

                if n_evals >= eval_budget:
                    break

            if not improved:
                break
            if n_evals >= eval_budget:
                break

        elapsed = time.perf_counter() - t0
        evaluate_aep(farm, cur_x, cur_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=cur_x, final_y=cur_y, final_aep=cur_aep,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={"radius_D": radius_D, "n_trials": n_trials_per_turbine},
        )
