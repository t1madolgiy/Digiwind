"""
Simulated Annealing (SA) — własna implementacja.
==================================================
W każdej iteracji:
    - Wybiera jedną turbinę i przesuwa ją o gaussian (sigma maleje z T)
    - Repair: clip do boundaries + min_dist
    - Akceptacja: jeśli lepiej → tak; jeśli gorzej → z prob exp(-ΔE / T)
    - T (temperatura) maleje geometrycznie: T = T0 * α^iter
"""
from __future__ import annotations

import time
import numpy as np

from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep, repair_min_dist


class SimulatedAnnealing(LayoutAlgorithm):
    name = "Simulated Annealing"
    description = "SA z gaussian perturbation pojedynczych turbin."

    def run(
        self,
        farm,
        bounds_rect,
        min_dist,
        eval_budget,
        seed=42,
        **params,
    ) -> AlgorithmResult:
        T0 = float(params.get("T0", 0.5))            # początkowa temperatura (w jednostkach AEP %)
        alpha = float(params.get("alpha", 0.97))     # geometric cooling
        sigma_D = float(params.get("sigma_D", 1.0))  # sigma perturbacji w średnicach

        n = farm.n_turbines
        D = farm.D
        sigma = sigma_D * D

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        x_min, y_min, x_max, y_max = bounds_rect
        rng = np.random.default_rng(seed)

        cur_x, cur_y = init_x.copy(), init_y.copy()
        cur_aep = init_aep
        best_x, best_y = cur_x.copy(), cur_y.copy()
        best_aep = cur_aep
        history = [best_aep]

        # T0 jako % obecnego AEP (dzięki temu skala zgadza się z ΔE w GWh)
        T = T0 * max(init_aep, 1.0) / 100.0
        t_start = time.perf_counter()
        n_evals = 1
        n_iters = eval_budget - 1

        for it in range(n_iters):
            # Wybierz turbinę i przesuń
            i = int(rng.integers(0, n))
            cand_x = cur_x.copy()
            cand_y = cur_y.copy()
            cand_x[i] += rng.normal(0, sigma)
            cand_y[i] += rng.normal(0, sigma)
            np.clip(cand_x, x_min, x_max, out=cand_x)
            np.clip(cand_y, y_min, y_max, out=cand_y)
            cand_x, cand_y = repair_min_dist(cand_x, cand_y, min_dist, bounds_rect, max_passes=3)

            cand_aep = evaluate_aep(farm, cand_x, cand_y)
            n_evals += 1

            dE = cand_aep - cur_aep  # >0 = lepiej
            if dE > 0 or rng.random() < np.exp(dE / max(T, 1e-9)):
                cur_x, cur_y = cand_x, cand_y
                cur_aep = cand_aep
                if cand_aep > best_aep:
                    best_aep = cand_aep
                    best_x, best_y = cand_x.copy(), cand_y.copy()

            history.append(best_aep)
            T *= alpha
            # zmniejszamy też sigma — fine-tuning pod koniec
            sigma = max(sigma * 0.999, 0.1 * D)

        elapsed = time.perf_counter() - t_start
        evaluate_aep(farm, best_x, best_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=best_x, final_y=best_y, final_aep=best_aep,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={"T0": T0, "alpha": alpha, "sigma_D_start": sigma_D},
        )
