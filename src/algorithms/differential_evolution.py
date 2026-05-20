"""
Differential Evolution (DE) — własna implementacja (rand/1/bin).
==================================================================
Silny globalny optymalizator dla zmiennych ciągłych. Każdy osobnik to layout
zakodowany jako wektor [x_0..x_{N-1}, y_0..y_{N-1}].

W każdej generacji, dla każdego osobnika i:
    - wylosuj 3 różne osobniki a, b, c
    - wektor próbny:  v = a + F * (b - c)
    - krzyżowanie dwumianowe (CR) v z osobnikiem i  ->  u
    - repair: clip do boundaries + min_dist
    - selekcja: jeśli AEP(u) > AEP(i)  ->  zastąp

DE jest mniej wrażliwy na lokalne minima niż gradient i często bije GA/PSO
przy tym samym budżecie ewaluacji.
"""
from __future__ import annotations

import time
import numpy as np

from .base import (
    LayoutAlgorithm, AlgorithmResult, evaluate_aep,
    random_feasible_layout, repair_min_dist,
)


class DifferentialEvolution(LayoutAlgorithm):
    name = "Differential Evolution"
    description = "DE rand/1/bin — silny globalny optymalizator ciągły."
    PARAMS = [
        {"key": "pop_size", "label": "Wielkość populacji", "type": "int", "min": 6, "max": 60, "default": 20, "step": 2},
        {"key": "F", "label": "Współczynnik mutacji F", "type": "float", "min": 0.1, "max": 1.5, "default": 0.6, "step": 0.05},
        {"key": "CR", "label": "Prawdopodobieństwo krzyżowania CR", "type": "float", "min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05},
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
        pop_size = max(4, int(params.get("pop_size", 20)))
        F = float(params.get("F", 0.6))
        CR = float(params.get("CR", 0.8))

        n = farm.n_turbines
        dim = 2 * n
        x_min, y_min, x_max, y_max = bounds_rect

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        rng = np.random.default_rng(seed)

        def _decode(vec):
            return vec[:n].copy(), vec[n:].copy()

        def _clip_repair(vec):
            xx, yy = _decode(vec)
            np.clip(xx, x_min, x_max, out=xx)
            np.clip(yy, y_min, y_max, out=yy)
            xx, yy = repair_min_dist(xx, yy, min_dist, bounds_rect, max_passes=3)
            return np.concatenate([xx, yy])

        # --- Populacja startowa (osobnik 0 = obecny layout) ---
        pop = [np.concatenate([init_x, init_y])]
        for _ in range(pop_size - 1):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            x, y = random_feasible_layout(n, bounds_rect, min_dist, seed=sub_seed)
            x, y = repair_min_dist(x, y, min_dist, bounds_rect)
            pop.append(np.concatenate([x, y]))
        pop = np.array(pop, dtype=float)

        fitness = np.array([evaluate_aep(farm, *_decode(ind)) for ind in pop])
        n_evals = pop_size

        best_idx = int(np.argmax(fitness))
        best_vec = pop[best_idx].copy()
        best_aep = float(fitness[best_idx])
        history = [best_aep]

        t0 = time.perf_counter()

        while n_evals < eval_budget:
            for i in range(pop_size):
                if n_evals >= eval_budget:
                    break
                # 3 różne osobniki != i
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = pop[rng.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)

                # krzyżowanie dwumianowe
                cross = rng.random(dim) < CR
                if not np.any(cross):
                    cross[rng.integers(0, dim)] = True
                trial = np.where(cross, mutant, pop[i])
                trial = _clip_repair(trial)

                trial_aep = evaluate_aep(farm, *_decode(trial))
                n_evals += 1

                if trial_aep >= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_aep
                    if trial_aep > best_aep:
                        best_aep = float(trial_aep)
                        best_vec = trial.copy()

            history.append(best_aep)

        elapsed = time.perf_counter() - t0
        best_x, best_y = _decode(best_vec)
        evaluate_aep(farm, best_x, best_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=best_x, final_y=best_y, final_aep=best_aep,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={"pop_size": pop_size, "F": F, "CR": CR},
        )
