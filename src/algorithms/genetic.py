"""
Genetic Algorithm (GA) — własna implementacja bez zewnętrznych zależności.
============================================================================
Operacje:
    - Selekcja: turniej (tournament size = 3)
    - Krzyżowanie: blend (BLX-α, α=0.3)
    - Mutacja: gaussian perturbation z prawdopodobieństwem mutation_rate
    - Elitism: top 2 osobników przechodzi bez zmian
    - Repair: po krzyżowaniu i mutacji wymusza min_dist + boundaries
"""
from __future__ import annotations

import time
import numpy as np

from .base import (
    LayoutAlgorithm, AlgorithmResult, evaluate_aep,
    random_feasible_layout, repair_min_dist,
)


def _encode(x, y):
    return np.concatenate([x, y])


def _decode(genome, n):
    return genome[:n].copy(), genome[n:].copy()


class GeneticAlgorithm(LayoutAlgorithm):
    name = "Genetic Algorithm"
    description = "GA z BLX-α crossover + gaussian mutation + elitism."
    PARAMS = [
        {"key": "pop_size", "label": "Wielkość populacji", "type": "int", "min": 6, "max": 60, "default": 20, "step": 2},
        {"key": "mutation_rate", "label": "Prawdopodobieństwo mutacji", "type": "float", "min": 0.0, "max": 1.0, "default": 0.2, "step": 0.05},
        {"key": "mutation_sigma_D", "label": "Siła mutacji [×D]", "type": "float", "min": 0.1, "max": 2.0, "default": 0.5, "step": 0.1},
        {"key": "n_elite", "label": "Elita (najlepsi bez zmian)", "type": "int", "min": 0, "max": 6, "default": 2, "step": 1},
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
        pop_size = int(params.get("pop_size", 20))
        mutation_rate = float(params.get("mutation_rate", 0.2))
        mutation_sigma_D = float(params.get("mutation_sigma_D", 0.5))
        n_elite = int(params.get("n_elite", 2))

        n = farm.n_turbines
        D = farm.D
        mutation_sigma = mutation_sigma_D * D

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        x_min, y_min, x_max, y_max = bounds_rect
        rng = np.random.default_rng(seed)

        # --- Populacja startowa ---
        population = [_encode(init_x, init_y)]  # zawsze start od obecnego
        for _ in range(pop_size - 1):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            x, y = random_feasible_layout(n, bounds_rect, min_dist, seed=sub_seed)
            x, y = repair_min_dist(x, y, min_dist, bounds_rect)
            population.append(_encode(x, y))

        # --- Ewaluacja startowa ---
        fitness = []
        for g in population:
            xx, yy = _decode(g, n)
            fitness.append(evaluate_aep(farm, xx, yy))
        fitness = np.array(fitness)
        n_evals = pop_size

        best_idx = int(np.argmax(fitness))
        best_aep = float(fitness[best_idx])
        best_x, best_y = _decode(population[best_idx], n)
        history = [best_aep]

        t0 = time.perf_counter()

        n_generations = max(1, (eval_budget - pop_size) // pop_size)

        for gen in range(n_generations):
            # Elitism: zachowaj top-N
            elite_idx = np.argsort(fitness)[-n_elite:]
            new_pop = [population[i].copy() for i in elite_idx]

            # Generuj resztę
            tour_k = min(3, pop_size)  # turniej nie może być większy niż populacja
            while len(new_pop) < pop_size:
                # Tournament selection
                i1 = rng.choice(pop_size, tour_k, replace=False)
                i2 = rng.choice(pop_size, tour_k, replace=False)
                p1 = population[i1[np.argmax(fitness[i1])]]
                p2 = population[i2[np.argmax(fitness[i2])]]

                # BLX-α crossover
                alpha = 0.3
                lo = np.minimum(p1, p2) - alpha * np.abs(p1 - p2)
                hi = np.maximum(p1, p2) + alpha * np.abs(p1 - p2)
                child = rng.uniform(lo, hi)

                # Mutacja
                mut_mask = rng.random(len(child)) < mutation_rate
                child[mut_mask] += rng.normal(0, mutation_sigma, mut_mask.sum())

                # Repair: boundaries + min_dist
                cx, cy = _decode(child, n)
                np.clip(cx, x_min, x_max, out=cx)
                np.clip(cy, y_min, y_max, out=cy)
                cx, cy = repair_min_dist(cx, cy, min_dist, bounds_rect, max_passes=5)
                new_pop.append(_encode(cx, cy))

            population = new_pop
            # Re-ewaluacja
            fitness = []
            for g in population:
                xx, yy = _decode(g, n)
                fitness.append(evaluate_aep(farm, xx, yy))
            fitness = np.array(fitness)
            n_evals += pop_size

            cur_best_idx = int(np.argmax(fitness))
            if fitness[cur_best_idx] > best_aep:
                best_aep = float(fitness[cur_best_idx])
                best_x, best_y = _decode(population[cur_best_idx], n)
            history.append(best_aep)

        elapsed = time.perf_counter() - t0
        evaluate_aep(farm, best_x, best_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=best_x, final_y=best_y, final_aep=best_aep,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={
                "pop_size": pop_size,
                "n_generations": n_generations,
                "mutation_rate": mutation_rate,
                "mutation_sigma_D": mutation_sigma_D,
            },
        )
