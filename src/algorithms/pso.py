"""
Particle Swarm Optimization (PSO) — własna implementacja.
============================================================
Każda "cząstka" to layout (x_i, y_i, ..., x_N, y_N).
W każdej iteracji:
    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x <- x + v
    repair: clip do boundaries + min_dist
"""
from __future__ import annotations

import time
import numpy as np

from .base import (
    LayoutAlgorithm, AlgorithmResult, evaluate_aep,
    random_feasible_layout, repair_min_dist,
)


class ParticleSwarm(LayoutAlgorithm):
    name = "Particle Swarm (PSO)"
    description = "Roj cząstek poruszających się w przestrzeni layoutów."

    def run(
        self,
        farm,
        bounds_rect,
        min_dist,
        eval_budget,
        seed=42,
        **params,
    ) -> AlgorithmResult:
        n_particles = int(params.get("n_particles", 15))
        w = float(params.get("inertia", 0.7))
        c1 = float(params.get("cognitive", 1.5))
        c2 = float(params.get("social", 1.5))

        n = farm.n_turbines
        D = farm.D
        x_min, y_min, x_max, y_max = bounds_rect
        v_max = 2.0 * D  # max prędkość per krok

        init_x = farm.layout_x.copy()
        init_y = farm.layout_y.copy()
        init_aep = evaluate_aep(farm, init_x, init_y)

        rng = np.random.default_rng(seed)

        # --- Inicjalizacja cząstek ---
        positions = []  # każda: array długości 2N
        velocities = []
        positions.append(np.concatenate([init_x, init_y]))  # 0. cząstka = obecny layout
        velocities.append(rng.uniform(-v_max / 2, v_max / 2, 2 * n))

        for _ in range(n_particles - 1):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            x, y = random_feasible_layout(n, bounds_rect, min_dist, seed=sub_seed)
            x, y = repair_min_dist(x, y, min_dist, bounds_rect)
            positions.append(np.concatenate([x, y]))
            velocities.append(rng.uniform(-v_max, v_max, 2 * n))

        positions = np.array(positions)
        velocities = np.array(velocities)

        # --- Ewaluacja startowa ---
        fitness = np.zeros(n_particles)
        for i in range(n_particles):
            fitness[i] = evaluate_aep(farm, positions[i, :n], positions[i, n:])
        n_evals = n_particles

        pbest_pos = positions.copy()
        pbest_fit = fitness.copy()
        g_idx = int(np.argmax(pbest_fit))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])
        history = [max(gbest_fit, init_aep)]

        t0 = time.perf_counter()
        n_iters = max(1, (eval_budget - n_particles) // n_particles)

        for it in range(n_iters):
            r1 = rng.random((n_particles, 2 * n))
            r2 = rng.random((n_particles, 2 * n))
            velocities = (
                w * velocities
                + c1 * r1 * (pbest_pos - positions)
                + c2 * r2 * (gbest_pos[None, :] - positions)
            )
            np.clip(velocities, -v_max, v_max, out=velocities)
            positions = positions + velocities

            # Repair + ewaluacja każdej
            for i in range(n_particles):
                xx = positions[i, :n]
                yy = positions[i, n:]
                np.clip(xx, x_min, x_max, out=xx)
                np.clip(yy, y_min, y_max, out=yy)
                xx, yy = repair_min_dist(xx, yy, min_dist, bounds_rect, max_passes=3)
                positions[i, :n] = xx
                positions[i, n:] = yy
                aep = evaluate_aep(farm, xx, yy)
                fitness[i] = aep
                n_evals += 1

                if aep > pbest_fit[i]:
                    pbest_fit[i] = aep
                    pbest_pos[i] = positions[i].copy()

            cur_g = int(np.argmax(pbest_fit))
            if pbest_fit[cur_g] > gbest_fit:
                gbest_fit = float(pbest_fit[cur_g])
                gbest_pos = pbest_pos[cur_g].copy()

            history.append(gbest_fit)

        elapsed = time.perf_counter() - t0

        best_x = gbest_pos[:n]
        best_y = gbest_pos[n:]
        evaluate_aep(farm, best_x, best_y)

        return AlgorithmResult(
            name=self.name,
            initial_x=init_x, initial_y=init_y, initial_aep=init_aep,
            final_x=best_x, final_y=best_y, final_aep=gbest_fit,
            elapsed_s=elapsed,
            n_evaluations=n_evals,
            history=history,
            extra={"n_particles": n_particles, "w": w, "c1": c1, "c2": c2},
        )
