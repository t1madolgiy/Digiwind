"""
Hybryda GA → SLSQP.
===================
1. GA (szukanie globalne) dostaje część budżetu ewaluacji (ga_fraction),
2. SLSQP (dopracowanie lokalne) startuje z najlepszego layoutu GA i
   dostaje resztę budżetu.
Wynik nigdy nie jest gorszy od wyniku GA (SLSQP zwraca najlepszy
dopuszczalny layout, a jego start = najlepszy layout GA).

Params:
    ga_fraction   udział budżetu dla GA (domyślnie 0.5)
    evaluator     ParallelEvaluator — wspólny dla obu etapów
    pozostałe parametry GA (pop_size, mutation_rate, mutation_sigma_D,
    n_elite, init_mode, init_sigma_D) i SLSQP (fd_step_m) przekazywane dalej
"""
from __future__ import annotations

import numpy as np

from .base import LayoutAlgorithm, AlgorithmResult
from .genetic import GeneticAlgorithm
from .slsqp_parallel import SLSQPParallel

_GA_KEYS = ("pop_size", "mutation_rate", "mutation_sigma_D", "n_elite", "init_mode", "init_sigma_D")
_SLSQP_KEYS = ("fd_step_m", "maxiter")


class HybridGASLSQP(LayoutAlgorithm):
    name = "Hybrid GA → SLSQP"
    description = "GA szuka globalnie, SLSQP dopracowuje najlepszy wynik GA."
    PARAMS = [
        {"key": "ga_fraction", "label": "Udział budżetu dla GA", "type": "float", "min": 0.1, "max": 0.9, "default": 0.5, "step": 0.1},
        {"key": "pop_size", "label": "Wielkość populacji GA", "type": "int", "min": 6, "max": 60, "default": 20, "step": 2},
    ]

    def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params) -> AlgorithmResult:
        frac = float(params.get("ga_fraction", 0.5))
        ev = params.get("evaluator")
        ga_params = {k: params[k] for k in _GA_KEYS if k in params}
        sl_params = {k: params[k] for k in _SLSQP_KEYS if k in params}

        ga_budget = max(1, int(eval_budget * frac))
        r_ga = GeneticAlgorithm().run(farm, bounds_rect, min_dist, ga_budget, seed=seed,
                                      evaluator=ev, **ga_params)
        farm.set_layout_custom(r_ga.final_x, r_ga.final_y, name="_hybrid_ga")
        r_sl = SLSQPParallel().run(farm, bounds_rect, min_dist, eval_budget - r_ga.n_evaluations,
                                   seed=seed, evaluator=ev, **sl_params)

        final_aep = max(r_sl.final_aep, r_ga.final_aep)
        fx, fy = (r_sl.final_x, r_sl.final_y) if r_sl.final_aep >= r_ga.final_aep else (r_ga.final_x, r_ga.final_y)
        farm.set_layout_custom(fx, fy, name="_hybrid_best")
        return AlgorithmResult(
            name=self.name,
            initial_x=r_ga.initial_x, initial_y=r_ga.initial_y, initial_aep=r_ga.initial_aep,
            final_x=fx, final_y=fy, final_aep=final_aep,
            elapsed_s=r_ga.elapsed_s + r_sl.elapsed_s,
            n_evaluations=r_ga.n_evaluations + r_sl.n_evaluations,
            history=list(r_ga.history) + list(r_sl.history),
            extra={"ga_fraction": frac, "ga_aep": r_ga.final_aep, "slsqp_aep": r_sl.final_aep,
                   "ga_evals": r_ga.n_evaluations, "slsqp_evals": r_sl.n_evaluations,
                   "slsqp_message": r_sl.extra.get("message")},
        )
