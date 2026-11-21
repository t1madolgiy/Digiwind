"""
Rejestr algorytmów optymalizacji layoutu farmy wiatrowej.
============================================================
Każdy algorytm to osobny plik dziedziczący po `LayoutAlgorithm`.
Aby dodać nowy algorytm:
    1. Stwórz `src/algorithms/moj_algorytm.py`
    2. Zdefiniuj klasę dziedziczącą po `LayoutAlgorithm`
    3. Zaimportuj poniżej i dodaj do `ALGORITHMS`

Zakładka "🧪 Lab algorytmów" automatycznie pokaże nowy algorytm.
"""
from .base import (LayoutAlgorithm, AlgorithmResult, evaluate_aep, bounds_from_layout,
                   Region, is_feasible)
from .scipy_gradient import ScipyGradient
from .random_search import RandomSearch
from .genetic import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .pso import ParticleSwarm
from .greedy import GreedyRelocate
from .differential_evolution import DifferentialEvolution
from .slsqp_parallel import SLSQPParallel
from .hybrid import HybridGASLSQP

ALGORITHMS: dict[str, type[LayoutAlgorithm]] = {
    "scipy": ScipyGradient,
    "slsqp_par": SLSQPParallel,
    "hybrid": HybridGASLSQP,
    "random_search": RandomSearch,
    "genetic": GeneticAlgorithm,
    "differential_evolution": DifferentialEvolution,
    "simulated_annealing": SimulatedAnnealing,
    "pso": ParticleSwarm,
    "greedy": GreedyRelocate,
}

__all__ = [
    "LayoutAlgorithm", "AlgorithmResult", "evaluate_aep", "bounds_from_layout",
    "Region", "is_feasible", "ALGORITHMS",
    "ScipyGradient", "RandomSearch", "GeneticAlgorithm", "DifferentialEvolution",
    "SimulatedAnnealing", "ParticleSwarm", "GreedyRelocate",
    "SLSQPParallel", "HybridGASLSQP",
]
