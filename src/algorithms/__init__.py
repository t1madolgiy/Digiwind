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
from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep, bounds_from_layout
from .scipy_gradient import ScipyGradient
from .random_search import RandomSearch
from .genetic import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .pso import ParticleSwarm
from .greedy import GreedyRelocate

ALGORITHMS: dict[str, type[LayoutAlgorithm]] = {
    "scipy": ScipyGradient,
    "random_search": RandomSearch,
    "genetic": GeneticAlgorithm,
    "simulated_annealing": SimulatedAnnealing,
    "pso": ParticleSwarm,
    "greedy": GreedyRelocate,
}

__all__ = [
    "LayoutAlgorithm", "AlgorithmResult", "evaluate_aep", "bounds_from_layout",
    "ALGORITHMS",
    "ScipyGradient", "RandomSearch", "GeneticAlgorithm",
    "SimulatedAnnealing", "ParticleSwarm", "GreedyRelocate",
]
