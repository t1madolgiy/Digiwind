"""
Bazowy interfejs algorytmów optymalizacji layoutu farmy wiatrowej.
==================================================================
Każdy algorytm to osobny plik w `src/algorithms/`, dziedziczący po
`LayoutAlgorithm`. Wynik jest standardowym `AlgorithmResult` co pozwala
porównywać różne podejścia w zakładce "🧪 Lab algorytmów".

Konwencje:
    - Algorytm dostaje obiekt FarmModel z już ustawionym wind_data
      (zazwyczaj wąski bin: jeden WD + jeden WS, do szybkiej ewaluacji)
    - Ograniczenia: rectangle (x_min, y_min, x_max, y_max) + min_dist [m]
    - Eval budget: liczba wywołań farm.run() (różne algo używają różnie)
    - Zwraca pełną historię (best_aep per iteracja) do wykresu zbieżności
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist


@dataclass
class AlgorithmResult:
    """Standardowy wynik algorytmu — do porównań między algorytmami."""
    name: str
    initial_x: np.ndarray
    initial_y: np.ndarray
    initial_aep: float
    final_x: np.ndarray
    final_y: np.ndarray
    final_aep: float
    elapsed_s: float
    n_evaluations: int
    history: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def improvement_pct(self) -> float:
        if self.initial_aep <= 0:
            return 0.0
        return (self.final_aep - self.initial_aep) / self.initial_aep * 100.0

    @property
    def improvement_gwh(self) -> float:
        return self.final_aep - self.initial_aep


class LayoutAlgorithm(ABC):
    """Bazowa klasa wszystkich algorytmów optymalizacji layoutu."""
    name: str = "base"
    description: str = ""
    # Strojalne parametry — UI (Lab algorytmów) generuje z tego suwaki.
    # Każdy wpis: {"key","label","type" ("int"|"float"),"min","max","default","step"}
    PARAMS: list[dict] = []

    @abstractmethod
    def run(
        self,
        farm,
        bounds_rect: tuple[float, float, float, float],
        min_dist: float,
        eval_budget: int,
        seed: int = 42,
        **params,
    ) -> AlgorithmResult:
        """Uruchom algorytm.

        Args:
            farm: FarmModel z ustawionym wind_data (zazwyczaj wąski bin).
            bounds_rect: (x_min, y_min, x_max, y_max) w metrach.
            min_dist: minimalna odległość między turbinami [m].
            eval_budget: ile wywołań ewaluacji AEP wolno wykonać.
            seed: ziarno losowości (powtarzalność).
            **params: dodatkowe parametry specyficzne dla algorytmu.

        Returns:
            AlgorithmResult.
        """
        ...


# ---------------------------------------------------------------------------
# Helpery wspólne dla wszystkich algorytmów
# ---------------------------------------------------------------------------
def evaluate_aep(farm, x: np.ndarray, y: np.ndarray) -> float:
    """Ewaluacja AEP dla layoutu (x, y). Zwraca 0.0 przy błędzie."""
    try:
        farm.set_layout_custom(np.asarray(x, dtype=float), np.asarray(y, dtype=float), name="_eval")
        farm.run()
        aep = float(farm.get_aep_gwh())
        return aep if np.isfinite(aep) else 0.0
    except Exception:
        return 0.0


def is_feasible(
    x: np.ndarray,
    y: np.ndarray,
    bounds_rect: tuple[float, float, float, float],
    min_dist: float,
) -> bool:
    """Sprawdza czy layout mieści się w boundaries i spełnia min_dist."""
    x_min, y_min, x_max, y_max = bounds_rect
    if np.any(x < x_min) or np.any(x > x_max):
        return False
    if np.any(y < y_min) or np.any(y > y_max):
        return False
    if len(x) > 1:
        if pdist(np.column_stack([x, y])).min() < min_dist:
            return False
    return True


def repair_min_dist(
    x: np.ndarray,
    y: np.ndarray,
    min_dist: float,
    bounds_rect: tuple[float, float, float, float],
    max_passes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteracyjnie odpycha pary turbin, które są za blisko.

    Prosty repair: dla każdej zbyt bliskiej pary — odsuń obie wzdłuż wektora
    łączącego, potem przytnij do boundaries. Może nie zbiec dla bardzo
    ciasnych boundaries, ale dla 3x3/4x4 z marginesem wystarcza.
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    n = len(x)
    x_min, y_min, x_max, y_max = bounds_rect

    for _ in range(max_passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d = np.hypot(dx, dy)
                if d < min_dist and d > 1e-6:
                    push = (min_dist - d) / 2.0 + 1.0
                    ux, uy = dx / d, dy / d
                    x[i] -= ux * push
                    y[i] -= uy * push
                    x[j] += ux * push
                    y[j] += uy * push
                    moved = True
                elif d <= 1e-6:
                    # nakładające się — pchnij losowo
                    x[j] += min_dist
                    moved = True
        # boundaries
        np.clip(x, x_min, x_max, out=x)
        np.clip(y, y_min, y_max, out=y)
        if not moved:
            break

    return x, y


def random_feasible_layout(
    n: int,
    bounds_rect: tuple[float, float, float, float],
    min_dist: float,
    seed: int = 42,
    max_attempts: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generuje losowy poprawny layout (Poisson disk sampling, naive).

    Returns:
        (x, y) lub (None, None) jeśli nie udało się znaleźć w max_attempts.
    """
    rng = np.random.default_rng(seed)
    x_min, y_min, x_max, y_max = bounds_rect

    for _ in range(max_attempts):
        x = rng.uniform(x_min, x_max, n)
        y = rng.uniform(y_min, y_max, n)
        if n == 1 or pdist(np.column_stack([x, y])).min() >= min_dist:
            return x, y

    # Fallback: kratka + jitter
    side = int(np.ceil(np.sqrt(n)))
    xs = np.linspace(x_min + min_dist, x_max - min_dist, side)
    ys = np.linspace(y_min + min_dist, y_max - min_dist, side)
    xx, yy = np.meshgrid(xs, ys)
    x = xx.flatten()[:n]
    y = yy.flatten()[:n]
    return x, y


def bounds_from_layout(
    x: np.ndarray,
    y: np.ndarray,
    margin: float,
) -> tuple[float, float, float, float]:
    """Bounding rectangle z marginesem (w metrach)."""
    return (
        float(x.min() - margin),
        float(y.min() - margin),
        float(x.max() + margin),
        float(y.max() + margin),
    )
