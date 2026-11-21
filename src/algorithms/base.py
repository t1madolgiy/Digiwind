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
    # Uzupełniane automatycznie po run() (LayoutAlgorithm.__init_subclass__)
    feasible: Optional[bool] = None
    min_spacing_m: Optional[float] = None

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

    def __init_subclass__(cls, **kwargs):
        """Opakowuje run() każdej podklasy kontrolą ograniczeń wyniku.

        Dzięki temu KAŻDY algorytm (także własny, dopisany wg README)
        zwraca feasible / min_spacing_m — SLSQP potrafił oddać layout z
        turbinami na sobie (0,00D) i nikt tego nie widział (błąd #5).
        """
        super().__init_subclass__(**kwargs)
        inner = cls.__dict__.get("run")
        if inner is None or getattr(inner, "_feasibility_wrapped", False):
            return

        def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params):
            res = inner(self, farm, bounds_rect, min_dist, eval_budget, seed=seed, **params)
            return check_result(res, bounds_rect, min_dist)

        run.__doc__ = inner.__doc__
        run._feasibility_wrapped = True
        cls.run = run

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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    tol = 1e-6 * max(1.0, min_dist)
    x_min, y_min, x_max, y_max = bounds_rect
    if np.any(x < x_min - tol) or np.any(x > x_max + tol):
        return False
    if np.any(y < y_min - tol) or np.any(y > y_max + tol):
        return False
    if len(x) > 1:
        if pdist(np.column_stack([x, y])).min() < min_dist - tol:
            return False
    return True


def check_result(
    res: "AlgorithmResult",
    bounds_rect: tuple[float, float, float, float],
    min_dist: float,
) -> "AlgorithmResult":
    """Ustawia res.feasible i res.min_spacing_m dla końcowego layoutu."""
    if res is None or res.final_x is None or len(res.final_x) == 0:
        return res
    fx = np.asarray(res.final_x, dtype=float)
    fy = np.asarray(res.final_y, dtype=float)
    res.min_spacing_m = float(pdist(np.column_stack([fx, fy])).min()) if len(fx) > 1 else None
    res.feasible = bool(is_feasible(fx, fy, bounds_rect, min_dist))
    return res


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
    """Generuje losowy layout spełniający min_dist (sekwencyjny Poisson disk).

    Punkty dodawane są jeden po drugim (dart throwing): kandydat jest
    przyjmowany tylko jeśli ma ≥ min_dist do już przyjętych. Dawna wersja
    losowała od razu wszystkie n punktów i dla dużych n (107) zawsze
    spadała do deterministycznej kratki — identycznej dla każdego seeda
    i niespełniającej odstępu (błąd #6).

    Jeśli w prostokącie nie da się zmieścić n punktów, uzupełnia brakujące
    punktami najdalszymi od istniejących (wynik może łamać min_dist —
    sprawdź is_feasible()).

    Args:
        max_attempts: maks. liczba kolejnych nieudanych prób dla jednego punktu.

    Returns:
        (x, y)
    """
    rng = np.random.default_rng(seed)
    x_min, y_min, x_max, y_max = bounds_rect
    pts = np.empty((0, 2))
    md2 = min_dist ** 2

    while len(pts) < n:
        placed = False
        # losuj w paczkach — szybciej niż po jednym
        for _ in range(max(1, max_attempts // 50)):
            cand = np.column_stack([
                rng.uniform(x_min, x_max, 50), rng.uniform(y_min, y_max, 50)
            ])
            if len(pts):
                d2 = ((cand[:, None, :] - pts[None, :, :]) ** 2).sum(-1).min(1)
                ok = np.flatnonzero(d2 >= md2)
            else:
                ok = np.arange(len(cand))
            if len(ok):
                pts = np.vstack([pts, cand[ok[0]]])
                placed = True
                break
        if not placed:
            break

    if len(pts) < n:
        # Uzupełnij: punkt z siatki kandydatów najdalszy od istniejących
        gx, gy = np.meshgrid(np.linspace(x_min, x_max, 60), np.linspace(y_min, y_max, 60))
        grid = np.column_stack([gx.ravel(), gy.ravel()])
        while len(pts) < n:
            if len(pts):
                d2 = ((grid[:, None, :] - pts[None, :, :]) ** 2).sum(-1).min(1)
                pts = np.vstack([pts, grid[np.argmax(d2)]])
            else:
                pts = grid[rng.integers(len(grid))][None, :]

    return pts[:, 0].copy(), pts[:, 1].copy()


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
