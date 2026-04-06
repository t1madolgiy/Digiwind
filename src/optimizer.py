"""
Optimizer for Wind Farm Layout Optimization Project
=====================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Owija FLORIS optimization tools:
    - LayoutOptimizationScipy — gradient-based layout optimization
    - LayoutOptimizationRandomSearch — genetic/random search
    - YawOptimizationSR — SerialRefine yaw angle optimization
    - Porównanie przed/po optymalizacji
    - Wizualizacja wyników

Autor: Temat 2
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from floris import FlorisModel, WindRose, TimeSeries
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)
from floris.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)

from .farm_model import FarmModel, WAKE_MODELS, TURBINE_LIBRARY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfiguracja optymalizacji
# ---------------------------------------------------------------------------
@dataclass
class OptimizationConfig:
    """Parametry optymalizacji layoutu."""
    # Granice obszaru — wielokąt [(x1,y1), (x2,y2), ...]
    boundaries: list[tuple[float, float]] = field(default_factory=list)

    # Minimalna odległość między turbinami [m]
    min_dist_m: Optional[float] = None  # Jeśli None, oblicza z 3D

    # Scipy options
    scipy_maxiter: int = 100

    # RandomSearch options
    random_search_seconds: float = 60.0
    random_search_seed: int = 42

    # Yaw optimization
    yaw_min: float = 0.0
    yaw_max: float = 25.0


# ---------------------------------------------------------------------------
# Wynik optymalizacji
# ---------------------------------------------------------------------------
@dataclass
class OptimizationResult:
    """Przechowuje wyniki optymalizacji."""
    method: str
    # Layout przed
    initial_x: np.ndarray
    initial_y: np.ndarray
    initial_aep_gwh: float
    # Layout po
    optimized_x: np.ndarray
    optimized_y: np.ndarray
    optimized_aep_gwh: float
    # Metryki
    aep_improvement_pct: float
    elapsed_seconds: float
    # Yaw (opcjonalnie)
    yaw_angles: Optional[np.ndarray] = None
    yaw_aep_gwh: Optional[float] = None


# ---------------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------------
class Optimizer:
    """Optymalizator layoutu farmy wiatrowej.

    Użycie:
        >>> opt = Optimizer(farm)
        >>> opt.set_boundaries_rectangle(width=5000, height=5000)
        >>> result = opt.optimize_layout_scipy()
        >>> opt.plot_optimization_result(result)
        >>> opt.apply_result(result)  # aktualizuje farm
    """

    def __init__(self, farm: FarmModel):
        """
        Args:
            farm: FarmModel z ustawionym layoutem i danymi wiatrowymi.
        """
        self.farm = farm
        self._boundaries = None
        self._min_dist = 3.0 * farm.D  # domyślnie 3D

    # ------------------------------------------------------------------
    # Granice obszaru
    # ------------------------------------------------------------------
    def set_boundaries_rectangle(
        self,
        width: float,
        height: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> None:
        """Ustawia prostokątne granice obszaru.

        Args:
            width: Szerokość [m].
            height: Wysokość [m].
            origin_x: X lewego dolnego rogu [m].
            origin_y: Y lewego dolnego rogu [m].
        """
        self._boundaries = [
            (origin_x, origin_y),
            (origin_x + width, origin_y),
            (origin_x + width, origin_y + height),
            (origin_x, origin_y + height),
            (origin_x, origin_y),  # zamknięcie
        ]
        logger.info(f"Granice prostokątne: {width:.0f}m x {height:.0f}m")

    def set_boundaries_from_layout(
        self,
        margin_D: float = 2.0,
    ) -> None:
        """Ustawia granice na podstawie aktualnego layoutu + margines.

        Args:
            margin_D: Margines w wielokrotnościach D.
        """
        margin = margin_D * self.farm.D
        x = self.farm.layout_x
        y = self.farm.layout_y

        x_min, x_max = x.min() - margin, x.max() + margin
        y_min, y_max = y.min() - margin, y.max() + margin

        self.set_boundaries_rectangle(
            width=x_max - x_min,
            height=y_max - y_min,
            origin_x=x_min,
            origin_y=y_min,
        )

    def set_boundaries_custom(
        self,
        boundaries: list[tuple[float, float]],
    ) -> None:
        """Ustawia niestandardowe granice (wielokąt).

        Args:
            boundaries: Lista wierzchołków [(x1,y1), (x2,y2), ...].
                        Musi być zamknięty (ostatni punkt = pierwszy).
        """
        self._boundaries = boundaries

    def set_min_distance(self, min_dist_D: float = 3.0) -> None:
        """Ustawia minimalną odległość między turbinami.

        Args:
            min_dist_D: Minimalna odległość w wielokrotnościach D.
        """
        self._min_dist = min_dist_D * self.farm.D
        logger.info(f"Min. odległość: {min_dist_D}D = {self._min_dist:.0f}m")

    # ------------------------------------------------------------------
    # Layout optimization — Scipy (gradient-based)
    # ------------------------------------------------------------------
    def optimize_layout_scipy(
        self,
        maxiter: int = 100,
        use_value: bool = False,
    ) -> OptimizationResult:
        """Optymalizacja layoutu metodą Scipy (gradient-based).

        Szybka, ale może utknąć w lokalnym minimum.
        Dobra do fine-tuning istniejącego layoutu.

        Args:
            maxiter: Maksymalna liczba iteracji.
            use_value: Jeśli True, optymalizuje AVP zamiast AEP.

        Returns:
            OptimizationResult z wynikami.
        """
        self._validate_ready()

        # Zapamiętaj początkowy layout
        init_x = self.farm.layout_x.copy()
        init_y = self.farm.layout_y.copy()

        # Oblicz początkowe AEP
        self.farm.run()
        init_aep = self.farm.get_aep_gwh()

        logger.info(
            f"Scipy optimization: {self.farm.n_turbines} turbin, "
            f"start AEP={init_aep:.2f} GWh"
        )

        # FLORIS optimizer
        opt_options = {"maxiter": maxiter, "disp": True}
        layout_opt = LayoutOptimizationScipy(
            self.farm.fmodel,
            self._boundaries,
            min_dist=self._min_dist,
            optOptions=opt_options,
            use_value=use_value,
        )

        t0 = time.perf_counter()
        sol = layout_opt.optimize()
        elapsed = time.perf_counter() - t0

        opt_x = np.array(sol[0])
        opt_y = np.array(sol[1])

        # Oblicz AEP po optymalizacji
        self.farm.set_layout_custom(opt_x, opt_y, name="scipy_optimized")
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()
        opt_aep = self.farm.get_aep_gwh()

        improvement = (opt_aep - init_aep) / init_aep * 100

        result = OptimizationResult(
            method="scipy",
            initial_x=init_x,
            initial_y=init_y,
            initial_aep_gwh=init_aep,
            optimized_x=opt_x,
            optimized_y=opt_y,
            optimized_aep_gwh=opt_aep,
            aep_improvement_pct=improvement,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"Scipy done: AEP {init_aep:.2f} → {opt_aep:.2f} GWh "
            f"(+{improvement:.2f}%) w {elapsed:.1f}s"
        )

        return result

    # ------------------------------------------------------------------
    # Layout optimization — Random Search (genetic)
    # ------------------------------------------------------------------
    def optimize_layout_random_search(
        self,
        seconds: float = 60.0,
        seed: int = 42,
        use_value: bool = False,
    ) -> OptimizationResult:
        """Optymalizacja layoutu metodą Random Search.

        Wolniejsza, ale lepiej radzi sobie z dużymi farmami
        i nie utknie w lokalnym minimum.

        Args:
            seconds: Czas optymalizacji [s].
            seed: Ziarno generatora losowego.
            use_value: Jeśli True, optymalizuje AVP zamiast AEP.

        Returns:
            OptimizationResult z wynikami.
        """
        self._validate_ready()

        # Zapamiętaj początkowy layout
        init_x = self.farm.layout_x.copy()
        init_y = self.farm.layout_y.copy()

        self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()
        init_aep = self.farm.get_aep_gwh()

        logger.info(
            f"RandomSearch optimization: {self.farm.n_turbines} turbin, "
            f"time_limit={seconds}s, start AEP={init_aep:.2f} GWh"
        )

        # Świeży FlorisModel z tego samego configa — obejście buga FLORIS
        # gdzie rotor_diameters po run() staje się tablicą 2D
        # i LayoutOptimizationRandomSearch.__init__ się na niej wykłada.
        fresh_fmodel = FlorisModel(self.farm.fmodel.core.as_dict())
        fresh_fmodel.set(
            layout_x=init_x.tolist(),
            layout_y=init_y.tolist(),
            wind_data=self.farm._wind_data,
        )

        layout_opt = LayoutOptimizationRandomSearch(
            fresh_fmodel,
            self._boundaries,
            min_dist_D=self._min_dist / self.farm.D,
            seconds_per_iteration=seconds,
            total_optimization_seconds=seconds,
            random_seed=seed,
            use_value=use_value,
        )

        t0 = time.perf_counter()
        sol = layout_opt.optimize()
        elapsed = time.perf_counter() - t0

        opt_x = np.array(sol[0])
        opt_y = np.array(sol[1])

        # Oblicz AEP po optymalizacji
        self.farm.set_layout_custom(opt_x, opt_y, name="random_search_optimized")
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()
        opt_aep = self.farm.get_aep_gwh()

        improvement = (opt_aep - init_aep) / init_aep * 100

        result = OptimizationResult(
            method="random_search",
            initial_x=init_x,
            initial_y=init_y,
            initial_aep_gwh=init_aep,
            optimized_x=opt_x,
            optimized_y=opt_y,
            optimized_aep_gwh=opt_aep,
            aep_improvement_pct=improvement,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"RandomSearch done: AEP {init_aep:.2f} → {opt_aep:.2f} GWh "
            f"(+{improvement:.2f}%) w {elapsed:.1f}s"
        )

        return result

    # ------------------------------------------------------------------
    # Yaw optimization — SerialRefine
    # ------------------------------------------------------------------
    def optimize_yaw(
        self,
        yaw_min: float = 0.0,
        yaw_max: float = 25.0,
    ) -> OptimizationResult:
        """Optymalizacja kątów yaw (wake steering).

        Znajduje optymalne kąty yaw dla każdej turbiny
        w zależności od kierunku wiatru. Przydatne dla Tematu 3.

        Args:
            yaw_min: Minimalny kąt yaw [°].
            yaw_max: Maksymalny kąt yaw [°].

        Returns:
            OptimizationResult z yaw_angles.
        """
        if self.farm._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych.")

        # AEP bez yaw
        self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()
        init_aep = self.farm.get_aep_gwh()

        logger.info(f"Yaw optimization: start AEP={init_aep:.2f} GWh")

        yaw_opt = YawOptimizationSR(
            fmodel=self.farm.fmodel,
            minimum_yaw_angle=yaw_min,
            maximum_yaw_angle=yaw_max,
            Ny_passes=[5, 4],
            exclude_downstream_turbines=True,
        )

        t0 = time.perf_counter()
        df_opt = yaw_opt.optimize()
        elapsed = time.perf_counter() - t0

        # Wyciągnij optymalne kąty yaw
        yaw_angles = yaw_opt.yaw_angles_opt

        # Oblicz AEP z optymalnymi kątami yaw
        self.farm.fmodel.set(yaw_angles=yaw_angles)
        self.farm.fmodel.run()
        yaw_aep = self.farm.get_aep_gwh()

        improvement = (yaw_aep - init_aep) / init_aep * 100

        # Reset yaw
        self.farm.fmodel.set(
            yaw_angles=np.zeros_like(yaw_angles),
            wind_data=self.farm._wind_data,
        )

        result = OptimizationResult(
            method="yaw_sr",
            initial_x=self.farm.layout_x,
            initial_y=self.farm.layout_y,
            initial_aep_gwh=init_aep,
            optimized_x=self.farm.layout_x,
            optimized_y=self.farm.layout_y,
            optimized_aep_gwh=yaw_aep,
            aep_improvement_pct=improvement,
            elapsed_seconds=elapsed,
            yaw_angles=yaw_angles,
            yaw_aep_gwh=yaw_aep,
        )

        logger.info(
            f"Yaw optimization done: AEP {init_aep:.2f} → {yaw_aep:.2f} GWh "
            f"(+{improvement:.2f}%) w {elapsed:.1f}s"
        )

        return result

    # ------------------------------------------------------------------
    # Aplikacja wyniku
    # ------------------------------------------------------------------
    def apply_result(self, result: OptimizationResult) -> None:
        """Aplikuje wynik optymalizacji do FarmModel.

        Args:
            result: Wynik z optimize_layout_* lub optimize_yaw.
        """
        self.farm.set_layout_custom(
            result.optimized_x,
            result.optimized_y,
            name=f"optimized_{result.method}",
        )
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)
        logger.info(f"Zaaplikowano wynik: {result.method}")

    # ------------------------------------------------------------------
    # Wizualizacja
    # ------------------------------------------------------------------
    def plot_optimization_result(
        self,
        result: OptimizationResult,
        figsize: tuple = (16, 6),
    ) -> plt.Figure:
        """Rysuje porównanie layoutu przed i po optymalizacji.

        Args:
            result: Wynik optymalizacji.
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: Layout PRZED
        ax = axes[0]
        ax.scatter(
            result.initial_x, result.initial_y,
            s=80, c="#c8531a", edgecolors="white", linewidths=1, zorder=5,
        )
        if self._boundaries:
            bx = [b[0] for b in self._boundaries]
            by = [b[1] for b in self._boundaries]
            ax.plot(bx, by, "k--", linewidth=1, alpha=0.5)
        ax.set_title(f"Przed\nAEP = {result.initial_aep_gwh:.2f} GWh")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Panel 2: Layout PO
        ax = axes[1]
        ax.scatter(
            result.optimized_x, result.optimized_y,
            s=80, c="#1e5c3a", edgecolors="white", linewidths=1, zorder=5,
        )
        if self._boundaries:
            ax.plot(bx, by, "k--", linewidth=1, alpha=0.5)
        ax.set_title(f"Po ({result.method})\nAEP = {result.optimized_aep_gwh:.2f} GWh")
        ax.set_xlabel("X [m]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Panel 3: Nałożone + strzałki przesunięcia
        ax = axes[2]
        ax.scatter(
            result.initial_x, result.initial_y,
            s=60, c="#c8531a", alpha=0.4, label="Przed", zorder=4,
        )
        ax.scatter(
            result.optimized_x, result.optimized_y,
            s=60, c="#1e5c3a", edgecolors="white", linewidths=1,
            label="Po", zorder=5,
        )
        # Strzałki przesunięcia
        for i in range(len(result.initial_x)):
            dx = result.optimized_x[i] - result.initial_x[i]
            dy = result.optimized_y[i] - result.initial_y[i]
            if np.sqrt(dx**2 + dy**2) > 1.0:  # nie rysuj znikomo małych
                ax.annotate(
                    "", xy=(result.optimized_x[i], result.optimized_y[i]),
                    xytext=(result.initial_x[i], result.initial_y[i]),
                    arrowprops=dict(arrowstyle="->", color="#534AB7", lw=1.2, alpha=0.6),
                )
        if self._boundaries:
            ax.plot(bx, by, "k--", linewidth=1, alpha=0.5)
        ax.set_title(
            f"Porównanie\n+{result.aep_improvement_pct:.2f}% AEP | "
            f"{result.elapsed_seconds:.1f}s"
        )
        ax.set_xlabel("X [m]")
        ax.set_aspect("equal")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Optymalizacja layoutu — {result.method} | "
            f"{self.farm.turbine_info['name']} | {self.farm.n_turbines} turbin",
            fontsize=14, fontweight="500",
        )
        fig.tight_layout()
        return fig

    def plot_yaw_result(
        self,
        result: OptimizationResult,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """Rysuje optymalne kąty yaw per kierunek wiatru.

        Args:
            result: Wynik z optimize_yaw().
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        if result.yaw_angles is None:
            raise ValueError("Wynik nie zawiera kątów yaw.")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Heatmapa yaw per turbina per warunek
        ax = axes[0]
        im = ax.imshow(
            result.yaw_angles.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-result.yaw_angles.max(),
            vmax=result.yaw_angles.max(),
        )
        ax.set_xlabel("Warunek wiatrowy (index)")
        ax.set_ylabel("Turbina ID")
        ax.set_title("Kąty yaw [°]")
        plt.colorbar(im, ax=ax, label="Yaw [°]")

        # Panel 2: Średni yaw per turbina
        ax = axes[1]
        mean_yaw = np.mean(np.abs(result.yaw_angles), axis=0)
        turbine_ids = np.arange(len(mean_yaw))
        ax.barh(turbine_ids, mean_yaw, color="#534AB7", alpha=0.8)
        ax.set_xlabel("Średni |yaw| [°]")
        ax.set_ylabel("Turbina ID")
        ax.set_title("Średni kąt yaw per turbina")
        ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            f"Yaw optimization — AEP: {result.initial_aep_gwh:.2f} → "
            f"{result.yaw_aep_gwh:.2f} GWh (+{result.aep_improvement_pct:.2f}%)",
            fontsize=14, fontweight="500",
        )
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Porównanie metod
    # ------------------------------------------------------------------
    def compare_methods(
        self,
        scipy_maxiter: int = 50,
        random_search_seconds: float = 30.0,
        include_yaw: bool = True,
    ) -> dict[str, OptimizationResult]:
        """Porównuje metody optymalizacji na tym samym layoucie.

        Args:
            scipy_maxiter: Iteracje Scipy.
            random_search_seconds: Czas RandomSearch [s].
            include_yaw: Czy uwzględnić optymalizację yaw.

        Returns:
            Dict {nazwa_metody: OptimizationResult}.
        """
        # Zapamiętaj oryginalny layout
        orig_x = self.farm.layout_x.copy()
        orig_y = self.farm.layout_y.copy()

        results = {}

        # 1. Scipy
        print("Optymalizacja Scipy...")
        self.farm.set_layout_custom(orig_x, orig_y, name="initial")
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)
        results["scipy"] = self.optimize_layout_scipy(maxiter=scipy_maxiter)

        # 2. Random Search
        print("Optymalizacja RandomSearch...")
        self.farm.set_layout_custom(orig_x, orig_y, name="initial")
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)
        results["random_search"] = self.optimize_layout_random_search(
            seconds=random_search_seconds
        )

        # 3. Yaw (opcjonalnie)
        if include_yaw:
            print("Optymalizacja Yaw...")
            self.farm.set_layout_custom(orig_x, orig_y, name="initial")
            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)
            try:
                results["yaw"] = self.optimize_yaw()
            except Exception as e:
                logger.warning(f"Yaw optimization failed: {e}")
                print(f"  Yaw optimization pominięta: {e}")

        # Przywróć oryginalny layout
        self.farm.set_layout_custom(orig_x, orig_y, name="initial")
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _validate_ready(self) -> None:
        """Sprawdza czy optimizer jest gotowy."""
        if self._boundaries is None:
            raise RuntimeError(
                "Brak granic obszaru. Użyj set_boundaries_rectangle() "
                "lub set_boundaries_from_layout() najpierw."
            )
        if self.farm._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych w FarmModel.")
        if self.farm.n_turbines < 2:
            raise RuntimeError("Potrzebne min. 2 turbiny do optymalizacji.")

    def __repr__(self) -> str:
        bounds = "set" if self._boundaries else "not set"
        return (
            f"Optimizer(farm={self.farm.n_turbines} turbin, "
            f"boundaries={bounds}, "
            f"min_dist={self._min_dist:.0f}m)"
        )
