"""
AEP Calculator for Wind Farm Layout Optimization Project
==========================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Szczegółowa analiza produkcji energii:
    - AEP per turbina i per warunek wiatrowy
    - Rozbicie sezonowe (DJF/MAM/JJA/SON) i miesięczne
    - Analiza niepewności (UncertainFlorisModel)
    - Porównanie scenariuszy (różne layouty, turbiny, dane wiatrowe)
    - Eksport aep_timeseries.csv dla Tematu 5

Autor: Temat 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from floris import FlorisModel, UncertainFlorisModel, WindRose, TimeSeries

from .farm_model import FarmModel
from .wind_data import WindDataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------------
class AEPCalculator:
    """Kalkulator AEP z rozbiciem czasowym i analizą niepewności.

    Użycie:
        >>> calc = AEPCalculator(farm)
        >>> summary = calc.compute_aep_summary()
        >>> seasonal = calc.compute_seasonal_aep(loader)
        >>> calc.plot_aep_breakdown()
        >>> calc.export_for_team5("outputs/exports/aep_timeseries.csv")
    """

    def __init__(self, farm: FarmModel):
        """
        Args:
            farm: FarmModel z ustawionym layoutem i danymi wiatrowymi.
        """
        self.farm = farm
        self._last_summary = None

    # ------------------------------------------------------------------
    # Podstawowe AEP
    # ------------------------------------------------------------------
    def compute_aep_summary(self) -> dict:
        """Oblicza podsumowanie AEP farmy.

        Returns:
            Dict ze statystykami AEP.
        """
        if self.farm._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych.")

        self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()

        farm_aep = self.farm.get_aep_gwh()
        turbine_powers = self.farm.get_turbine_powers_mw()

        # AEP per turbina — z nansum po warunkach wiatrowych
        # Wymiar powers: (n_wind_conditions, n_turbines)
        mean_power_per_turbine = np.nanmean(turbine_powers, axis=0)
        total_power_per_condition = np.nansum(turbine_powers, axis=-1)

        # Capacity factor
        rated_power = self.farm.turbine_info["rated_power"]
        capacity_factor = farm_aep / (
            rated_power * self.farm.n_turbines * 8.76
        ) * 100

        # Wake losses
        wake_losses = self.farm.get_wake_losses_percent()

        summary = {
            "farm_aep_gwh": farm_aep,
            "n_turbines": self.farm.n_turbines,
            "turbine": self.farm.turbine_info["name"],
            "wake_model": self.farm.wake_model_name,
            "capacity_factor_pct": capacity_factor,
            "wake_losses_pct": wake_losses,
            "rated_power_mw": rated_power,
            "total_capacity_mw": rated_power * self.farm.n_turbines,
            "mean_power_per_turbine_mw": float(np.nanmean(mean_power_per_turbine)),
        }

        self._last_summary = summary
        return summary

    # ------------------------------------------------------------------
    # Analiza sezonowa
    # ------------------------------------------------------------------
    def compute_seasonal_aep(
        self,
        loader: WindDataLoader,
    ) -> pd.DataFrame:
        """Oblicza AEP rozbite na sezony.

        Wymaga WindDataLoader z timestampami.

        Args:
            loader: WindDataLoader z danymi wiatrowymi.

        Returns:
            DataFrame z AEP per sezon.
        """
        seasons = loader.seasonal_split()
        results = []

        for season_name, season_loader in seasons.items():
            # Przelicz na WindRose
            wr = season_loader.to_wind_rose()

            # Ustaw i oblicz
            self.farm.fmodel.set(wind_data=wr)
            self.farm.run()

            n_hours = len(season_loader.wind_speeds)
            aep_season = self.farm.get_aep_gwh()

            # Przelicz na proporcję roku
            fraction = n_hours / 8760.0
            stats = season_loader.summary()

            results.append({
                "season": season_name,
                "n_hours": n_hours,
                "fraction": fraction,
                "aep_gwh": aep_season,
                "mean_wind_speed_ms": stats["wind_speed"]["mean"],
                "dominant_direction_deg": stats["wind_direction"]["dominant"],
                "mean_ti": stats["turbulence_intensity"]["mean"],
            })

        # Przywróć pełne dane
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Analiza miesięczna
    # ------------------------------------------------------------------
    def compute_monthly_aep(
        self,
        loader: WindDataLoader,
    ) -> pd.DataFrame:
        """Oblicza AEP rozbite na miesiące.

        Args:
            loader: WindDataLoader z timestampami.

        Returns:
            DataFrame z AEP per miesiąc.
        """
        if loader.timestamps is None:
            raise RuntimeError("Brak timestamps w WindDataLoader.")

        months = loader.timestamps.month
        results = []

        for m in range(1, 13):
            mask = months == m
            if mask.sum() < 24:  # min. 1 dzień danych
                continue

            month_loader = WindDataLoader(
                wind_speeds=loader.wind_speeds[mask],
                wind_directions=loader.wind_directions[mask],
                turbulence_intensities=loader.turbulence_intensities[mask],
                timestamps=loader.timestamps[mask],
            )

            wr = month_loader.to_wind_rose()
            self.farm.fmodel.set(wind_data=wr)
            self.farm.run()

            aep = self.farm.get_aep_gwh()
            stats = month_loader.summary()

            month_names = [
                "", "Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec",
                "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień",
            ]

            results.append({
                "month": m,
                "month_name": month_names[m],
                "n_hours": int(mask.sum()),
                "aep_gwh": aep,
                "mean_wind_speed_ms": stats["wind_speed"]["mean"],
                "dominant_direction_deg": stats["wind_direction"]["dominant"],
            })

        # Przywróć pełne dane
        if self.farm._wind_data is not None:
            self.farm.fmodel.set(wind_data=self.farm._wind_data)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Analiza niepewności (UncertainFlorisModel)
    # ------------------------------------------------------------------
    def compute_uncertain_aep(
        self,
        wd_std: float = 3.0,
        n_samples: int = 5,
    ) -> dict:
        """Oblicza AEP z uwzględnieniem niepewności kierunku wiatru.

        Używa UncertainFlorisModel — rozmycie gaussowskie po kierunku.
        Symuluje efekt niepewności danych wiatrowych na AEP.

        Args:
            wd_std: Odchylenie standardowe kierunku wiatru [°].
            n_samples: Liczba próbek w rozmyciu (per stronę).

        Returns:
            Dict z AEP deterministic vs uncertain.
        """
        if self.farm._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych.")

        # AEP deterministyczne
        self.farm.fmodel.set(wind_data=self.farm._wind_data)
        self.farm.run()
        aep_deterministic = self.farm.get_aep_gwh()

        # AEP z niepewnością
        sample_points = list(range(-n_samples, n_samples + 1))

        ufmodel = UncertainFlorisModel(
            self.farm.fmodel,
            wd_std=wd_std,
            wd_sample_points=sample_points,
        )
        ufmodel.set(wind_data=self.farm._wind_data)
        ufmodel.run()
        aep_uncertain = ufmodel.get_farm_AEP() / 1e9

        difference_pct = (aep_uncertain - aep_deterministic) / aep_deterministic * 100

        return {
            "aep_deterministic_gwh": aep_deterministic,
            "aep_uncertain_gwh": aep_uncertain,
            "wd_std_deg": wd_std,
            "difference_pct": difference_pct,
        }

    def uncertainty_sweep(
        self,
        wd_stds: list[float] = None,
    ) -> pd.DataFrame:
        """Testuje wpływ różnych poziomów niepewności na AEP.

        Args:
            wd_stds: Lista odchyleń std kierunku wiatru [°].

        Returns:
            DataFrame z AEP per poziom niepewności.
        """
        if wd_stds is None:
            wd_stds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

        results = []
        for wd_std in wd_stds:
            r = self.compute_uncertain_aep(wd_std=wd_std)
            results.append(r)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Porównanie scenariuszy
    # ------------------------------------------------------------------
    def compare_scenarios(
        self,
        scenarios: dict[str, dict],
    ) -> pd.DataFrame:
        """Porównuje różne scenariusze konfiguracji farmy.

        Args:
            scenarios: Dict {nazwa: {wake_model, turbine, spacing_D, ...}}.
                Dostępne klucze: wake_model, turbine, spacing_D,
                n_rows, n_cols, layout_type.

        Returns:
            DataFrame z wynikami per scenariusz.

        Przykład:
            >>> scenarios = {
            ...     "Base": {"wake_model": "gch", "turbine": "iea_15MW", "spacing_D": 7.0},
            ...     "Dense": {"wake_model": "gch", "turbine": "iea_15MW", "spacing_D": 5.0},
            ...     "Sparse": {"wake_model": "gch", "turbine": "iea_15MW", "spacing_D": 10.0},
            ... }
            >>> df = calc.compare_scenarios(scenarios)
        """
        results = []
        orig_wake = self.farm.wake_model_name
        orig_turbine = self.farm.turbine_name

        for name, config in scenarios.items():
            # Zmień konfigurację
            wake = config.get("wake_model", orig_wake)
            turbine = config.get("turbine", orig_turbine)
            spacing = config.get("spacing_D", 7.0)
            n_rows = config.get("n_rows", 5)
            n_cols = config.get("n_cols", 5)
            layout_type = config.get("layout_type", "grid")

            self.farm.switch_wake_model(wake)
            self.farm.switch_turbine(turbine)

            if layout_type == "grid":
                self.farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing)
            elif layout_type == "staggered":
                self.farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing)

            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)
            self.farm.run()

            aep = self.farm.get_aep_gwh()
            wake_loss = self.farm.get_wake_losses_percent()
            rated = self.farm.turbine_info["rated_power"]
            cf = aep / (rated * self.farm.n_turbines * 8.76) * 100

            results.append({
                "scenario": name,
                "wake_model": wake,
                "turbine": self.farm.turbine_info["name"],
                "n_turbines": self.farm.n_turbines,
                "spacing_D": spacing,
                "layout_type": layout_type,
                "aep_gwh": aep,
                "wake_loss_pct": wake_loss,
                "capacity_factor_pct": cf,
                "total_capacity_mw": rated * self.farm.n_turbines,
            })

        # Przywróć
        self.farm.switch_wake_model(orig_wake)
        self.farm.switch_turbine(orig_turbine)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Wizualizacja
    # ------------------------------------------------------------------
    def plot_aep_breakdown(
        self,
        loader: WindDataLoader,
        figsize: tuple = (16, 10),
    ) -> plt.Figure:
        """Rysuje przegląd AEP: roczne, sezonowe, miesięczne.

        Args:
            loader: WindDataLoader z timestamps.
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        seasonal = self.compute_seasonal_aep(loader)
        monthly = self.compute_monthly_aep(loader)
        summary = self.compute_aep_summary()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Panel 1: AEP roczne — podsumowanie
        ax = axes[0, 0]
        labels = ["AEP\n[GWh]", "Capacity\nFactor [%]", "Wake\nLosses [%]"]
        values = [
            summary["farm_aep_gwh"],
            summary["capacity_factor_pct"],
            summary["wake_losses_pct"],
        ]
        colors = ["#1e5c3a", "#534AB7", "#c8531a"]
        bars = ax.bar(labels, values, color=colors, width=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=11, fontweight="500",
            )
        ax.set_title(
            f"Podsumowanie roczne — {summary['turbine']} | "
            f"{summary['n_turbines']} turbin | {summary['wake_model'].upper()}"
        )
        ax.set_ylabel("Wartość")
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: AEP sezonowe
        ax = axes[0, 1]
        season_colors = ["#3B8BD4", "#4CAF50", "#FF9800", "#c8531a"]
        ax.bar(
            seasonal["season"], seasonal["aep_gwh"],
            color=season_colors, edgecolor="white", linewidth=1,
        )
        for i, row in seasonal.iterrows():
            ax.text(
                i, row["aep_gwh"] + 0.5,
                f"{row['aep_gwh']:.1f}\n({row['mean_wind_speed_ms']:.1f} m/s)",
                ha="center", fontsize=9,
            )
        ax.set_title("AEP sezonowe")
        ax.set_ylabel("AEP [GWh]")
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 3: AEP miesięczne
        ax = axes[1, 0]
        ax.bar(
            monthly["month_name"], monthly["aep_gwh"],
            color="#1e5c3a", alpha=0.8, edgecolor="white", linewidth=0.5,
        )
        ax.set_title("AEP miesięczne")
        ax.set_ylabel("AEP [GWh]")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 4: Prędkość wiatru miesięczna
        ax = axes[1, 1]
        ax.plot(
            monthly["month_name"], monthly["mean_wind_speed_ms"],
            "o-", color="#534AB7", linewidth=2, markersize=6,
        )
        ax.fill_between(
            monthly["month_name"], monthly["mean_wind_speed_ms"],
            alpha=0.15, color="#534AB7",
        )
        ax.set_title("Średnia prędkość wiatru miesięczna")
        ax.set_ylabel("Prędkość [m/s]")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Analiza AEP — {summary['turbine']} | "
            f"{summary['total_capacity_mw']:.0f} MW total",
            fontsize=14, fontweight="500",
        )
        fig.tight_layout()
        return fig

    def plot_scenario_comparison(
        self,
        df: pd.DataFrame,
        figsize: tuple = (14, 5),
    ) -> plt.Figure:
        """Rysuje porównanie scenariuszy.

        Args:
            df: DataFrame z compare_scenarios().
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

        # AEP
        axes[0].barh(df["scenario"], df["aep_gwh"], color=colors)
        axes[0].set_xlabel("AEP [GWh]")
        axes[0].set_title("Roczna produkcja energii")
        axes[0].grid(True, alpha=0.3, axis="x")

        # Wake losses
        axes[1].barh(df["scenario"], df["wake_loss_pct"], color=colors)
        axes[1].set_xlabel("Wake losses [%]")
        axes[1].set_title("Straty wake")
        axes[1].grid(True, alpha=0.3, axis="x")

        # Capacity factor
        axes[2].barh(df["scenario"], df["capacity_factor_pct"], color=colors)
        axes[2].set_xlabel("Capacity factor [%]")
        axes[2].set_title("Współczynnik wykorzystania")
        axes[2].grid(True, alpha=0.3, axis="x")

        fig.suptitle("Porównanie scenariuszy", fontsize=14, fontweight="500")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Eksport dla Tematu 5
    # ------------------------------------------------------------------
    def export_for_team5(
        self,
        filepath: Union[str, Path] = "outputs/exports/aep_timeseries.csv",
        loader: Optional[WindDataLoader] = None,
    ) -> pd.DataFrame:
        """Eksportuje dane AEP dla Tematu 5 (analiza ekonomiczna).

        Format: aep_timeseries.csv z rozbiciem miesięcznym i rocznym.

        Args:
            filepath: Ścieżka do pliku.
            loader: WindDataLoader (potrzebny do rozbicia miesięcznego).

        Returns:
            DataFrame z wyeksportowanymi danymi.
        """
        summary = self.compute_aep_summary()

        rows = []

        # Roczne
        rows.append({
            "period": "annual",
            "period_label": "Roczne",
            "aep_gwh": summary["farm_aep_gwh"],
            "capacity_factor_pct": summary["capacity_factor_pct"],
            "wake_losses_pct": summary["wake_losses_pct"],
            "n_turbines": summary["n_turbines"],
            "turbine_type": summary["turbine"],
            "wake_model": summary["wake_model"],
            "total_capacity_mw": summary["total_capacity_mw"],
        })

        # Miesięczne (jeśli mamy loader)
        if loader is not None and loader.timestamps is not None:
            monthly = self.compute_monthly_aep(loader)
            for _, row in monthly.iterrows():
                rows.append({
                    "period": f"month_{row['month']:02d}",
                    "period_label": row["month_name"],
                    "aep_gwh": row["aep_gwh"],
                    "capacity_factor_pct": None,
                    "wake_losses_pct": None,
                    "n_turbines": summary["n_turbines"],
                    "turbine_type": summary["turbine"],
                    "wake_model": summary["wake_model"],
                    "total_capacity_mw": summary["total_capacity_mw"],
                })

            # Sezonowe
            seasonal = self.compute_seasonal_aep(loader)
            for _, row in seasonal.iterrows():
                rows.append({
                    "period": f"season_{row['season'][:3]}",
                    "period_label": row["season"],
                    "aep_gwh": row["aep_gwh"],
                    "capacity_factor_pct": None,
                    "wake_losses_pct": None,
                    "n_turbines": summary["n_turbines"],
                    "turbine_type": summary["turbine"],
                    "wake_model": summary["wake_model"],
                    "total_capacity_mw": summary["total_capacity_mw"],
                })

        df = pd.DataFrame(rows)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"AEP wyeksportowane do: {filepath}")

        return df

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"AEPCalculator(farm={self.farm.n_turbines} turbin, "
            f"turbine={self.farm.turbine_info['name']})"
        )
