"""
Wind Data Loader for Wind Farm Layout Optimization Project
===========================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Obsługiwane źródła danych:
    - Mock data (rozkład Weibulla, typowy dla Bałtyku Południowego)
    - CSV (dowolny format tabelaryczny z kolumnami prędkość/kierunek)
    - ERA5 via cdsapi (dane reanalysis z Copernicus)

Zwraca natywne obiekty FLORIS:
    - floris.wind_data.TimeSeries
    - floris.wind_data.WindRose
    - floris.wind_data.WindTIRose

Autor: Temat 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FLORIS imports — wind data objects
from floris import TimeSeries, WindRose, WindTIRose

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Konfiguracja domyślna dla Bałtyku Południowego
# ---------------------------------------------------------------------------
@dataclass
class BalticWindConfig:
    """Parametry wiatrowe typowe dla Bałtyku Południowego (~54.5°N, 16.5°E).

    Źródła wartości domyślnych:
        - Weibull shape/scale: literatura dla polskiej strefy offshore
          (k ≈ 2.1, A ≈ 9.5 m/s na 150m n.p.m.)
        - Dominujący kierunek: SW–W (ok. 240°)
        - TI offshore: 0.04–0.08 (niższa niż onshore)
    """
    weibull_k: float = 2.1          # parametr kształtu
    weibull_A: float = 9.5          # parametr skali [m/s]
    dominant_direction: float = 240.0  # kierunek dominujący [°]
    direction_spread: float = 60.0    # odchylenie std kierunku [°]
    ti_mean: float = 0.06           # średnia intensywność turbulencji
    ti_std: float = 0.015           # odchylenie std TI
    hub_height: float = 150.0       # wysokość piasty [m] — IEA 15MW
    n_hours: int = 8760             # 1 rok = 8760 godzin


# ---------------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------------
class WindDataLoader:
    """Ładuje dane wiatrowe z różnych źródeł i konwertuje do formatów FLORIS.

    Użycie:
        >>> loader = WindDataLoader()
        >>> ts = loader.generate_mock_data()          # TimeSeries
        >>> wr = loader.to_wind_rose(ts)              # WindRose
        >>> wr_ti = loader.to_wind_ti_rose(ts)        # WindTIRose

        >>> loader2 = WindDataLoader.from_csv("data/raw/era5_wind.csv")
        >>> ts2 = loader2.time_series
    """

    def __init__(
        self,
        wind_speeds: Optional[np.ndarray] = None,
        wind_directions: Optional[np.ndarray] = None,
        turbulence_intensities: Optional[np.ndarray] = None,
        timestamps: Optional[pd.DatetimeIndex] = None,
    ):
        self.wind_speeds = wind_speeds
        self.wind_directions = wind_directions
        self.turbulence_intensities = turbulence_intensities
        self.timestamps = timestamps

        # Walidacja — jeśli dane podane, muszą mieć tę samą długość
        if wind_speeds is not None:
            self._validate_arrays()

    # ------------------------------------------------------------------
    # Walidacja
    # ------------------------------------------------------------------
    def _validate_arrays(self) -> None:
        """Sprawdza spójność danych wejściowych."""
        n = len(self.wind_speeds)

        if self.wind_directions is not None and len(self.wind_directions) != n:
            raise ValueError(
                f"wind_directions length ({len(self.wind_directions)}) "
                f"!= wind_speeds length ({n})"
            )
        if self.turbulence_intensities is not None and len(self.turbulence_intensities) != n:
            raise ValueError(
                f"turbulence_intensities length ({len(self.turbulence_intensities)}) "
                f"!= wind_speeds length ({n})"
            )

        # Zakres fizyczny
        if np.any(self.wind_speeds < 0):
            logger.warning("Ujemne prędkości wiatru — ustawiam na 0.")
            self.wind_speeds = np.clip(self.wind_speeds, 0, None)

        if self.wind_directions is not None:
            self.wind_directions = self.wind_directions % 360.0

        if self.turbulence_intensities is not None:
            if np.any(self.turbulence_intensities < 0):
                logger.warning("Ujemne TI — clipuję do [0.01, 0.5].")
                self.turbulence_intensities = np.clip(
                    self.turbulence_intensities, 0.01, 0.5
                )

    # ------------------------------------------------------------------
    # 1. MOCK DATA — generowanie syntetycznych danych
    # ------------------------------------------------------------------
    def generate_mock_data(
        self,
        config: Optional[BalticWindConfig] = None,
        seed: int = 42,
    ) -> TimeSeries:
        """Generuje realistyczne dane wiatrowe dla Bałtyku Południowego.

        Rozkład prędkości: Weibull (k=2.1, A=9.5 m/s)
        Rozkład kierunków: von Mises (dominujący SW–W z sezonową zmiennością)
        TI: losowy z ograniczeniami offshore (0.04–0.08)

        Args:
            config: Parametry wiatrowe. Domyślnie BalticWindConfig().
            seed: Ziarno generatora losowego (powtarzalność).

        Returns:
            floris.TimeSeries z danymi mock.
        """
        if config is None:
            config = BalticWindConfig()

        rng = np.random.default_rng(seed)
        n = config.n_hours

        # --- Prędkość wiatru: rozkład Weibulla ---
        self.wind_speeds = config.weibull_A * rng.weibull(config.weibull_k, n)
        # Ograniczenie do realistycznego zakresu
        self.wind_speeds = np.clip(self.wind_speeds, 0.0, 35.0)

        # --- Kierunek wiatru: von Mises z sezonowością ---
        # Bałtyk: zimą więcej SW/W, latem więcej NW
        hours = np.arange(n)
        day_of_year = (hours / 24.0) % 365.25

        # Sezonowe przesunięcie kierunku dominującego
        seasonal_shift = 30.0 * np.sin(2 * np.pi * day_of_year / 365.25)
        dominant_rad = np.radians(config.dominant_direction + seasonal_shift)

        # Koncentracja von Mises (kappa) — im wyższa, tym węższy rozkład
        kappa = 1.5  # umiarkowana koncentracja, realistyczna dla offshore
        raw_dirs = rng.vonmises(dominant_rad, kappa, n)
        self.wind_directions = np.degrees(raw_dirs) % 360.0

        # --- Intensywność turbulencji ---
        # TI offshore jest niższa niż onshore i maleje z prędkością wiatru
        base_ti = config.ti_mean - 0.002 * (self.wind_speeds - 10.0)
        noise = rng.normal(0, config.ti_std, n)
        self.turbulence_intensities = np.clip(base_ti + noise, 0.02, 0.20)

        # --- Timestamps ---
        self.timestamps = pd.date_range(
            start="2020-01-01", periods=n, freq="h"
        )

        logger.info(
            f"Mock data: {n} godzin, "
            f"śr. prędkość={self.wind_speeds.mean():.1f} m/s, "
            f"śr. TI={self.turbulence_intensities.mean():.3f}"
        )

        return self.to_time_series()

    # ------------------------------------------------------------------
    # 2. CSV LOADER — uniwersalny import
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        speed_col: str = "wind_speed",
        direction_col: str = "wind_direction",
        ti_col: Optional[str] = "turbulence_intensity",
        time_col: Optional[str] = "timestamp",
        separator: str = ",",
    ) -> "WindDataLoader":
        """Ładuje dane z pliku CSV.

        Obsługuje różne formaty — wystarczy podać nazwy kolumn.
        Działa z danymi ERA5, IMGW, lub dowolnymi innymi.

        Args:
            filepath: Ścieżka do pliku CSV.
            speed_col: Nazwa kolumny z prędkością wiatru [m/s].
            direction_col: Nazwa kolumny z kierunkiem wiatru [°].
            ti_col: Nazwa kolumny z TI (opcjonalne).
            time_col: Nazwa kolumny z timestampem (opcjonalne).
            separator: Separator CSV.

        Returns:
            WindDataLoader z załadowanymi danymi.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {filepath}")

        df = pd.read_csv(filepath, sep=separator)
        logger.info(f"Załadowano CSV: {filepath.name} ({len(df)} wierszy)")

        # Wymagane kolumny
        for col_name, col_label in [(speed_col, "prędkość"), (direction_col, "kierunek")]:
            if col_name not in df.columns:
                available = ", ".join(df.columns.tolist())
                raise KeyError(
                    f"Brak kolumny '{col_name}' ({col_label} wiatru). "
                    f"Dostępne kolumny: {available}"
                )

        wind_speeds = df[speed_col].values.astype(float)
        wind_directions = df[direction_col].values.astype(float)

        # Opcjonalne: TI
        ti = None
        if ti_col and ti_col in df.columns:
            ti = df[ti_col].values.astype(float)
        else:
            # Estymacja TI na podstawie prędkości (model IEC offshore)
            ti = 0.06 * np.ones_like(wind_speeds)
            logger.info("Brak kolumny TI — ustawiono domyślne 0.06 (offshore).")

        # Opcjonalne: timestamp
        timestamps = None
        if time_col and time_col in df.columns:
            timestamps = pd.to_datetime(df[time_col])

        return cls(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            turbulence_intensities=ti,
            timestamps=timestamps,
        )

    # ------------------------------------------------------------------
    # 3. ERA5 LOADER — Copernicus Climate Data Store
    # ------------------------------------------------------------------
    @classmethod
    def from_era5(
        cls,
        latitude: float = 54.5,
        longitude: float = 16.5,
        years: list[int] = None,
        hub_height: float = 150.0,
        output_dir: Union[str, Path] = "data/raw",
    ) -> "WindDataLoader":
        """Pobiera dane ERA5 z Copernicus CDS via cdsapi.

        ERA5 dostarcza dane wiatrowe na 10m i 100m n.p.m.
        Ekstrapolacja do hub_height za pomocą profilu logarytmicznego.

        UWAGA: Wymaga skonfigurowanego ~/.cdsapirc z kluczem API.

        Args:
            latitude: Szerokość geograficzna.
            longitude: Długość geograficzna.
            years: Lista lat do pobrania. Domyślnie [2015–2024].
            hub_height: Wysokość piasty do ekstrapolacji [m].
            output_dir: Katalog na pobrane pliki.

        Returns:
            WindDataLoader z danymi ERA5.
        """
        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "Zainstaluj cdsapi: pip install cdsapi\n"
                "Skonfiguruj klucz API: https://cds.climate.copernicus.eu/api-how-to"
            )

        if years is None:
            years = list(range(2015, 2025))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        client = cdsapi.Client()
        all_data = []

        for year in years:
            output_file = output_dir / f"era5_{year}_{latitude}_{longitude}.nc"

            if output_file.exists():
                logger.info(f"ERA5 {year} — plik istnieje, pomijam pobieranie.")
            else:
                logger.info(f"Pobieram ERA5 {year}...")
                client.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": [
                            "100m_u_component_of_wind",
                            "100m_v_component_of_wind",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind",
                        ],
                        "year": str(year),
                        "month": [f"{m:02d}" for m in range(1, 13)],
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": [f"{h:02d}:00" for h in range(24)],
                        "area": [
                            latitude + 0.1,
                            longitude - 0.1,
                            latitude - 0.1,
                            longitude + 0.1,
                        ],
                        "format": "netcdf",
                    },
                    str(output_file),
                )

            # Parsowanie NetCDF
            year_data = cls._parse_era5_netcdf(output_file, hub_height)
            all_data.append(year_data)

        combined = pd.concat(all_data, ignore_index=True)

        return cls(
            wind_speeds=combined["wind_speed"].values,
            wind_directions=combined["wind_direction"].values,
            turbulence_intensities=combined["ti"].values,
            timestamps=pd.to_datetime(combined["timestamp"]),
        )

    @staticmethod
    def _parse_era5_netcdf(filepath: Path, hub_height: float) -> pd.DataFrame:
        """Parsuje plik NetCDF ERA5 i ekstrapoluje wiatr do hub_height.

        Ekstrapolacja: profil logarytmiczny
            U(h) = U(h_ref) * ln(h / z0) / ln(h_ref / z0)
            z0 ≈ 0.0002 m (chropowatość morza)
        """
        import xarray as xr

        ds = xr.open_dataset(filepath)

        # Składowe wiatru na 100m
        u100 = ds["u100"].values.flatten()
        v100 = ds["v100"].values.flatten()

        # Prędkość i kierunek na 100m
        ws_100 = np.sqrt(u100**2 + v100**2)
        wd_100 = (270.0 - np.degrees(np.arctan2(v100, u100))) % 360.0

        # Ekstrapolacja logarytmiczna do hub_height
        z0 = 0.0002  # chropowatość morza [m]
        z_ref = 100.0  # wysokość referencyjna ERA5
        log_ratio = np.log(hub_height / z0) / np.log(z_ref / z0)
        ws_hub = ws_100 * log_ratio

        # TI — estymacja na podstawie u10/u100 jako proxy turbulencji
        u10 = ds["u10"].values.flatten()
        v10 = ds["v10"].values.flatten()
        ws_10 = np.sqrt(u10**2 + v10**2)

        # Prosty model: TI ∝ wind shear
        shear = np.where(ws_100 > 1.0, ws_10 / ws_100, 1.0)
        ti = np.clip(0.12 * shear, 0.02, 0.20)

        # Timestamps
        timestamps = pd.to_datetime(ds["time"].values)

        ds.close()

        return pd.DataFrame({
            "timestamp": timestamps[:len(ws_hub)],
            "wind_speed": ws_hub,
            "wind_direction": wd_100,
            "ti": ti,
        })

    # ------------------------------------------------------------------
    # Konwersja do obiektów FLORIS
    # ------------------------------------------------------------------
    def to_time_series(
        self,
        values: Optional[np.ndarray] = None,
    ) -> TimeSeries:
        """Konwertuje dane do FLORIS TimeSeries.

        Args:
            values: Opcjonalna tablica wartości energii [$/MWh]
                    do optymalizacji AVP zamiast AEP.

        Returns:
            floris.TimeSeries
        """
        self._ensure_data_loaded()

        kwargs = dict(
            wind_directions=self.wind_directions,
            wind_speeds=self.wind_speeds,
            turbulence_intensities=self.turbulence_intensities,
        )
        if values is not None:
            kwargs["values"] = values

        return TimeSeries(**kwargs)

    def to_wind_rose(
        self,
        wd_step: float = 10.0,
        ws_step: float = 2.0,
        ws_min: float = 3.0,
        ws_max: float = 25.0,
    ) -> WindRose:
        """Konwertuje dane do FLORIS WindRose (binned).

        Tworzy róże wiatrów z tabel częstotliwości — optymalne do AEP.

        Args:
            wd_step: Krok binowania kierunku [°].
            ws_step: Krok binowania prędkości [m/s].
            ws_min: Minimalna prędkość w binach [m/s].
            ws_max: Maksymalna prędkość w binach [m/s].

        Returns:
            floris.WindRose
        """
        self._ensure_data_loaded()

        ts = self.to_time_series()
        wind_rose = ts.to_WindRose(
            wd_edges=np.arange(0, 360 + wd_step, wd_step),
            ws_edges=np.arange(ws_min, ws_max + ws_step, ws_step),
        )
        return wind_rose

    def to_wind_ti_rose(
        self,
        wd_step: float = 10.0,
        ws_step: float = 2.0,
        ti_step: float = 0.02,
    ) -> WindTIRose:
        """Konwertuje dane do FLORIS WindTIRose (binned z TI).

        Zachowuje zmienność TI per bin — dokładniejsze niż WindRose
        z pojedynczą wartością TI.

        Args:
            wd_step: Krok binowania kierunku [°].
            ws_step: Krok binowania prędkości [m/s].
            ti_step: Krok binowania TI.

        Returns:
            floris.WindTIRose
        """
        self._ensure_data_loaded()

        ts = self.to_time_series()
        wind_ti_rose = ts.to_WindTIRose(
            wd_edges=np.arange(0, 360 + wd_step, wd_step),
            ws_edges=np.arange(3.0, 25.0 + ws_step, ws_step),
            ti_edges=np.arange(0.01, 0.20 + ti_step, ti_step),
        )
        return wind_ti_rose

    # ------------------------------------------------------------------
    # Analityka i statystyki
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """Zwraca słownik ze statystykami danych wiatrowych."""
        self._ensure_data_loaded()

        stats = {
            "n_records": len(self.wind_speeds),
            "wind_speed": {
                "mean": float(np.mean(self.wind_speeds)),
                "std": float(np.std(self.wind_speeds)),
                "median": float(np.median(self.wind_speeds)),
                "min": float(np.min(self.wind_speeds)),
                "max": float(np.max(self.wind_speeds)),
                "weibull_k_est": self._estimate_weibull_k(),
                "weibull_A_est": self._estimate_weibull_A(),
            },
            "wind_direction": {
                "dominant": float(self._dominant_direction()),
                "mean_circular": float(self._circular_mean_direction()),
            },
            "turbulence_intensity": {
                "mean": float(np.mean(self.turbulence_intensities)),
                "std": float(np.std(self.turbulence_intensities)),
            },
        }

        if self.timestamps is not None:
            stats["time_range"] = {
                "start": str(self.timestamps.min()),
                "end": str(self.timestamps.max()),
                "duration_days": (self.timestamps.max() - self.timestamps.min()).days,
            }

        return stats

    def _estimate_weibull_k(self) -> float:
        """Estymacja parametru kształtu Weibulla (metoda momentów)."""
        mean_ws = np.mean(self.wind_speeds)
        std_ws = np.std(self.wind_speeds)
        if std_ws == 0:
            return 0.0
        # Przybliżenie: k ≈ (σ/μ)^(-1.086)
        return (std_ws / mean_ws) ** (-1.086)

    def _estimate_weibull_A(self) -> float:
        """Estymacja parametru skali Weibulla."""
        from scipy.special import gamma
        k = self._estimate_weibull_k()
        if k <= 0:
            return 0.0
        return float(np.mean(self.wind_speeds) / gamma(1 + 1 / k))

    def _dominant_direction(self) -> float:
        """Znajduje dominujący kierunek wiatru (moda binowana)."""
        bins = np.arange(0, 370, 10)
        hist, _ = np.histogram(self.wind_directions, bins=bins)
        dominant_bin = np.argmax(hist)
        return float(bins[dominant_bin] + 5.0)  # środek binu

    def _circular_mean_direction(self) -> float:
        """Średni kierunek kołowy (poprawna statystyka dla kątów)."""
        rad = np.radians(self.wind_directions)
        mean_sin = np.mean(np.sin(rad))
        mean_cos = np.mean(np.cos(rad))
        return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)

    # ------------------------------------------------------------------
    # Wizualizacja
    # ------------------------------------------------------------------
    def plot_wind_rose(
        self,
        ax: Optional[plt.Axes] = None,
        n_sectors: int = 36,
        n_speed_bins: int = 6,
        title: str = "Róża wiatrów",
        figsize: tuple = (8, 8),
    ) -> plt.Figure:
        """Rysuje różę wiatrów w matplotlib.

        Args:
            ax: Oś matplotlib (polar). Jeśli None, tworzy nową figurę.
            n_sectors: Liczba sektorów kierunku.
            n_speed_bins: Liczba przedziałów prędkości.
            title: Tytuł wykresu.
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        self._ensure_data_loaded()

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="polar")
        else:
            fig = ax.get_figure()

        # Binowanie
        dir_bins = np.linspace(0, 360, n_sectors + 1)
        speed_bins = np.linspace(0, np.percentile(self.wind_speeds, 98), n_speed_bins + 1)
        colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, n_speed_bins))

        dir_centers = np.radians((dir_bins[:-1] + dir_bins[1:]) / 2)
        bar_width = np.radians(360 / n_sectors) * 0.85

        # Rysowanie słupków
        bottom = np.zeros(n_sectors)
        for i in range(n_speed_bins):
            mask_speed = (self.wind_speeds >= speed_bins[i]) & (
                self.wind_speeds < speed_bins[i + 1]
            )
            counts = np.zeros(n_sectors)
            for j in range(n_sectors):
                mask_dir = (self.wind_directions >= dir_bins[j]) & (
                    self.wind_directions < dir_bins[j + 1]
                )
                counts[j] = np.sum(mask_speed & mask_dir)

            freq = counts / len(self.wind_speeds) * 100  # procent
            label = f"{speed_bins[i]:.0f}–{speed_bins[i+1]:.0f} m/s"
            ax.bar(
                dir_centers, freq, width=bar_width, bottom=bottom,
                color=colors[i], edgecolor="white", linewidth=0.3,
                label=label, zorder=3,
            )
            bottom += freq

        # Formatowanie
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(title, pad=20, fontsize=14, fontweight="500")
        ax.legend(
            loc="lower left", bbox_to_anchor=(-0.15, -0.15),
            fontsize=9, ncol=3, framealpha=0.9,
        )

        return fig

    def plot_time_series(
        self,
        figsize: tuple = (14, 8),
        title: str = "Dane wiatrowe — seria czasowa",
    ) -> plt.Figure:
        """Rysuje przegląd danych wiatrowych (4 panele).

        Panele:
            1. Prędkość wiatru vs czas
            2. Kierunek wiatru vs czas
            3. Histogram prędkości + fit Weibulla
            4. TI vs prędkość wiatru
        """
        self._ensure_data_loaded()

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight="500", y=1.02)

        x_axis = self.timestamps if self.timestamps is not None else np.arange(len(self.wind_speeds))

        # Panel 1: Prędkość wiatru
        ax = axes[0, 0]
        ax.plot(x_axis, self.wind_speeds, linewidth=0.3, alpha=0.6, color="#1e5c3a")
        # Średnia krocząca 24h
        ws_series = pd.Series(self.wind_speeds)
        rolling_mean = ws_series.rolling(window=24, center=True).mean()
        ax.plot(x_axis, rolling_mean, linewidth=1.2, color="#c8531a", label="Śr. 24h")
        ax.set_ylabel("Prędkość [m/s]")
        ax.set_title("Prędkość wiatru")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 2: Kierunek wiatru
        ax = axes[0, 1]
        ax.scatter(
            x_axis, self.wind_directions,
            s=0.3, alpha=0.3, color="#534AB7", rasterized=True,
        )
        ax.set_ylabel("Kierunek [°]")
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_yticklabels(["N", "E", "S", "W", "N"])
        ax.set_title("Kierunek wiatru")
        ax.grid(True, alpha=0.3)

        # Panel 3: Histogram + Weibull
        ax = axes[1, 0]
        ax.hist(
            self.wind_speeds, bins=50, density=True,
            alpha=0.6, color="#1e5c3a", edgecolor="white", linewidth=0.3,
        )
        # Fit Weibulla
        from scipy.stats import weibull_min
        params = weibull_min.fit(self.wind_speeds[self.wind_speeds > 0], floc=0)
        x_fit = np.linspace(0, self.wind_speeds.max(), 200)
        ax.plot(
            x_fit, weibull_min.pdf(x_fit, *params),
            linewidth=2, color="#c8531a",
            label=f"Weibull (k={params[0]:.2f}, A={params[2]:.1f})",
        )
        ax.set_xlabel("Prędkość [m/s]")
        ax.set_ylabel("Gęstość")
        ax.set_title("Rozkład prędkości")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Panel 4: TI vs prędkość
        ax = axes[1, 1]
        ax.scatter(
            self.wind_speeds, self.turbulence_intensities,
            s=0.5, alpha=0.3, color="#534AB7", rasterized=True,
        )
        # Średnia TI per bin
        ws_bins = np.arange(0, 26, 1)
        ti_means = []
        ws_centers = []
        for i in range(len(ws_bins) - 1):
            mask = (self.wind_speeds >= ws_bins[i]) & (self.wind_speeds < ws_bins[i + 1])
            if np.sum(mask) > 10:
                ti_means.append(np.mean(self.turbulence_intensities[mask]))
                ws_centers.append((ws_bins[i] + ws_bins[i + 1]) / 2)
        ax.plot(ws_centers, ti_means, "o-", color="#c8531a", markersize=4, label="Śr. TI per bin")
        ax.set_xlabel("Prędkość [m/s]")
        ax.set_ylabel("TI [-]")
        ax.set_title("Intensywność turbulencji")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Eksport
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Eksportuje dane jako pandas DataFrame."""
        self._ensure_data_loaded()

        data = {
            "wind_speed": self.wind_speeds,
            "wind_direction": self.wind_directions,
            "turbulence_intensity": self.turbulence_intensities,
        }
        if self.timestamps is not None:
            data["timestamp"] = self.timestamps

        return pd.DataFrame(data)

    def to_csv(self, filepath: Union[str, Path]) -> None:
        """Zapisuje dane do CSV."""
        df = self.to_dataframe()
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Zapisano dane do: {filepath}")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _ensure_data_loaded(self) -> None:
        """Sprawdza czy dane są załadowane."""
        if self.wind_speeds is None:
            raise RuntimeError(
                "Brak danych wiatrowych. Użyj generate_mock_data(), "
                "from_csv() lub from_era5() najpierw."
            )

    def filter_by_speed(self, min_speed: float = 3.0, max_speed: float = 25.0) -> "WindDataLoader":
        """Filtruje dane po zakresie prędkości (cut-in / cut-out turbiny)."""
        self._ensure_data_loaded()
        mask = (self.wind_speeds >= min_speed) & (self.wind_speeds <= max_speed)
        return WindDataLoader(
            wind_speeds=self.wind_speeds[mask],
            wind_directions=self.wind_directions[mask],
            turbulence_intensities=self.turbulence_intensities[mask],
            timestamps=self.timestamps[mask] if self.timestamps is not None else None,
        )

    def seasonal_split(self) -> dict[str, "WindDataLoader"]:
        """Dzieli dane na sezony (DJF, MAM, JJA, SON).

        Przydatne do analizy sezonowej AEP.
        """
        if self.timestamps is None:
            raise RuntimeError("Brak timestamps — nie można podzielić na sezony.")

        months = self.timestamps.month
        seasons = {
            "DJF (zima)": months.isin([12, 1, 2]),
            "MAM (wiosna)": months.isin([3, 4, 5]),
            "JJA (lato)": months.isin([6, 7, 8]),
            "SON (jesień)": months.isin([9, 10, 11]),
        }

        result = {}
        for name, mask in seasons.items():
            result[name] = WindDataLoader(
                wind_speeds=self.wind_speeds[mask],
                wind_directions=self.wind_directions[mask],
                turbulence_intensities=self.turbulence_intensities[mask],
                timestamps=self.timestamps[mask],
            )

        return result

    def __repr__(self) -> str:
        if self.wind_speeds is None:
            return "WindDataLoader(empty)"
        return (
            f"WindDataLoader("
            f"n={len(self.wind_speeds)}, "
            f"ws_mean={np.mean(self.wind_speeds):.1f} m/s, "
            f"dominant_dir={self._dominant_direction():.0f}°)"
        )
