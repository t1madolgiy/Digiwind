"""
Farm Model for Wind Farm Layout Optimization Project
=====================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Owija FlorisModel i daje:
    - Łatwe przełączanie modeli wake (Jensen, GCH, TurbOPark, Empirical Gauss, CC)
    - Wybór turbin z wbudowanej biblioteki (NREL 5MW, IEA 10MW/15MW/22MW)
    - Generatory layoutów (grid, staggered, circular, custom)
    - Wizualizację flow field (horizontal, cross-section)
    - Porównanie konfiguracji

Autor: Temat 2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

import floris
import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz
from floris import FlorisModel, WindRose, TimeSeries
from floris.utilities import load_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stałe — dostępne modele i turbiny
# ---------------------------------------------------------------------------

# Ścieżka do domyślnego YAML-a FLORIS
_FLORIS_DIR = Path(floris.__file__).parent
_DEFAULT_YAML = _FLORIS_DIR / "default_inputs.yaml"

# Mapowanie nazw modeli wake na parametry FLORIS
WAKE_MODELS = {
    "jensen": {
        "velocity_model": "jensen",
        "deflection_model": "jimenez",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
    },
    "gch": {
        "velocity_model": "gauss",
        "deflection_model": "gauss",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
    },
    "turbopark": {
        "velocity_model": "turboparkgauss",
        "deflection_model": "none",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
    },
    "empirical_gauss": {
        "velocity_model": "empirical_gauss",
        "deflection_model": "empirical_gauss",
        "turbulence_model": "wake_induced_mixing",
        "combination_model": "sosfs",
    },
    "cc": {
        "velocity_model": "cc",
        "deflection_model": "gauss",
        "turbulence_model": "crespo_hernandez",
        "combination_model": "sosfs",
    },
}

# Dostępne turbiny w FLORIS turbine_library
TURBINE_LIBRARY = {
    "nrel_5MW": {"name": "NREL 5 MW", "diameter": 126.0, "hub_height": 90.0, "rated_power": 5.0},
    "iea_10MW": {"name": "IEA 10 MW", "diameter": 198.0, "hub_height": 119.0, "rated_power": 10.0},
    "iea_15MW": {"name": "IEA 15 MW", "diameter": 240.0, "hub_height": 150.0, "rated_power": 15.0},
    "iea_22MW": {"name": "IEA 22 MW", "diameter": 280.0, "hub_height": 170.0, "rated_power": 22.0},
}


# ---------------------------------------------------------------------------
# Konfiguracja layoutu
# ---------------------------------------------------------------------------
@dataclass
class LayoutConfig:
    """Parametry generowania layoutu farmy."""
    n_turbines: int = 25
    spacing_D: float = 7.0       # rozstaw w wielokrotnościach średnicy
    n_rows: int = 5
    n_cols: int = 5
    stagger_offset: float = 0.5  # przesunięcie co drugi rząd (0–1, w jednostkach spacing)
    boundary_radius: float = 5000.0  # promień granicy [m] (dla circular)


# ---------------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------------
class FarmModel:
    """Wrapper na FlorisModel z łatwym przełączaniem konfiguracji.

    Użycie:
        >>> farm = FarmModel(wake_model="gch", turbine="iea_15MW")
        >>> farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
        >>> farm.set_wind_data(wind_rose)
        >>> farm.run()
        >>> print(f"AEP: {farm.get_aep_gwh():.2f} GWh")
        >>> farm.plot_flow_field()
    """

    def __init__(
        self,
        wake_model: str = "gch",
        turbine: str = "iea_15MW",
        wind_data: Optional[Union[WindRose, TimeSeries]] = None,
    ):
        """
        Args:
            wake_model: Nazwa modelu wake. Dostępne: jensen, gch, turbopark,
                        empirical_gauss, cc.
            turbine: Nazwa turbiny. Dostępne: nrel_5MW, iea_10MW, iea_15MW, iea_22MW.
            wind_data: Opcjonalne dane wiatrowe (WindRose lub TimeSeries).
        """
        if wake_model not in WAKE_MODELS:
            available = ", ".join(WAKE_MODELS.keys())
            raise ValueError(f"Nieznany model wake '{wake_model}'. Dostępne: {available}")

        if turbine not in TURBINE_LIBRARY:
            available = ", ".join(TURBINE_LIBRARY.keys())
            raise ValueError(f"Nieznana turbina '{turbine}'. Dostępne: {available}")

        self.wake_model_name = wake_model
        self.turbine_name = turbine
        self.turbine_info = TURBINE_LIBRARY[turbine]

        # Layout — domyślnie pusta (1 turbina)
        self._layout_x = np.array([0.0])
        self._layout_y = np.array([0.0])
        self._layout_type = "single"
        self._wind_data = None

        # Buduj FlorisModel z odpowiednimi parametrami
        self._fmodel = self._build_floris_model(wake_model, turbine)

        # Ustaw dane wiatrowe jeśli podane
        if wind_data is not None:
            self.set_wind_data(wind_data)

        logger.info(
            f"FarmModel: wake={wake_model}, turbina={turbine} "
            f"({self.turbine_info['name']}, D={self.D}m)"
        )

    # ------------------------------------------------------------------
    # Właściwości (properties)
    # ------------------------------------------------------------------
    @property
    def D(self) -> float:
        """Średnica rotora [m]."""
        return self.turbine_info["diameter"]

    @property
    def hub_height(self) -> float:
        """Wysokość piasty [m]."""
        return self.turbine_info["hub_height"]

    @property
    def n_turbines(self) -> int:
        """Liczba turbin w layoucie."""
        return len(self._layout_x)

    @property
    def layout_x(self) -> np.ndarray:
        """Współrzędne X turbin [m]."""
        return self._layout_x.copy()

    @property
    def layout_y(self) -> np.ndarray:
        """Współrzędne Y turbin [m]."""
        return self._layout_y.copy()

    @property
    def fmodel(self) -> FlorisModel:
        """Dostęp do surowego FlorisModel (do zaawansowanych operacji)."""
        return self._fmodel

    # ------------------------------------------------------------------
    # Budowanie modelu FLORIS
    # ------------------------------------------------------------------
    def _build_floris_model(self, wake_model: str, turbine: str) -> FlorisModel:
        """Tworzy FlorisModel z podanym modelem wake i turbiną."""

        # Załaduj domyślną konfigurację jako słownik
        input_dict = load_yaml(_DEFAULT_YAML)

        # Ustaw model wake
        wake_params = WAKE_MODELS[wake_model]
        input_dict["wake"]["model_strings"]["velocity_model"] = wake_params["velocity_model"]
        input_dict["wake"]["model_strings"]["deflection_model"] = wake_params["deflection_model"]
        input_dict["wake"]["model_strings"]["turbulence_model"] = wake_params["turbulence_model"]
        input_dict["wake"]["model_strings"]["combination_model"] = wake_params["combination_model"]

        # GCH-specific features — wyłącz dla modeli które ich nie obsługują
        gch_only = wake_model == "gch"
        input_dict["wake"]["enable_secondary_steering"] = gch_only
        input_dict["wake"]["enable_yaw_added_recovery"] = gch_only
        input_dict["wake"]["enable_transverse_velocities"] = gch_only

        # Parametry wake — zamień na klucze odpowiednie dla wybranego modelu.
        # Puste dicty = FLORIS użyje wbudowanych domyślnych wartości.
        vel_model = wake_params["velocity_model"]
        defl_model = wake_params["deflection_model"]
        turb_model = wake_params["turbulence_model"]

        input_dict["wake"]["wake_velocity_parameters"] = {vel_model: {}}
        input_dict["wake"]["wake_deflection_parameters"] = {defl_model: {}}
        input_dict["wake"]["wake_turbulence_parameters"] = {turb_model: {}}

        # Ustaw turbinę
        input_dict["farm"]["turbine_type"] = [turbine]

        # Ustaw reference_wind_height na hub height turbiny
        input_dict["flow_field"]["reference_wind_height"] = TURBINE_LIBRARY[turbine]["hub_height"]

        # Stwórz model ze słownika (nie z pliku)
        fmodel = FlorisModel(input_dict)

        return fmodel

    # ------------------------------------------------------------------
    # Przełączanie modeli i turbin
    # ------------------------------------------------------------------
    def switch_wake_model(self, wake_model: str) -> None:
        """Zmienia model wake zachowując resztę konfiguracji.

        Args:
            wake_model: Nowy model wake (jensen, gch, turbopark, empirical_gauss, cc).
        """
        if wake_model not in WAKE_MODELS:
            available = ", ".join(WAKE_MODELS.keys())
            raise ValueError(f"Nieznany model: '{wake_model}'. Dostępne: {available}")

        self.wake_model_name = wake_model
        self._fmodel = self._build_floris_model(wake_model, self.turbine_name)
        self._apply_current_layout()
        if self._wind_data is not None:
            self._fmodel.set(wind_data=self._wind_data)
        logger.info(f"Zmieniono model wake na: {wake_model}")

    def switch_turbine(self, turbine: str) -> None:
        """Zmienia turbinę zachowując resztę konfiguracji.

        UWAGA: Jeśli layout był generowany z spacing w wielokrotnościach D,
        rozstaw fizyczny (w metrach) pozostaje taki sam. Rozważ
        przeliczenie layoutu.

        Args:
            turbine: Nowa turbina (nrel_5MW, iea_10MW, iea_15MW, iea_22MW).
        """
        if turbine not in TURBINE_LIBRARY:
            available = ", ".join(TURBINE_LIBRARY.keys())
            raise ValueError(f"Nieznana turbina: '{turbine}'. Dostępne: {available}")

        self.turbine_name = turbine
        self.turbine_info = TURBINE_LIBRARY[turbine]
        self._fmodel = self._build_floris_model(self.wake_model_name, turbine)
        self._apply_current_layout()
        if self._wind_data is not None:
            self._fmodel.set(wind_data=self._wind_data)
        logger.info(f"Zmieniono turbinę na: {turbine} ({self.turbine_info['name']})")

    # ------------------------------------------------------------------
    # Generatory layoutów
    # ------------------------------------------------------------------
    def set_layout_grid(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        spacing_D: float = 7.0,
    ) -> None:
        """Układ siatki regularnej (rectangular grid).

        Args:
            n_rows: Liczba rzędów.
            n_cols: Liczba kolumn.
            spacing_D: Rozstaw w wielokrotnościach średnicy rotora.
        """
        spacing_m = spacing_D * self.D
        x, y = np.meshgrid(
            np.arange(n_cols) * spacing_m,
            np.arange(n_rows) * spacing_m,
        )
        self._layout_x = x.flatten()
        self._layout_y = y.flatten()
        self._layout_type = f"grid {n_rows}x{n_cols} @ {spacing_D}D"
        self._apply_current_layout()

        logger.info(
            f"Layout grid: {n_rows}x{n_cols} = {self.n_turbines} turbin, "
            f"rozstaw {spacing_D}D = {spacing_m:.0f}m"
        )

    def set_layout_staggered(
        self,
        n_rows: int = 5,
        n_cols: int = 5,
        spacing_D: float = 7.0,
        offset: float = 0.5,
    ) -> None:
        """Układ siatki przesuniętej (staggered/offset grid).

        Co drugi rząd jest przesunięty o offset * spacing w kierunku Y.
        Zmniejsza wake losses w porównaniu do siatki regularnej.

        Args:
            n_rows: Liczba rzędów.
            n_cols: Liczba kolumn.
            spacing_D: Rozstaw w wielokrotnościach D.
            offset: Przesunięcie co drugiego rzędu (0–1).
        """
        spacing_m = spacing_D * self.D
        xs, ys = [], []

        for row in range(n_rows):
            for col in range(n_cols):
                x = col * spacing_m
                y = row * spacing_m
                if row % 2 == 1:
                    x += offset * spacing_m
                xs.append(x)
                ys.append(y)

        self._layout_x = np.array(xs)
        self._layout_y = np.array(ys)
        self._layout_type = f"staggered {n_rows}x{n_cols} @ {spacing_D}D (offset={offset})"
        self._apply_current_layout()

        logger.info(
            f"Layout staggered: {n_rows}x{n_cols} = {self.n_turbines} turbin, "
            f"rozstaw {spacing_D}D, offset={offset}"
        )

    def set_layout_circular(
        self,
        n_turbines: int = 12,
        radius_D: float = 10.0,
        n_rings: int = 1,
    ) -> None:
        """Układ kołowy (circular/ring layout).

        Args:
            n_turbines: Liczba turbin na pierścień zewnętrzny.
            radius_D: Promień pierścienia w wielokrotnościach D.
            n_rings: Liczba pierścieni (1 = sam pierścień, 2+ = zagnieżdżone).
        """
        xs, ys = [], []

        for ring in range(1, n_rings + 1):
            r = ring * radius_D * self.D / n_rings
            n_on_ring = max(4, int(n_turbines * ring / n_rings))
            angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
            for a in angles:
                xs.append(r * np.cos(a))
                ys.append(r * np.sin(a))

        # Dodaj turbinę centralną jeśli jest więcej niż 1 pierścień
        if n_rings > 1:
            xs.append(0.0)
            ys.append(0.0)

        self._layout_x = np.array(xs)
        self._layout_y = np.array(ys)
        self._layout_type = f"circular {len(xs)} turbin, {n_rings} ring(s) @ {radius_D}D"
        self._apply_current_layout()

        logger.info(
            f"Layout circular: {len(xs)} turbin, "
            f"{n_rings} pierścień(i), promień {radius_D}D"
        )

    def set_layout_custom(
        self,
        x: np.ndarray,
        y: np.ndarray,
        name: str = "custom",
    ) -> None:
        """Układ niestandardowy — podaj współrzędne ręcznie.

        Args:
            x: Współrzędne X turbin [m].
            y: Współrzędne Y turbin [m].
            name: Nazwa layoutu (do logów).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) != len(y):
            raise ValueError(f"Różna długość x ({len(x)}) i y ({len(y)})")

        self._layout_x = x
        self._layout_y = y
        self._layout_type = f"custom '{name}' ({len(x)} turbin)"
        self._apply_current_layout()

    def set_layout_from_csv(self, filepath: Union[str, Path]) -> None:
        """Ładuje layout z pliku CSV (eksport z layout.csv).

        Oczekiwany format: turbine_id, x_m, y_m, turbine_type

        Args:
            filepath: Ścieżka do pliku CSV.
        """
        import pandas as pd
        df = pd.read_csv(filepath)
        self.set_layout_custom(
            x=df["x_m"].values,
            y=df["y_m"].values,
            name=Path(filepath).stem,
        )

    def _apply_current_layout(self) -> None:
        """Aplikuje aktualny layout do modelu FLORIS."""
        self._fmodel.set(
            layout_x=self._layout_x.tolist(),
            layout_y=self._layout_y.tolist(),
        )

    # ------------------------------------------------------------------
    # Dane wiatrowe
    # ------------------------------------------------------------------
    def set_wind_data(self, wind_data: Union[WindRose, TimeSeries]) -> None:
        """Ustawia dane wiatrowe w modelu.

        Args:
            wind_data: WindRose lub TimeSeries z WindDataLoader.
        """
        self._wind_data = wind_data
        self._fmodel.set(wind_data=wind_data)
        logger.info(f"Ustawiono dane wiatrowe: {type(wind_data).__name__}")

    # ------------------------------------------------------------------
    # Symulacja
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Uruchamia symulację FLORIS."""
        self._fmodel.run()

    def get_aep_gwh(self) -> float:
        """Zwraca AEP farmy w GWh."""
        return self._fmodel.get_farm_AEP() / 1e9

    def get_aep_per_turbine_gwh(self) -> np.ndarray:
        """Zwraca AEP per turbina w GWh."""
        return self._fmodel.get_turbine_AEPs() / 1e9

    def get_turbine_powers_mw(self) -> np.ndarray:
        """Zwraca moce turbin [MW] dla wszystkich warunków wiatrowych."""
        return self._fmodel.get_turbine_powers() / 1e6

    def get_farm_power_mw(self) -> np.ndarray:
        """Zwraca moc farmy [MW] per warunek wiatrowy."""
        powers = self.get_turbine_powers_mw()
        return np.nansum(powers, axis=-1)

    def get_wake_losses_percent(self) -> float:
        """Oblicza straty wake jako procent.

        Porównuje AEP farmy z AEP bez wake (no-wake baseline).
        Baseline = n_turbin × AEP jednej turbiny bez interakcji wake.
        """
        if self._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych — użyj set_wind_data() najpierw.")

        # AEP z wake
        self._fmodel.set(wind_data=self._wind_data)
        self.run()
        aep_wake = self.get_aep_gwh()

        # AEP bez wake — jedna turbina × n_turbin
        single = FarmModel(
            wake_model=self.wake_model_name,
            turbine=self.turbine_name,
        )
        single.set_layout_custom(
            x=np.array([0.0]),
            y=np.array([0.0]),
            name="single",
        )
        single.set_wind_data(self._wind_data)
        single.run()
        aep_single = single.get_aep_gwh()
        aep_no_wake = aep_single * self.n_turbines

        if aep_no_wake == 0:
            return 0.0

        return (1.0 - aep_wake / aep_no_wake) * 100.0

    # ------------------------------------------------------------------
    # Wizualizacja — Flow Field
    # ------------------------------------------------------------------
    def plot_flow_field(
        self,
        wind_direction: float = 270.0,
        wind_speed: float = 9.0,
        ti: float = 0.06,
        height: Optional[float] = None,
        x_bounds: Optional[tuple] = None,
        y_bounds: Optional[tuple] = None,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        show_rotors: bool = True,
        show_labels: bool = True,
        figsize: tuple = (14, 6),
    ) -> plt.Figure:
        """Rysuje mapę cieplną pola przepływu (horizontal cut plane).

        Args:
            wind_direction: Kierunek wiatru [°].
            wind_speed: Prędkość wiatru [m/s].
            ti: Intensywność turbulencji.
            height: Wysokość cięcia [m]. Domyślnie hub_height.
            x_bounds: Zakres X (min, max) [m].
            y_bounds: Zakres Y (min, max) [m].
            ax: Oś matplotlib. Jeśli None, tworzy nową.
            title: Tytuł wykresu.
            show_rotors: Czy rysować rotory turbin.
            show_labels: Czy rysować numery turbin.
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        if height is None:
            height = self.hub_height

        # Ustaw pojedynczy warunek wiatrowy do wizualizacji
        self._fmodel.set(
            wind_directions=[wind_direction],
            wind_speeds=[wind_speed],
            turbulence_intensities=[ti],
        )

        # Automatyczne granice z zapasem
        if x_bounds is None:
            margin = 3 * self.D
            x_bounds = (
                float(self._layout_x.min() - margin),
                float(self._layout_x.max() + 15 * self.D),
            )
        if y_bounds is None:
            margin = 3 * self.D
            y_bounds = (
                float(self._layout_y.min() - margin),
                float(self._layout_y.max() + margin),
            )

        # Oblicz pole przepływu
        horizontal_plane = self._fmodel.calculate_horizontal_plane(
            height=height,
            x_resolution=200,
            y_resolution=100,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Rysuj heatmapę
        flowviz.visualize_cut_plane(
            horizontal_plane,
            ax=ax,
            label_contours=False,
            title=title or (
                f"Flow field — {self.wake_model_name.upper()} | "
                f"{self.turbine_info['name']} | "
                f"WD={wind_direction}° WS={wind_speed} m/s"
            ),
        )

        # Rysuj turbiny
        if show_rotors:
            layoutviz.plot_turbine_rotors(self._fmodel, ax=ax)
        if show_labels:
            layoutviz.plot_turbine_labels(self._fmodel, ax=ax)

        return fig

    def plot_cross_section(
        self,
        downstream_D: float = 5.0,
        wind_direction: float = 270.0,
        wind_speed: float = 9.0,
        ti: float = 0.06,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (8, 6),
    ) -> plt.Figure:
        """Rysuje przekrój poprzeczny śladu (vertical cross-section).

        Args:
            downstream_D: Odległość za turbiną w wielokrotnościach D.
            wind_direction: Kierunek wiatru [°].
            wind_speed: Prędkość wiatru [m/s].
            ti: Intensywność turbulencji.
            ax: Oś matplotlib.
            figsize: Rozmiar figury.

        Returns:
            matplotlib Figure.
        """
        self._fmodel.set(
            wind_directions=[wind_direction],
            wind_speeds=[wind_speed],
            turbulence_intensities=[ti],
        )

        x_loc = float(self._layout_x.min() + downstream_D * self.D)
        y_bounds = (
            float(self._layout_y.min() - 3 * self.D),
            float(self._layout_y.max() + 3 * self.D),
        )

        try:
            cross_plane = self._fmodel.calculate_cross_plane(
                downstream_dist=x_loc,
                y_resolution=100,
                z_resolution=100,
                y_bounds=y_bounds,
                z_bounds=(0.1, 2.5 * self.hub_height),
            )

            # Zastąp NaN/inf w danych wartością minimalną
            df = cross_plane.df
            u_col = "u" if "u" in df.columns else df.columns[-1]
            df[u_col] = df[u_col].fillna(df[u_col].min())
            df[u_col] = df[u_col].replace([np.inf, -np.inf], df[u_col].min())

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()

            flowviz.visualize_cut_plane(
                cross_plane,
                ax=ax,
                label_contours=False,
                title=f"Cross-section @ {downstream_D}D downstream | {self.wake_model_name.upper()}",
            )
        except Exception as e:
            logger.warning(f"Cross-section plot failed: {e}")
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            ax.text(
                0.5, 0.5,
                f"Cross-section niedostępny\ndla tego modelu wake\n({e})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#8a8a82",
            )
            ax.set_title(f"Cross-section @ {downstream_D}D | {self.wake_model_name.upper()}")

        return fig

    def plot_layout(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (8, 8),
        show_spacing: bool = True,
    ) -> plt.Figure:
        """Rysuje rozmieszczenie turbin (top-down view).

        Args:
            ax: Oś matplotlib.
            figsize: Rozmiar figury.
            show_spacing: Czy pokazywać linie rozstawu.

        Returns:
            matplotlib Figure.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Punkty turbin
        ax.scatter(
            self._layout_x, self._layout_y,
            s=120, c="#1e5c3a", edgecolors="white",
            linewidths=1.5, zorder=5,
        )

        # Numery turbin
        for i, (x, y) in enumerate(zip(self._layout_x, self._layout_y)):
            ax.annotate(
                str(i), (x, y),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8, color="#4a4a45",
            )

        # Koła rozstawu (opcjonalne)
        if show_spacing and self.n_turbines > 1:
            for x, y in zip(self._layout_x, self._layout_y):
                circle = plt.Circle(
                    (x, y), self.D / 2, fill=False,
                    color="#d8d5cc", linewidth=0.5, linestyle="--",
                )
                ax.add_patch(circle)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(
            f"Layout: {self._layout_type}\n"
            f"{self.turbine_info['name']} | {self.n_turbines} turbin | D={self.D}m"
        )
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        return fig

    # ------------------------------------------------------------------
    # Porównania
    # ------------------------------------------------------------------
    def compare_wake_models(
        self,
        models: list[str] = None,
        wind_direction: float = 270.0,
        wind_speed: float = 9.0,
        ti: float = 0.06,
        figsize: tuple = (16, 10),
    ) -> tuple[plt.Figure, dict]:
        """Porównuje modele wake na tym samym layoucie.

        Args:
            models: Lista modeli do porównania. Domyślnie wszystkie.
            wind_direction: Kierunek wiatru [°].
            wind_speed: Prędkość wiatru [m/s].
            ti: Intensywność turbulencji.
            figsize: Rozmiar figury.

        Returns:
            Tuple (Figure, dict z wynikami AEP per model).
        """
        if models is None:
            models = list(WAKE_MODELS.keys())

        n = len(models)
        fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

        original_model = self.wake_model_name
        results = {}

        for i, model_name in enumerate(models):
            ax = axes[0, i]
            self.switch_wake_model(model_name)

            try:
                self.plot_flow_field(
                    wind_direction=wind_direction,
                    wind_speed=wind_speed,
                    ti=ti,
                    ax=ax,
                    title=model_name.upper(),
                    show_labels=False,
                )

                # AEP jeśli mamy wind_data
                try:
                    if self._wind_data is not None:
                        self._fmodel.set(wind_data=self._wind_data)
                    self.run()
                    aep = self.get_aep_gwh()
                    results[model_name] = aep
                    ax.text(
                        0.02, 0.02, f"AEP: {aep:.1f} GWh",
                        transform=ax.transAxes, fontsize=10,
                        color="white", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
                    )
                except Exception as e_aep:
                    results[model_name] = None
                    logger.warning(f"AEP calc failed for {model_name}: {e_aep}")

            except Exception as e:
                ax.text(0.5, 0.5, f"Błąd:\n{str(e)[:80]}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, wrap=True)
                results[model_name] = None
                logger.warning(f"Model {model_name} failed: {e}")

        fig.suptitle(
            f"Porównanie modeli wake — {self.turbine_info['name']} | "
            f"{self.n_turbines} turbin | WD={wind_direction}° WS={wind_speed} m/s",
            fontsize=14, fontweight="500",
        )
        fig.tight_layout()

        # Przywróć oryginalny model
        self.switch_wake_model(original_model)

        return fig, results

    def compare_turbines(
        self,
        turbines: list[str] = None,
        spacing_D: float = 7.0,
        n_rows: int = 5,
        n_cols: int = 5,
    ) -> dict:
        """Porównuje turbiny na tym samym layoucie (AEP, straty wake).

        Przelicza layout dla każdej turbiny zachowując spacing w D.

        Args:
            turbines: Lista turbin. Domyślnie wszystkie.
            spacing_D: Rozstaw w wielokrotnościach D.
            n_rows: Liczba rzędów siatki.
            n_cols: Liczba kolumn siatki.

        Returns:
            Dict z wynikami per turbina.
        """
        if turbines is None:
            turbines = list(TURBINE_LIBRARY.keys())

        original_turbine = self.turbine_name
        results = {}

        for turbine_name in turbines:
            self.switch_turbine(turbine_name)
            self.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)

            try:
                self.run()
                aep = self.get_aep_gwh()
                wake_loss = self.get_wake_losses_percent()
                results[turbine_name] = {
                    "name": TURBINE_LIBRARY[turbine_name]["name"],
                    "diameter": TURBINE_LIBRARY[turbine_name]["diameter"],
                    "rated_power_mw": TURBINE_LIBRARY[turbine_name]["rated_power"],
                    "n_turbines": self.n_turbines,
                    "aep_gwh": aep,
                    "wake_loss_pct": wake_loss,
                    "capacity_factor": aep / (
                        TURBINE_LIBRARY[turbine_name]["rated_power"]
                        * self.n_turbines * 8.76
                    ) * 100,
                }
            except Exception as e:
                logger.warning(f"Błąd dla {turbine_name}: {e}")
                results[turbine_name] = {"error": str(e)}

        # Przywróć oryginalną turbinę
        self.switch_turbine(original_turbine)
        self.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)

        return results

    def spacing_sweep(
        self,
        spacings_D: np.ndarray = None,
        n_rows: int = 5,
        n_cols: int = 5,
    ) -> dict:
        """Testuje różne rozstawy i zwraca AEP vs spacing.

        Args:
            spacings_D: Tablica rozstawów w wielokrotnościach D.
            n_rows: Liczba rzędów.
            n_cols: Liczba kolumn.

        Returns:
            Dict z tablicami spacings i aep_values.
        """
        if spacings_D is None:
            spacings_D = np.arange(4.0, 12.5, 0.5)

        aep_values = []
        wake_losses = []

        for spacing in spacings_D:
            self.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing)
            if self._wind_data is not None:
                self._fmodel.set(wind_data=self._wind_data)
            self.run()
            aep_values.append(self.get_aep_gwh())
            wake_losses.append(self.get_wake_losses_percent())

        return {
            "spacings_D": spacings_D,
            "aep_gwh": np.array(aep_values),
            "wake_loss_pct": np.array(wake_losses),
        }

    # ------------------------------------------------------------------
    # Eksport
    # ------------------------------------------------------------------
    def export_layout_csv(
        self,
        filepath: Union[str, Path] = "outputs/exports/layout.csv",
    ) -> None:
        """Eksportuje layout do CSV (format dla Tematu 3 i 4).

        Kolumny: turbine_id, x_m, y_m, turbine_type
        """
        import pandas as pd

        df = pd.DataFrame({
            "turbine_id": range(self.n_turbines),
            "x_m": self._layout_x,
            "y_m": self._layout_y,
            "turbine_type": self.turbine_name,
        })

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Layout wyeksportowany do: {filepath}")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"FarmModel("
            f"wake={self.wake_model_name}, "
            f"turbine={self.turbine_info['name']}, "
            f"n_turbin={self.n_turbines}, "
            f"layout={self._layout_type})"
        )
