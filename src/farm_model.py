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
# Wyciszenie ostrzeżeń FLORIS ("Computing AEP with uniform frequencies").
# FLORIS przekonfigurowuje logowanie przy tworzeniu modelu, więc ustawienie
# poziomu trzeba ponawiać po każdym zbudowaniu FlorisModel (patrz _silence_floris).
# ---------------------------------------------------------------------------
class _FlorisQuietFilter(logging.Filter):
    def filter(self, record):
        return "uniform frequencies" not in record.getMessage().lower()


def _silence_floris():
    lg = logging.getLogger("floris")
    lg.setLevel(logging.ERROR)
    for h in lg.handlers:
        if not any(isinstance(f, _FlorisQuietFilter) for f in h.filters):
            h.addFilter(_FlorisQuietFilter())


_silence_floris()


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
    "iea_15MW_floating": {
        "name": "IEA 15 MW Floating",
        "diameter": 242.24,
        "hub_height": 150.0,
        "rated_power": 15.0,
        "floris_id": "iea_15MW_floating_multi_dim_cp_ct",
        "floating": True,
    },
}

# Turbiny pływające — wymagają multidim_conditions (fale)
FLOATING_TURBINES = {"iea_15MW_floating"}

# Ścieżka do niestandardowych turbin użytkownika
_CUSTOM_TURBINE_DIR = Path("data/turbines")


# ---------------------------------------------------------------------------
# Generator krzywych mocy dla custom turbin
# ---------------------------------------------------------------------------
def generate_power_curve(
    rated_power_kw: float,
    rotor_diameter: float,
    cut_in: float = 3.0,
    rated_speed: float = 11.0,
    cut_out: float = 25.0,
    n_points: int = 50,
) -> tuple[list, list, list]:
    """Generuje realistyczne krzywe mocy i Ct z parametrów podstawowych.

    Model: cubic growth do rated_speed, potem const do cut_out.
    Ct: estymacja z mocy (Ct = P / (0.5 * rho * A * V^3)), max 0.9.

    Args:
        rated_power_kw: Moc znamionowa [kW].
        rotor_diameter: Średnica rotora [m].
        cut_in: Prędkość cut-in [m/s].
        rated_speed: Prędkość znamionowa [m/s].
        cut_out: Prędkość cut-out [m/s].
        n_points: Liczba punktów na krzywej.

    Returns:
        Tuple (wind_speeds, powers_kw, thrust_coefficients).
    """
    rho = 1.225
    area = np.pi * (rotor_diameter / 2) ** 2

    ws = np.concatenate([
        [0.0],
        np.linspace(cut_in - 0.1, cut_in, 2),
        np.linspace(cut_in + 0.5, rated_speed, n_points),
        np.linspace(rated_speed + 0.5, cut_out, 10),
        [cut_out + 0.1, 50.0],
    ])
    ws = np.unique(np.round(ws, 3))

    powers = np.zeros_like(ws)
    ct = np.zeros_like(ws)

    for i, v in enumerate(ws):
        if v < cut_in or v > cut_out:
            powers[i] = 0.0
            ct[i] = 0.0
        elif v <= rated_speed:
            # Cubic region
            frac = ((v - cut_in) / (rated_speed - cut_in)) ** 3
            powers[i] = rated_power_kw * frac
            # Ct z bilansu mocy
            p_avail = 0.5 * rho * area * v ** 3
            if p_avail > 0:
                cp = (powers[i] * 1000) / p_avail
                ct[i] = min(0.9, cp / 0.4 * 0.8)  # empiryczne przybliżenie
            else:
                ct[i] = 0.0
        else:
            # Rated region
            powers[i] = rated_power_kw
            p_avail = 0.5 * rho * area * v ** 3
            if p_avail > 0:
                cp = (powers[i] * 1000) / p_avail
                ct[i] = min(0.5, cp / 0.4 * 0.8)
            else:
                ct[i] = 0.0

    return ws.tolist(), powers.tolist(), ct.tolist()


def create_custom_turbine_yaml(
    name: str,
    rated_power_mw: float,
    rotor_diameter: float,
    hub_height: float,
    cut_in: float = 3.0,
    rated_speed: float = 11.0,
    cut_out: float = 25.0,
    tsr: float = 8.0,
    save_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Tworzy plik YAML turbiny kompatybilny z FLORIS.

    Args:
        name: Unikalna nazwa turbiny (np. 'my_turbine_8MW').
        rated_power_mw: Moc znamionowa [MW].
        rotor_diameter: Średnica rotora [m].
        hub_height: Wysokość piasty [m].
        cut_in: Prędkość cut-in [m/s].
        rated_speed: Prędkość znamionowa [m/s].
        cut_out: Prędkość cut-out [m/s].
        tsr: Tip-speed ratio.
        save_dir: Katalog do zapisu. Domyślnie data/turbines/.

    Returns:
        Ścieżka do zapisanego pliku YAML.
    """
    import yaml

    if save_dir is None:
        save_dir = _CUSTOM_TURBINE_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ws, power, ct = generate_power_curve(
        rated_power_kw=rated_power_mw * 1000,
        rotor_diameter=rotor_diameter,
        cut_in=cut_in,
        rated_speed=rated_speed,
        cut_out=cut_out,
    )

    turbine_dict = {
        "turbine_type": name,
        "hub_height": float(hub_height),
        "rotor_diameter": float(rotor_diameter),
        "TSR": float(tsr),
        "operation_model": "cosine-loss",
        "power_thrust_table": {
            "ref_air_density": 1.225,
            "ref_tilt": 5.0,
            "cosine_loss_exponent_tilt": 1.88,
            "cosine_loss_exponent_yaw": 1.88,
            "wind_speed": ws,
            "power": power,
            "thrust_coefficient": ct,
        },
    }

    filepath = save_dir / f"{name}.yaml"
    with open(filepath, "w") as f:
        yaml.dump(turbine_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Custom turbine saved: {filepath}")
    return filepath


def register_custom_turbine(
    name: str,
    rated_power_mw: float,
    rotor_diameter: float,
    hub_height: float,
    **kwargs,
) -> str:
    """Tworzy YAML i rejestruje turbinę w TURBINE_LIBRARY.

    Returns:
        Klucz w TURBINE_LIBRARY.
    """
    filepath = create_custom_turbine_yaml(
        name=name,
        rated_power_mw=rated_power_mw,
        rotor_diameter=rotor_diameter,
        hub_height=hub_height,
        **kwargs,
    )

    key = name.lower().replace(" ", "_").replace("-", "_")
    TURBINE_LIBRARY[key] = {
        "name": f"{name} ({rated_power_mw:.1f} MW)",
        "diameter": rotor_diameter,
        "hub_height": hub_height,
        "rated_power": rated_power_mw,
        "custom_yaml": str(filepath),
    }

    logger.info(f"Registered custom turbine: {key}")
    return key


def load_custom_turbines_from_dir(directory: Union[str, Path] = None) -> int:
    """Ładuje wszystkie custom turbiny z katalogu do TURBINE_LIBRARY.

    Returns:
        Liczba załadowanych turbin.
    """
    import yaml

    if directory is None:
        directory = _CUSTOM_TURBINE_DIR
    directory = Path(directory)

    if not directory.exists():
        return 0

    count = 0
    for filepath in directory.glob("*.yaml"):
        try:
            with open(filepath) as f:
                t = yaml.safe_load(f)
            key = filepath.stem
            if key not in TURBINE_LIBRARY:
                rated_kw = max(t.get("power_thrust_table", {}).get("power", [0]))
                TURBINE_LIBRARY[key] = {
                    "name": f"{t['turbine_type']} ({rated_kw/1000:.1f} MW)",
                    "diameter": t["rotor_diameter"],
                    "hub_height": t["hub_height"],
                    "rated_power": rated_kw / 1000,
                    "custom_yaml": str(filepath),
                }
                count += 1
        except Exception as e:
            logger.warning(f"Nie udało się załadować {filepath}: {e}")

    return count


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
        wave_period: float = 2.0,
        wave_height: float = 1.0,
    ):
        """
        Args:
            wake_model: Nazwa modelu wake. Dostępne: jensen, gch, turbopark,
                        empirical_gauss, cc.
            turbine: Nazwa turbiny. Dostępne: nrel_5MW, iea_10MW, iea_15MW,
                      iea_22MW, iea_15MW_floating + custom.
            wind_data: Opcjonalne dane wiatrowe (WindRose lub TimeSeries).
            wave_period: Okres fali [s] — tylko dla turbin pływających (Tp=2 lub 4).
            wave_height: Wysokość fali [m] — tylko dla turbin pływających (Hs=1 lub 5).
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

        # Parametry fal (floating)
        self._wave_period = wave_period
        self._wave_height = wave_height

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
            + (f", floating Tp={wave_period}s Hs={wave_height}m" if self.is_floating else "")
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

    @property
    def is_floating(self) -> bool:
        """Czy turbina jest pływająca."""
        return self.turbine_name in FLOATING_TURBINES or self.turbine_info.get("floating", False)

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

        # Ustaw turbinę — obsługa floating i custom
        t_info = TURBINE_LIBRARY[turbine]

        if "custom_yaml" in t_info:
            # Custom turbina — ładuj z pliku YAML użytkownika
            import yaml as _yaml
            custom_path = Path(t_info["custom_yaml"])
            if custom_path.exists():
                with open(custom_path) as f:
                    custom_turbine = _yaml.safe_load(f)
                input_dict["farm"]["turbine_type"] = [custom_turbine]
            else:
                logger.warning(f"Custom YAML nie istnieje: {custom_path}, fallback na iea_15MW")
                input_dict["farm"]["turbine_type"] = ["iea_15MW"]
        elif "floris_id" in t_info:
            # Turbina z wbudowanej biblioteki FLORIS pod inną nazwą
            input_dict["farm"]["turbine_type"] = [t_info["floris_id"]]
        else:
            # Standardowa turbina
            input_dict["farm"]["turbine_type"] = [turbine]

        # Ustaw reference_wind_height na hub height turbiny
        input_dict["flow_field"]["reference_wind_height"] = t_info["hub_height"]

        # Floating — dodaj multidim_conditions (parametry fal)
        if turbine in FLOATING_TURBINES or t_info.get("floating", False):
            input_dict["flow_field"]["multidim_conditions"] = {
                "Tp": self._wave_period,
                "Hs": self._wave_height,
            }

        # Stwórz model ze słownika (nie z pliku)
        fmodel = FlorisModel(input_dict)
        _silence_floris()  # FLORIS mógł przekonfigurować logger — wycisz ponownie

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
            turbine: Nowa turbina (nrel_5MW, iea_10MW, iea_15MW, iea_22MW,
                      iea_15MW_floating + custom).
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

    def set_wave_conditions(self, wave_period: float, wave_height: float) -> None:
        """Zmienia parametry fal (tylko dla turbin pływających).

        Args:
            wave_period: Okres fali Tp [s] (dostępne: 2, 4).
            wave_height: Wysokość fali Hs [m] (dostępne: 1, 5).
        """
        self._wave_period = wave_period
        self._wave_height = wave_height

        if self.is_floating:
            self._fmodel = self._build_floris_model(self.wake_model_name, self.turbine_name)
            self._apply_current_layout()
            if self._wind_data is not None:
                self._fmodel.set(wind_data=self._wind_data)
            logger.info(f"Wave conditions: Tp={wave_period}s, Hs={wave_height}m")

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

    def set_layout_parallelogram(
        self,
        n_turbines: int = 25,
        r1_D: float = 7.0,
        r2_D: float = 7.0,
        theta1_deg: float = 90.0,
        theta2_deg: float = 18.0,
    ) -> None:
        """Układ oparty na równoległoboku (parallelogram grid).

        Bazowany na Malisani et al. (2025) — "Offshore wind farm layout
        optimization with alignment constraints", Wind Energ. Sci., 10.
        Turbiny umieszczone na przecięciach siatki równoległoboków.

        Parametry siatki: dwa wektory bazowe definiowane przez (r1, theta1)
        i (r2, theta2), gdzie r to długość boku, theta to kąt od osi X.

        Args:
            n_turbines: Docelowa liczba turbin.
            r1_D: Długość pierwszego boku [×D].
            r2_D: Długość drugiego boku [×D].
            theta1_deg: Kąt pierwszego wektora [°] od osi X.
            theta2_deg: Kąt drugiego wektora [°] od osi X.
        """
        r1 = r1_D * self.D
        r2 = r2_D * self.D
        t1 = np.radians(theta1_deg)
        t2 = np.radians(theta2_deg)

        # Wektory bazowe
        v1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1)])
        v2 = np.array([r2 * np.cos(t2), r2 * np.sin(t2)])

        # Generuj wystarczająco dużo punktów na siatce
        n_side = int(np.ceil(np.sqrt(n_turbines))) + 2
        xs, ys = [], []
        for i in range(-1, n_side + 1):
            for j in range(-1, n_side + 1):
                pos = i * v1 + j * v2
                xs.append(pos[0])
                ys.append(pos[1])

        xs = np.array(xs)
        ys = np.array(ys)

        # Przesuń tak żeby minimum to (0, 0)
        xs -= xs.min()
        ys -= ys.min()

        # Sortuj po odległości od centrum (żeby brać najbliższe)
        cx, cy = xs.mean(), ys.mean()
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        order = np.argsort(dists)

        # Weź n_turbines najbliższych centrum
        idx = order[:n_turbines]
        xs = xs[idx]
        ys = ys[idx]

        # Normalizuj — minimum na (0, 0)
        xs -= xs.min()
        ys -= ys.min()

        self._layout_x = xs
        self._layout_y = ys
        self._layout_type = (
            f"parallelogram {n_turbines}T, "
            f"r1={r1_D}D r2={r2_D}D θ1={theta1_deg}° θ2={theta2_deg}°"
        )
        self._apply_current_layout()

        logger.info(
            f"Layout parallelogram: {n_turbines} turbin, "
            f"r1={r1_D}D r2={r2_D}D θ1={theta1_deg}° θ2={theta2_deg}°"
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
        Wynik jest cache'owany — resetowany przy zmianie turbiny/wake modelu.
        """
        if self._wind_data is None:
            raise RuntimeError("Brak danych wiatrowych — użyj set_wind_data() najpierw.")

        # Cache key — MUSI zależeć od layoutu i danych wiatrowych, bo farma jest
        # współdzielona (cache_resource) i eval_wind się zmienia. Bez tego zwracano
        # nieaktualne straty wake po zmianie wiatru/layoutu.
        try:
            wd = self._wind_data
            wind_sig = (
                f"{float(np.sum(np.asarray(wd.wind_directions))):.2f}_"
                f"{float(np.sum(np.asarray(wd.wind_speeds))):.2f}_"
                f"{len(np.atleast_1d(wd.wind_directions))}"
            )
        except Exception:
            wind_sig = str(id(self._wind_data))
        layout_sig = f"{float(np.sum(self._layout_x)):.1f}_{float(np.sum(self._layout_y)):.1f}"
        cache_key = (
            f"{self.wake_model_name}_{self.turbine_name}_{self.n_turbines}_"
            f"{layout_sig}_{wind_sig}"
        )
        if hasattr(self, "_wl_cache_key") and self._wl_cache_key == cache_key:
            if hasattr(self, "_wl_cache_val"):
                return self._wl_cache_val

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

        result = (1.0 - aep_wake / aep_no_wake) * 100.0

        # Save to cache
        self._wl_cache_key = cache_key
        self._wl_cache_val = result

        return result

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
        show_wind_arrow: bool = True,
        figsize: tuple = None,
    ) -> plt.Figure:
        """Rysuje mapę cieplną pola przepływu (horizontal cut plane).

        Args:
            wind_direction: Kierunek wiatru [°] (meteorologiczny: 270=zachód).
            wind_speed: Prędkość wiatru [m/s].
            ti: Intensywność turbulencji.
            height: Wysokość cięcia [m]. Domyślnie hub_height.
            x_bounds: Zakres X (min, max) [m].
            y_bounds: Zakres Y (min, max) [m].
            ax: Oś matplotlib. Jeśli None, tworzy nową.
            title: Tytuł wykresu.
            show_rotors: Czy rysować rotory turbin.
            show_labels: Czy rysować numery turbin.
            show_wind_arrow: Czy rysować strzałkę kierunku wiatru.
            figsize: Rozmiar figury. Domyślnie dynamiczny z proporcji farmy.

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

        # Bounds — FLORIS obraca farmę, więc potrzebujemy duży margines
        # downstream (w kierunku wiatru) i mniejszy crosswind
        if x_bounds is None or y_bounds is None:
            # Margines: 3D upstream, 12D downstream, 5D crosswind
            margin_up = 3 * self.D
            margin_down = 12 * self.D
            margin_cross = 5 * self.D

            # Wektor wiatru (FLORIS: 270° = zachód = wiatr w prawo na osi X)
            wd_rad = np.radians(270.0 - wind_direction)
            wind_dx = np.cos(wd_rad)
            wind_dy = np.sin(wd_rad)

            # Oblicz potrzebne bounds na podstawie turbin + kierunku
            cx = (self._layout_x.min() + self._layout_x.max()) / 2
            cy = (self._layout_y.min() + self._layout_y.max()) / 2
            farm_radius = max(
                self._layout_x.max() - self._layout_x.min(),
                self._layout_y.max() - self._layout_y.min(),
            ) / 2 + self.D

            # Prostokąt obejmujący farmę + asymetryczny margines
            x_bounds = (
                float(cx - farm_radius - margin_up - abs(wind_dx) * margin_down),
                float(cx + farm_radius + margin_up + abs(wind_dx) * margin_down),
            )
            y_bounds = (
                float(cy - farm_radius - margin_cross - abs(wind_dy) * margin_down),
                float(cy + farm_radius + margin_cross + abs(wind_dy) * margin_down),
            )

        # Dynamiczny figsize z proporcji
        if figsize is None:
            x_span = x_bounds[1] - x_bounds[0]
            y_span = y_bounds[1] - y_bounds[0]
            ratio = y_span / x_span if x_span > 0 else 1.0
            fig_w = 12
            fig_h = max(4, min(12, fig_w * ratio))
            figsize = (fig_w, fig_h)

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

        # Strzałka kierunku wiatru
        if show_wind_arrow:
            wd_rad = np.radians(270.0 - wind_direction)
            arrow_len = 3 * self.D
            # Pozycja strzałki — lewy górny róg
            ax_xlim = ax.get_xlim()
            ax_ylim = ax.get_ylim()
            arrow_x = ax_xlim[0] + (ax_xlim[1] - ax_xlim[0]) * 0.08
            arrow_y = ax_ylim[1] - (ax_ylim[1] - ax_ylim[0]) * 0.10
            dx = arrow_len * np.cos(wd_rad)
            dy = arrow_len * np.sin(wd_rad)
            ax.annotate(
                "", xy=(arrow_x + dx, arrow_y + dy),
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.3",
                    color="white", lw=2.5,
                ),
                zorder=10,
            )
            ax.text(
                arrow_x + dx * 0.5, arrow_y + dy * 0.5 + arrow_len * 0.3,
                f"Wiatr {wind_direction}°",
                color="white", fontsize=9, fontweight="bold",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5),
                zorder=10,
            )

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
