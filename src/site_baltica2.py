"""
Geometria MFW Baltica 2 — granica, UTM 33N, layout bazowy
==========================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Granica: wielokąt obszaru zabudowy Baltica 2 z tabeli 7 raportu EIA
(71 punktów, UTM 33N, 137 km²) — plik data/sites/baltica_owf_eia_table7.csv.
Uwaga: wcześniej używany prostokąt 55°00'07"–55°06'51" N, 17°01'50"–17°27'20" E
był tylko obwiednią punktów z tej tabeli (obejmował też część Baltica 3).

Układy współrzędnych:
    - lat/lon WGS84 (mapa, eksport),
    - UTM strefa 33N (EPSG:32633) — metry, eksport dla T3/T4,
    - lokalny = UTM minus ORIGIN (lewy dolny róg granicy) — tego używa
      FLORIS i algorytmy (małe liczby, x = wschód, y = północ).

Przeliczenie UTM zaimplementowane tu (wzory Krügera, dokładność < 1 mm
w strefie), żeby nie wymagać pyproj.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import contains_xy

# ---------------------------------------------------------------------------
# UTM (WGS84) — forward / inverse, seria Krügera rzędu 6
# ---------------------------------------------------------------------------
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64 + _N**6 / 256)
_ALPHA = [
    _N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16 + 41 * _N**4 / 180 - 127 * _N**5 / 288 + 7891 * _N**6 / 37800,
    13 * _N**2 / 48 - 3 * _N**3 / 5 + 557 * _N**4 / 1440 + 281 * _N**5 / 630 - 1983433 * _N**6 / 1935360,
    61 * _N**3 / 240 - 103 * _N**4 / 140 + 15061 * _N**5 / 26880 + 167603 * _N**6 / 181440,
    49561 * _N**4 / 161280 - 179 * _N**5 / 168 + 6601661 * _N**6 / 7257600,
    34729 * _N**5 / 80640 - 3418889 * _N**6 / 1995840,
    212378941 * _N**6 / 319334400,
]
_BETA = [
    _N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96 - _N**4 / 360 - 81 * _N**5 / 512 + 96199 * _N**6 / 604800,
    _N**2 / 48 + _N**3 / 15 - 437 * _N**4 / 1440 + 46 * _N**5 / 105 - 1118711 * _N**6 / 3870720,
    17 * _N**3 / 480 - 37 * _N**4 / 840 - 209 * _N**5 / 4480 + 5569 * _N**6 / 90720,
    4397 * _N**4 / 161280 - 11 * _N**5 / 504 - 830251 * _N**6 / 7257600,
    4583 * _N**5 / 161280 - 108847 * _N**6 / 3991680,
    20648693 * _N**6 / 638668800,
]
_E2 = _F * (2 - _F)
_E = math.sqrt(_E2)
UTM_ZONE = 33
EPSG = 32633


def _lon0(zone: int) -> float:
    return math.radians(-183 + 6 * zone)


def latlon_to_utm(lat, lon, zone: int = UTM_ZONE):
    """WGS84 lat/lon [°] → UTM (E, N) [m], półkula północna."""
    lat = np.radians(np.asarray(lat, dtype=float))
    lon = np.radians(np.asarray(lon, dtype=float)) - _lon0(zone)
    t = np.sinh(np.arctanh(np.sin(lat)) - _E * np.arctanh(_E * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(lon))
    eta_p = np.arctanh(np.sin(lon) / np.sqrt(1 + t**2))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, a in enumerate(_ALPHA, start=1):
        xi = xi + a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    return 500000.0 + _K0 * _AA * eta, _K0 * _AA * xi


def utm_to_latlon(E, N, zone: int = UTM_ZONE):
    """UTM (E, N) [m] → WGS84 lat/lon [°], półkula północna."""
    xi = np.asarray(N, dtype=float) / (_K0 * _AA)
    eta = (np.asarray(E, dtype=float) - 500000.0) / (_K0 * _AA)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, b in enumerate(_BETA, start=1):
        xi_p = xi_p - b * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - b * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    tau_p = np.sin(xi_p) / np.sqrt(np.sinh(eta_p) ** 2 + np.cos(xi_p) ** 2)
    lon = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    tau = tau_p.copy()
    for _ in range(6):  # Newton dla tau
        sigma = np.sinh(_E * np.arctanh(_E * tau / np.sqrt(1 + tau**2)))
        tp = tau * np.sqrt(1 + sigma**2) - sigma * np.sqrt(1 + tau**2)
        dtau = (tau_p - tp) / np.sqrt(1 + tp**2) * (1 + (1 - _E2) * tau**2) / (
            (1 - _E2) * np.sqrt(1 + tau**2))
        tau = tau + dtau
    return np.degrees(np.arctan(tau)), np.degrees(lon + _lon0(zone))


# ---------------------------------------------------------------------------
# Granica Baltica 2
# ---------------------------------------------------------------------------
from pathlib import Path

# Tabela 7 raportu EIA „OWF Baltica” (EN, wersja A, s. 44–47): 122 punkty
# obszaru zabudowy farmy w UTM 33N. Punkty 52–122 = Baltica 2 (zachód,
# 137 km²), 1–51 = Baltica 3 (wschód, 101 km²); razem 237,63 km² wg raportu.
SITES_CSV = Path(__file__).resolve().parent.parent / "data" / "sites" / "baltica_owf_eia_table7.csv"
BOUNDARY_SOURCE = "Raport EIA OWF Baltica, tab. 7 (obszar zabudowy farmy, UTM 33N)"

# Wynik z danych IMGW 1991–2010, 150 m (P35 i P50, ważone energią ws³):
DOMINANT_WD_DEG = 260.0


# Miejsca ważne dla projektu (lat, lon) — do mapy
PLACES = {
    "Ustka": (54.5805, 16.8614),
    "Łeba": (54.7600, 17.5560),
    "Słupsk": (54.4641, 17.0285),
    "Choczewo (przyłącze lądowe)": (54.7450, 17.8930),
}
IMGW_POINTS_CSV = SITES_CSV.parent.parent / "processed" / "imgw_grid_points.csv"


def imgw_grid_points() -> pd.DataFrame:
    """50 punktów siatki IMGW (point_id, XLAT, XLONG, south_north, west_east)."""
    return pd.read_csv(IMGW_POINTS_CSV)


def _area_points(area: str = "baltica2") -> pd.DataFrame:
    df = pd.read_csv(SITES_CSV)
    df = df[df["area"] == area].sort_values("point_no")
    # ostatni punkt zamyka pierścień (= pierwszy) — usuń duplikat
    if np.hypot(df.utm33_e.iloc[0] - df.utm33_e.iloc[-1], df.utm33_n.iloc[0] - df.utm33_n.iloc[-1]) < 1.0:
        df = df.iloc[:-1]
    return df


def boundary_utm(area: str = "baltica2") -> tuple[np.ndarray, np.ndarray]:
    df = _area_points(area)
    return df.utm33_e.to_numpy(), df.utm33_n.to_numpy()


def boundary_latlon(area: str = "baltica2") -> tuple[np.ndarray, np.ndarray]:
    """Granica jako zamknięty wielokąt (lat, lon) — do rysowania na mapie."""
    df = _area_points(area)
    lat, lon = df.lat_deg.to_numpy(), df.lon_deg.to_numpy()
    return np.append(lat, lat[0]), np.append(lon, lon[0])


@dataclass
class Site:
    """Obszar farmy w układzie lokalnym [m] (x = wschód, y = północ)."""
    name: str
    polygon: Polygon          # lokalne metry
    origin_e: float           # UTM E punktu (0, 0)
    origin_n: float           # UTM N punktu (0, 0)
    zone: int = UTM_ZONE

    # --- przeliczenia ---
    def local_to_utm(self, x, y):
        return np.asarray(x, float) + self.origin_e, np.asarray(y, float) + self.origin_n

    def utm_to_local(self, e, n):
        return np.asarray(e, float) - self.origin_e, np.asarray(n, float) - self.origin_n

    def local_to_latlon(self, x, y):
        e, n = self.local_to_utm(x, y)
        return utm_to_latlon(e, n, self.zone)

    def latlon_to_local(self, lat, lon):
        e, n = latlon_to_utm(lat, lon, self.zone)
        return self.utm_to_local(e, n)

    # --- ograniczenia ---
    def contains(self, x, y, margin: float = 0.0) -> np.ndarray:
        """True dla punktów wewnątrz granicy (z marginesem do środka [m])."""
        poly = self.polygon.buffer(-margin) if margin > 0 else self.polygon
        return contains_xy(poly, np.asarray(x, float), np.asarray(y, float))

    @property
    def bounds_rect(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.polygon.bounds)

    @property
    def area_km2(self) -> float:
        return self.polygon.area / 1e6


def baltica2_site() -> Site:
    e, n = boundary_utm("baltica2")
    oe, on = float(np.floor(e.min())), float(np.floor(n.min()))
    return Site("Baltica 2", Polygon(np.column_stack([e - oe, n - on])), oe, on)


# ---------------------------------------------------------------------------
# Layout bazowy: siatka obrócona do dominującego kierunku wiatru
# ---------------------------------------------------------------------------
def rotated_grid_layout(
    site: Site,
    n_turbines: int = 107,
    D: float = 240.0,
    wd_deg: float = DOMINANT_WD_DEG,
    margin: float = 120.0,
    ratios: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0),
    n_offsets: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Regularna siatka: rzędy prostopadłe do dominującego wiatru, rozstaw
    wzdłuż wiatru = ratio × rozstaw w poprzek. Dla każdego ratio szuka
    największego rozstawu, przy którym ≥ n_turbines mieści się w granicy
    (z marginesem), i wybiera wariant o największym minimalnym odstępie.
    Nadmiarowe turbiny (najdalsze od środka) są odrzucane.

    Returns:
        x, y (lokalne [m]), info (rozstawy w D, ratio, kąt).
    """
    poly = site.polygon.buffer(-margin) if margin > 0 else site.polygon
    cx, cy = poly.centroid.x, poly.centroid.y
    # kierunek wiatru "skąd" → wektor wzdłuż przepływu (oś "along")
    th = math.radians(270.0 - wd_deg)
    ax_along = np.array([math.cos(th), math.sin(th)])
    ax_cross = np.array([-ax_along[1], ax_along[0]])
    R = math.hypot(*(np.array(poly.bounds[2:]) - np.array(poly.bounds[:2])))

    def fit(s_cross, ratio, ox, oy):
        s_along = s_cross * ratio
        i = np.arange(-int(R / s_along) - 1, int(R / s_along) + 2)
        j = np.arange(-int(R / s_cross) - 1, int(R / s_cross) + 2)
        I, J = np.meshgrid(i, j)
        a = (I.ravel() + ox) * s_along
        c = (J.ravel() + oy) * s_cross
        x = cx + a * ax_along[0] + c * ax_cross[0]
        y = cy + a * ax_along[1] + c * ax_cross[1]
        m = contains_xy(poly, x, y)
        return x[m], y[m]

    best = None
    offs = np.linspace(0, 1, n_offsets, endpoint=False)
    for ratio in ratios:
        lo, hi = 2.0 * D, 30.0 * D
        cand = None
        for _ in range(30):  # bisekcja po rozstawie
            mid = (lo + hi) / 2
            got = None
            for ox in offs:
                for oy in offs:
                    x, y = fit(mid, ratio, ox, oy)
                    if len(x) >= n_turbines:
                        got = (x, y, ox, oy)
                        break
                if got:
                    break
            if got:
                lo, cand = mid, (mid, *got)
            else:
                hi = mid
        if cand is None:
            continue
        s_cross, x, y, ox, oy = cand
        min_sp = min(s_cross, s_cross * ratio)
        if best is None or min_sp > best[0] + 1e-6:
            best = (min_sp, s_cross, ratio, x, y)

    if best is None:
        raise ValueError("Nie da się zmieścić turbin w granicy.")
    _, s_cross, ratio, x, y = best
    d = np.hypot(x - cx, y - cy)
    keep = np.argsort(d)[:n_turbines]
    x, y = x[keep], y[keep]
    order = np.lexsort((y, x))  # numeracja z zachodu na wschód
    info = {
        "spacing_cross_D": s_cross / D,
        "spacing_along_D": s_cross * ratio / D,
        "ratio": ratio,
        "wd_deg": wd_deg,
        "margin_m": margin,
    }
    return x[order], y[order], info


def layout_table(
    site: Site, x, y, turbine_type: str = "iea_15MW"
) -> pd.DataFrame:
    """Tabela do eksportu layout.csv: id, UTM 33N, lat/lon, lokalne, typ."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    e, n = site.local_to_utm(x, y)
    lat, lon = utm_to_latlon(e, n, site.zone)
    return pd.DataFrame({
        "turbine_id": np.arange(1, len(x) + 1),
        "x_utm33n_m": np.round(e, 1),
        "y_utm33n_m": np.round(n, 1),
        "lat_deg": np.round(lat, 6),
        "lon_deg": np.round(lon, 6),
        "x_local_m": np.round(x, 1),
        "y_local_m": np.round(y, 1),
        "turbine_type": turbine_type,
    })
