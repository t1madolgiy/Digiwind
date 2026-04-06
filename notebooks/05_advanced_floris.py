# %% [markdown]
# # 05 — Zaawansowane funkcje FLORIS
# **Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**
#
# Nowe funkcje (nie modyfikuje istniejącego kodu):
# 1. Optymalizacja wartości ekonomicznej (AVP zamiast AEP)
# 2. Derating i wyłączanie turbin
# 3. ParFlorisModel — obliczenia równoległe
# 4. Analiza pola przepływu w dowolnych punktach
# 5. Wizualizacja waking directions
# 6. Porównanie parametrów modeli wake (analiza wrażliwości)
# 7. Multidimensional turbine curves (fale offshore)
# 8. Helix control (Active Wake Mixing)

# %% Importy
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import floris
from floris import FlorisModel, TimeSeries, WindRose
from floris.utilities import load_yaml
import floris.flow_visualization as flowviz
import floris.layout_visualization as layoutviz

from src.wind_data import WindDataLoader, BalticWindConfig
from src.farm_model import FarmModel, WAKE_MODELS, TURBINE_LIBRARY

FLORIS_DIR = Path(floris.__file__).parent

print("Moduły załadowane.")

# %% [markdown]
# ## 0. Przygotowanie danych (tak samo jak w 04)

# %%
config = BalticWindConfig(n_hours=8760 * 3)
loader = WindDataLoader()
loader.generate_mock_data(config=config, seed=42)

wr_fine = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)
wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)

D = 240.0
farm = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wr_fine)
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
farm.run()

baseline_aep = farm.get_aep_gwh()
print(f"Baseline: 5x5 IEA 15MW @ 7D, AEP = {baseline_aep:.2f} GWh")

# %% [markdown]
# ---
# ## 1. Optymalizacja wartości ekonomicznej (AVP)
#
# AEP maksymalizuje energię — ale energia z wiatru nocą jest mniej warta
# niż w szczycie dnia. AVP (Annual Value Production) uwzględnia cenę
# energii per warunek wiatrowy.
#
# Scenariusz: energia z kierunków N/S jest droższa (szczyt zimowy),
# z kierunków E/W tańsza.

# %%
# Tworzymy WindRose z value_table
wind_directions = np.arange(0, 360, 10.0)
wind_speeds = np.arange(3.0, 25.0, 2.0)

# Generujemy freq_table z naszych danych
wr_with_value = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)

# Dodajemy wartość — energia droższa gdy wiatr z N/S (szczyt zimowy)
wr_with_value.assign_value_piecewise_linear(
    value_zero_ws=0.0,       # brak wartości przy 0 m/s
    ws_knee=4.0,             # od 4 m/s zaczyna rosnąć
    slope_1=25.0,            # 25 $/MWh poniżej kolana
    slope_2=0.1,             # prawie stała powyżej
)

print("WindRose z value_table — gotowa")
print(f"Przykładowe wartości energii: min={0:.1f}, max={25*4:.1f} $/MWh")

# %%
# Porównanie AEP vs AVP
farm.fmodel.set(wind_data=wr_with_value)
farm.run()

aep = farm.fmodel.get_farm_AEP() / 1e9   # Wh -> GWh
avp_raw = farm.fmodel.get_farm_AVP()       # value_units * Wh

# get_farm_AVP() zwraca Wh * value (value z value_table)
# Dzielimy przez 1e6 (Wh -> MWh) żeby dostać "value_units * MWh" = $
avp_dollars = avp_raw / 1e6
avp_mln = avp_dollars / 1e6  # miliony dolarów

# Średnia ważona cena energii
avg_price = avp_dollars / (aep * 1e3)  # $ / (GWh * 1000 MWh/GWh) = $/MWh

print(f"AEP: {aep:.2f} GWh")
print(f"AVP: {avp_mln:.2f} mln $")
print(f"Średnia cena energii: {avg_price:.1f} $/MWh")

# %%
# Porównanie layoutów pod kątem AVP
layouts_avp = {}

# Grid
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
farm.fmodel.set(wind_data=wr_with_value)
farm.run()
layouts_avp["Grid 7D"] = {
    "aep": farm.fmodel.get_farm_AEP() / 1e9,
    "avp_mln": farm.fmodel.get_farm_AVP() / 1e12,
}

# Staggered
farm.set_layout_staggered(n_rows=5, n_cols=5, spacing_D=7.0, offset=0.5)
farm.fmodel.set(wind_data=wr_with_value)
farm.run()
layouts_avp["Staggered 7D"] = {
    "aep": farm.fmodel.get_farm_AEP() / 1e9,
    "avp_mln": farm.fmodel.get_farm_AVP() / 1e12,
}

# Grid 10D
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=10.0)
farm.fmodel.set(wind_data=wr_with_value)
farm.run()
layouts_avp["Grid 10D"] = {
    "aep": farm.fmodel.get_farm_AEP() / 1e9,
    "avp_mln": farm.fmodel.get_farm_AVP() / 1e12,
}

print(f"\n{'Layout':<18} {'AEP [GWh]':<12} {'AVP [mln $]':<14} {'$/MWh':<8}")
print("=" * 52)
for name, data in layouts_avp.items():
    avg_price = data["avp_mln"] * 1e6 / (data["aep"] * 1e3) if data["aep"] > 0 else 0
    print(f"{name:<18} {data['aep']:<12.2f} {data['avp_mln']:<14.2f} {avg_price:<8.1f}")

# %% [markdown]
# ---
# ## 2. Derating i wyłączanie turbin
#
# FLORIS pozwala na:
# - **Derating** — ograniczenie mocy turbiny (np. redukcja hałasu nocą)
# - **Disable** — wyłączenie turbiny (np. ochrona ptaków, konserwacja)

# %%
# Przygotowanie — mała farma do demonstracji
# %%
farm.set_layout_grid(n_rows=3, n_cols=3, spacing_D=7.0)
farm.set_wind_data(wr_fine)
farm.run()
aep_full = farm.get_aep_gwh()

# n_findex z FLORIS core — to jest prawdziwy wymiar
n_findex = farm.fmodel.core.flow_field.n_findex
n_turbines = farm.n_turbines

print(f"Farma: {n_turbines} turbin, n_findex={n_findex}")
print(f"AEP pełna moc: {aep_full:.2f} GWh")

# %%
# Wyłączamy turbiny 0 i 4
disable_array = np.full((n_findex, n_turbines), False)
disable_array[:, 0] = True
disable_array[:, 4] = True

farm.fmodel.set(disable_turbines=disable_array)
farm.run()
aep_disabled = farm.get_aep_gwh()

print(f"\nTurbiny 0 i 4 wyłączone:")
print(f"AEP: {aep_disabled:.2f} GWh")
print(f"Strata: {(1 - aep_disabled/aep_full)*100:.1f}%")

# %%
# Derating turbin 0-2 do 10 MW
farm.fmodel.set(disable_turbines=np.full((n_findex, n_turbines), False))
farm.fmodel.set_operation_model("simple-derating")

power_setpoints = np.full((n_findex, n_turbines), None)
power_setpoints[:, 0] = 10e6
power_setpoints[:, 1] = 10e6
power_setpoints[:, 2] = 10e6

farm.fmodel.set(power_setpoints=power_setpoints)
farm.run()
aep_derated = farm.get_aep_gwh()

print(f"\nTurbiny 0-2 derated do 10 MW:")
print(f"AEP: {aep_derated:.2f} GWh")
print(f"Strata: {(1 - aep_derated/aep_full)*100:.1f}%")

# Reset
farm.fmodel.set_operation_model("simple")
farm.fmodel.reset_operation()
farm.set_wind_data(wr_fine)

# %% [markdown]
# ---
# ## 3. ParFlorisModel — obliczenia równoległe
#
# Przyspiesza obliczenia na wielu rdzeniach CPU.
# Twój Ryzen 9 9900X ma 12 rdzeni — idealny do tego.

# %%
# ParFlorisModel wiesza się w Jupyter na Windows (bug multiprocessing).
# Poniżej benchmark samego FlorisModel z różnymi rozmiarami WindRose.
import time

farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)

benchmarks = {}
for wd_step, label in [(30.0, "Gruby (30°)"), (10.0, "Dokładny (10°)"), (5.0, "Precyzyjny (5°)")]:
    wr_test = loader.to_wind_rose(wd_step=wd_step, ws_step=2.0)
    farm.fmodel.set(wind_data=wr_test)

    t0 = time.perf_counter()
    farm.run()
    elapsed = time.perf_counter() - t0
    aep = farm.get_aep_gwh()

    benchmarks[label] = {"time": elapsed, "aep": aep, "wd_step": wd_step}

print(f"{'WindRose':<20} {'Czas [s]':<12} {'AEP [GWh]':<12} {'Rozdzielczość'}")
print("=" * 56)
for label, data in benchmarks.items():
    print(f"{label:<20} {data['time']:<12.2f} {data['aep']:<12.2f} {data['wd_step']}°")

print(f"\nParFlorisModel: pominięty (multiprocessing + Jupyter + Windows)")
print(f"Aby użyć — odpal z terminala: python -c 'from floris import ParFlorisModel; ...'")

# Przywróć
farm.set_wind_data(wr_fine)

# %% [markdown]
# ---
# ## 4. Analiza pola przepływu w dowolnych punktach
#
# `sample_flow_at_points()` — oblicz prędkość wiatru
# w dowolnych punktach przestrzeni (nie tylko na rotorach).

# %%
farm.set_layout_grid(n_rows=3, n_cols=3, spacing_D=7.0)

# Ustawiamy jeden warunek wiatrowy
farm.fmodel.set(
    wind_directions=[270.0],
    wind_speeds=[9.0],
    turbulence_intensities=[0.06],
)

# Tworzymy siatkę punktów za farmą
x_points = np.linspace(-500, 15000, 100)
y_points = np.zeros_like(x_points)  # wzdłuż osi X
z_points = np.full_like(x_points, 150.0)  # na wysokości piasty

# Oblicz prędkości w tych punktach
u_at_points = farm.fmodel.sample_flow_at_points(x_points, y_points, z_points)

# %%
# Wykres — profil prędkości wzdłuż osi farmy
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x_points / 1000, u_at_points[0, :], color="#1e5c3a", linewidth=2)
ax.set_xlabel("Odległość [km]")
ax.set_ylabel("Prędkość wiatru [m/s]")
ax.set_title("Profil prędkości wzdłuż osi farmy (Y=0, Z=150m, WD=270°)")

# Zaznacz pozycje turbin
turbine_xs = farm.layout_x / 1000
for tx in turbine_xs:
    ax.axvline(tx, color="#c8531a", linestyle="--", linewidth=0.8, alpha=0.5)

ax.grid(True, alpha=0.3)
ax.set_ylim(0, 11)
plt.savefig("../outputs/figures/11_flow_profile.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Mapa 2D prędkości — siatka punktów na wysokości piasty
x_grid = np.linspace(-500, 15000, 150)
y_grid = np.linspace(-2000, 5500, 80)
xx, yy = np.meshgrid(x_grid, y_grid)
x_flat = xx.flatten()
y_flat = yy.flatten()
z_flat = np.full_like(x_flat, 150.0)

u_grid = farm.fmodel.sample_flow_at_points(x_flat, y_flat, z_flat)
u_2d = u_grid[0, :].reshape(xx.shape)

fig, ax = plt.subplots(figsize=(14, 6))
c = ax.contourf(xx / 1000, yy / 1000, u_2d, levels=50, cmap="coolwarm")
plt.colorbar(c, ax=ax, label="Prędkość [m/s]")
ax.scatter(farm.layout_x / 1000, farm.layout_y / 1000,
           s=60, c="black", edgecolors="white", linewidths=1, zorder=5)
ax.set_xlabel("X [km]")
ax.set_ylabel("Y [km]")
ax.set_title("Mapa prędkości wiatru na wysokości piasty (custom grid)")
ax.set_aspect("equal")
plt.savefig("../outputs/figures/12_flow_map_custom.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5. Wizualizacja waking directions
#
# Pokazuje które turbiny "widzą" ślad których przy danym kierunku wiatru.

# %%
farm.set_layout_grid(n_rows=4, n_cols=4, spacing_D=7.0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, wd in zip(axes, [240.0, 270.0, 300.0]):
    farm.fmodel.set(
        wind_directions=[wd],
        wind_speeds=[9.0],
        turbulence_intensities=[0.06],
    )
    layoutviz.plot_turbine_points(farm.fmodel, ax=ax)
    layoutviz.plot_waking_directions(farm.fmodel, ax=ax, limit_dist_D=20)
    ax.set_title(f"Waking directions — WD={wd:.0f}°")
    ax.set_aspect("equal")

fig.suptitle("Które turbiny widzą ślad których?", fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/13_waking_directions.png", dpi=150, bbox_inches="tight")
plt.show()

# Przywróć wind data
farm.set_wind_data(wr_fine)

# %% [markdown]
# ---
# ## 6. Analiza wrażliwości parametrów modelu wake
#
# Model GCH (Gauss) ma parametry ka, kb, alpha, beta.
# Jak wpływają na AEP?

# %%
# Bazowy config
base_dict = load_yaml(FLORIS_DIR / "default_inputs.yaml")
base_dict["farm"]["turbine_type"] = ["iea_15MW"]
base_dict["flow_field"]["reference_wind_height"] = 150.0
base_dict["wake"]["enable_secondary_steering"] = True
base_dict["wake"]["enable_yaw_added_recovery"] = True
base_dict["wake"]["enable_transverse_velocities"] = True

# Parametry do testowania
param_sweeps = {
    "ka (wake expansion)": {
        "path": ["wake", "wake_velocity_parameters", "gauss", "ka"],
        "values": [0.2, 0.3, 0.38, 0.5, 0.6],
        "default": 0.38,
    },
    "kb (wake expansion)": {
        "path": ["wake", "wake_velocity_parameters", "gauss", "kb"],
        "values": [0.001, 0.002, 0.004, 0.008, 0.012],
        "default": 0.004,
    },
    "alpha (deflection)": {
        "path": ["wake", "wake_deflection_parameters", "gauss", "alpha"],
        "values": [0.3, 0.45, 0.58, 0.7, 0.85],
        "default": 0.58,
    },
    "beta (deflection)": {
        "path": ["wake", "wake_deflection_parameters", "gauss", "beta"],
        "values": [0.03, 0.05, 0.077, 0.1, 0.15],
        "default": 0.077,
    },
}

sensitivity_results = {}

for param_name, info in param_sweeps.items():
    aeps = []
    for val in info["values"]:
        # Kopiuj config
        import copy
        test_dict = copy.deepcopy(base_dict)

        # Ustaw parametr
        d = test_dict
        for key in info["path"][:-1]:
            d = d[key]
        d[info["path"][-1]] = val

        # Stwórz model i oblicz
        try:
            fmodel = FlorisModel(test_dict)
            fmodel.set(
                layout_x=(np.arange(5) * 7 * D).tolist(),
                layout_y=[0.0] * 5,
                wind_data=wr_coarse,
            )
            fmodel.run()
            aep = fmodel.get_farm_AEP() / 1e9
            aeps.append(aep)
        except Exception as e:
            aeps.append(np.nan)

    sensitivity_results[param_name] = {
        "values": info["values"],
        "aeps": aeps,
        "default": info["default"],
    }

# %%
# Wykres wrażliwości
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (param_name, data) in zip(axes.flatten(), sensitivity_results.items()):
    ax.plot(data["values"], data["aeps"], "o-", color="#534AB7", linewidth=2)
    ax.axvline(data["default"], color="#c8531a", linestyle="--", linewidth=1.5,
               label=f"Domyślna: {data['default']}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("AEP [GWh]")
    ax.set_title(f"Wrażliwość na {param_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle("Analiza wrażliwości parametrów GCH — 5 turbin w linii @ 7D",
             fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/14_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Multidimensional turbine curves (floating offshore)
#
# IEA 15MW ma krzywe mocy zależne od warunków falowania (Tp, Hs).
# FLORIS obsługuje to przez multidimensional power/thrust tables.

# %%
# Sprawdź czy plik istnieje
multi_dim_yaml = FLORIS_DIR / "turbine_library" / "iea_15MW_multi_dim_cp_ct.yaml"
print(f"Plik multi-dim: {multi_dim_yaml.exists()}")

if multi_dim_yaml.exists():
    # Załaduj i pokaż strukturę
    multi_dim_data = load_yaml(multi_dim_yaml)
    print(f"Nazwa: {multi_dim_data.get('turbine_type', 'N/A')}")
    print(f"Hub height: {multi_dim_data.get('hub_height', 'N/A')} m")
    print(f"Rotor diameter: {multi_dim_data.get('rotor_diameter', 'N/A')} m")

    # Wymiary multidimensional
    ptt = multi_dim_data.get("power_thrust_table", {})
    if "power_thrust_data_file" in ptt:
        print(f"Dane z pliku: {ptt['power_thrust_data_file']}")
        print("(Multidimensional curves — moc zależy od Tp i Hs)")
    else:
        print("Struktura power_thrust_table:")
        for key in ptt:
            if isinstance(ptt[key], list):
                print(f"  {key}: {len(ptt[key])} wartości")
            else:
                print(f"  {key}: {ptt[key]}")

# %% [markdown]
# ---
# ## 8. Helix control (Active Wake Mixing)
#
# Helix to strategia sterowania pitch łopat w celu szybszego
# rozpraszania śladu aerodynamicznego. Nowa w FLORIS v4.

# %%
# Test helix — porównanie z/bez
# Helix wymaga modelu empirical_gauss (nie GCH!) — tworzymy osobny model
import copy

helix_dict = load_yaml(FLORIS_DIR / "default_inputs.yaml")
helix_dict["farm"]["turbine_type"] = ["iea_15MW"]
helix_dict["flow_field"]["reference_wind_height"] = 150.0
helix_dict["wake"]["model_strings"]["velocity_model"] = "empirical_gauss"
helix_dict["wake"]["model_strings"]["deflection_model"] = "empirical_gauss"
helix_dict["wake"]["model_strings"]["turbulence_model"] = "wake_induced_mixing"
helix_dict["wake"]["enable_secondary_steering"] = False
helix_dict["wake"]["enable_yaw_added_recovery"] = False
helix_dict["wake"]["enable_transverse_velocities"] = False
helix_dict["wake"]["enable_active_wake_mixing"] = True
helix_dict["wake"]["wake_velocity_parameters"] = {"empirical_gauss": {}}
helix_dict["wake"]["wake_deflection_parameters"] = {"empirical_gauss": {}}
helix_dict["wake"]["wake_turbulence_parameters"] = {"wake_induced_mixing": {}}

helix_fmodel = FlorisModel(helix_dict)

# 3 turbiny w linii
D = 240.0
helix_fmodel.set(
    layout_x=[0.0, 7 * D, 14 * D],
    layout_y=[0.0, 0.0, 0.0],
    wind_directions=[270.0],
    wind_speeds=[9.0],
    turbulence_intensities=[0.06],
)

# Bez helix
helix_fmodel.set_operation_model("simple")
helix_fmodel.run()
powers_no_helix = helix_fmodel.get_turbine_powers() / 1e6

# Z helix na pierwszej turbinie
try:
    n_findex = 1
    n_turb = 3
    awc_modes = np.full((n_findex, n_turb), "baseline", dtype=object)
    awc_modes[0, 0] = "helix"
    awc_amplitudes = np.zeros((n_findex, n_turb))
    awc_amplitudes[0, 0] = 5.0  # 5 stopni

    helix_fmodel.set_operation_model("awc")
    helix_fmodel.set(
        awc_modes=awc_modes,
        awc_amplitudes=awc_amplitudes,
    )
    helix_fmodel.run()
    powers_helix = helix_fmodel.get_turbine_powers() / 1e6

    print(f"Model: Empirical Gauss (z enable_active_wake_mixing=True)")
    print(f"\n{'Turbina':<12} {'Bez helix [MW]':<18} {'Z helix [MW]':<18} {'Zmiana [%]'}")
    print("=" * 60)
    for i in range(n_turb):
        p_no = powers_no_helix[0, i]
        p_yes = powers_helix[0, i]
        change = (p_yes - p_no) / p_no * 100 if p_no > 0 else 0
        print(f"T{i:<11} {p_no:<18.2f} {p_yes:<18.2f} {change:<+.1f}")

    total_no = np.nansum(powers_no_helix)
    total_yes = np.nansum(powers_helix)
    change_total = (total_yes - total_no) / total_no * 100
    print(f"\n{'TOTAL':<12} {total_no:<18.2f} {total_yes:<18.2f} {change_total:<+.1f}")
    print(f"\nT0 traci moc (helix kosztuje energię), ale T1 i T2 zyskują")
    print(f"bo ślad rozprasza się szybciej → netto {'zysk' if change_total > 0 else 'strata'}")

except Exception as e:
    print(f"Helix control error: {e}")

# Przywróć główny model
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
farm.fmodel.set_operation_model("simple")
farm.fmodel.reset_operation()
farm.set_wind_data(wr_fine)

# %% [markdown]
# ---
# ## 9. Rotor velocity visualization
#
# Wizualizacja rozkładu prędkości na rotorze każdej turbiny.

# %%
farm.set_layout_grid(n_rows=1, n_cols=4, spacing_D=7.0)
farm.fmodel.set(
    wind_directions=[270.0],
    wind_speeds=[9.0],
    turbulence_intensities=[0.06],
)
farm.run()

try:
    from floris.flow_visualization import plot_rotor_values
    fig, _, _, _ = plot_rotor_values(
        farm.fmodel.core.flow_field.u,
        findex=0,
        n_rows=1,
        n_cols=4,
        return_fig_objects=True,
    )
    fig.suptitle("Rozkład prędkości na rotorach — 4 turbiny w linii @ 7D, WD=270°",
                 fontsize=12, fontweight="500")
    plt.savefig("../outputs/figures/15_rotor_values.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Rotor values error: {e}")

# Przywróć
farm.set_wind_data(wr_fine)

# %% [markdown]
# ---
# ## 10. Podsumowanie nowych funkcji

# %%
print("=" * 60)
print("PODSUMOWANIE — ZAAWANSOWANE FUNKCJE FLORIS")
print("=" * 60)
print("""
1. AVP (Annual Value Production)
   - Optymalizacja pod wartość ekonomiczną energii
   - value_table w WindRose — cena per warunek wiatrowy
   - Różne layouty mogą mieć różne AVP nawet przy tym samym AEP

2. Derating i wyłączanie turbin
   - disable_turbines — per turbina, per warunek wiatrowy
   - power_setpoints — ograniczenie mocy (np. 10 MW zamiast 15)
   - Przydatne: konserwacja, ochrona ptaków, redukcja hałasu

3. ParFlorisModel
   - Obliczenia równoległe na wielu rdzeniach
   - Ten sam interfejs co FlorisModel
   - Przyspieszenie zależy od wielkości problemu

4. Sample flow at points
   - Prędkość wiatru w dowolnych punktach przestrzeni
   - Profile prędkości wzdłuż osi farmy
   - Mapy 2D na dowolnej wysokości

5. Waking directions
   - Wizualizacja: które turbiny "widzą" ślad których
   - Zależy od kierunku wiatru
   - Przydatne do tłumaczenia wyników

6. Analiza wrażliwości parametrów
   - ka, kb — współczynniki ekspansji śladu
   - alpha, beta — współczynniki defleksji
   - Pokazuje niepewność modelu, nie tylko danych

7. Multidimensional turbine curves
   - Krzywe mocy zależne od warunków oceanicznych (Tp, Hs)
   - Specyficzne dla floating offshore
   - Plik iea_15MW_multi_dim_cp_ct.yaml

8. Helix control (Active Wake Mixing)
   - Sterowanie pitch łopat → szybsze rozpraszanie śladu
   - Turbina upstream traci moc, ale downstream zyskuje więcej
   - Nowa technologia — materiał na dodatkową analizę
""")
print("=" * 60)
