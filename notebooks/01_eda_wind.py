# %% [markdown]
# # 01 — Eksploracja danych wiatrowych (EDA Wind)
# **Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**
#
# Ten notebook demonstruje:
# 1. Generowanie mock data dla Bałtyku Południowego
# 2. Statystyki i wizualizacje danych wiatrowych
# 3. Konwersję do natywnych obiektów FLORIS (TimeSeries, WindRose, WindTIRose)
# 4. Analizę sezonową
# 5. Porównanie z danymi z CSV (gdy dostępne)

# %% Importy
import sys
sys.path.insert(0, "..")  # Żeby importować z src/

import numpy as np
import matplotlib.pyplot as plt

from src.wind_data import WindDataLoader, BalticWindConfig

# Opcjonalnie — FLORIS (jeśli zainstalowany)
try:
    from floris import FlorisModel, WindRose, TimeSeries
    FLORIS_AVAILABLE = True
    print(f"FLORIS załadowany pomyślnie")
except ImportError:
    FLORIS_AVAILABLE = False
    print("FLORIS nie zainstalowany — używam samych danych bez symulacji wake.")

# %% [markdown]
# ## 1. Generowanie mock data

# %%
# Domyślna konfiguracja — Bałtyk Południowy
config = BalticWindConfig()
print(f"Konfiguracja:")
print(f"  Weibull k={config.weibull_k}, A={config.weibull_A} m/s")
print(f"  Kierunek dominujący: {config.dominant_direction}° (SW)")
print(f"  TI średnie: {config.ti_mean}")
print(f"  Hub height: {config.hub_height} m")
print(f"  Okres: {config.n_hours} godzin ({config.n_hours/8760:.0f} rok)")

# %%
# Generowanie danych
loader = WindDataLoader()
time_series = loader.generate_mock_data(config=config, seed=42)

print(f"\nLoader: {loader}")
print(f"\nTyp obiektu FLORIS: {type(time_series).__name__}")

# %% [markdown]
# ## 2. Statystyki

# %%
stats = loader.summary()

print("=" * 50)
print("STATYSTYKI DANYCH WIATROWYCH")
print("=" * 50)
print(f"\nLiczba rekordów: {stats['n_records']}")
print(f"\nPrędkość wiatru:")
print(f"  Średnia:  {stats['wind_speed']['mean']:.2f} m/s")
print(f"  Mediana:  {stats['wind_speed']['median']:.2f} m/s")
print(f"  Std:      {stats['wind_speed']['std']:.2f} m/s")
print(f"  Min/Max:  {stats['wind_speed']['min']:.1f} / {stats['wind_speed']['max']:.1f} m/s")
print(f"  Weibull k (estymacja): {stats['wind_speed']['weibull_k_est']:.2f}")
print(f"  Weibull A (estymacja): {stats['wind_speed']['weibull_A_est']:.1f} m/s")
print(f"\nKierunek wiatru:")
print(f"  Dominujący: {stats['wind_direction']['dominant']:.0f}°")
print(f"  Średni kołowy: {stats['wind_direction']['mean_circular']:.0f}°")
print(f"\nIntensywność turbulencji:")
print(f"  Średnia: {stats['turbulence_intensity']['mean']:.4f}")
print(f"  Std: {stats['turbulence_intensity']['std']:.4f}")

if "time_range" in stats:
    print(f"\nZakres czasowy:")
    print(f"  Od: {stats['time_range']['start']}")
    print(f"  Do: {stats['time_range']['end']}")
    print(f"  Dni: {stats['time_range']['duration_days']}")

# %% [markdown]
# ## 3. Wizualizacje

# %%
# Róża wiatrów
fig_rose = loader.plot_wind_rose(title="Róża wiatrów — Bałtyk Południowy (mock data)")
plt.savefig("../outputs/figures/wind_rose_mock.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Seria czasowa (4 panele)
fig_ts = loader.plot_time_series(title="Dane wiatrowe — Bałtyk Południowy (mock data)")
plt.savefig("../outputs/figures/time_series_mock.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Konwersja do obiektów FLORIS

# %%
# TimeSeries — surowe dane godzinowe
ts = loader.to_time_series()
print(f"TimeSeries: {type(ts).__name__}")

# WindRose — binowane z freq_table
wr = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)
print(f"WindRose: {type(wr).__name__}")

# WindTIRose — binowane z TI
wr_ti = loader.to_wind_ti_rose()
print(f"WindTIRose: {type(wr_ti).__name__}")

# %%
# Wbudowany plot FLORIS WindRose
if FLORIS_AVAILABLE:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 8))
    wr.plot(ax=ax)
    ax.set_title("WindRose — wbudowany plot FLORIS", pad=20)
    plt.savefig("../outputs/figures/wind_rose_floris.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 5. Analiza sezonowa

# %%
seasons = loader.seasonal_split()

fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "polar"})

for ax, (name, season_loader) in zip(axes.flatten(), seasons.items()):
    season_stats = season_loader.summary()
    season_loader.plot_wind_rose(
        ax=ax,
        title=f"{name}\n(śr. {season_stats['wind_speed']['mean']:.1f} m/s, n={season_stats['n_records']})"
    )

plt.suptitle("Róże wiatrów — rozbicie sezonowe", fontsize=16, fontweight="500", y=1.02)
plt.tight_layout()
plt.savefig("../outputs/figures/wind_rose_seasonal.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Porównanie konfiguracji wiatru
# Testujemy jak zmiana parametrów Weibulla wpływa na rozkład.

# %%
configs = {
    "Bałtyk typowy (k=2.1, A=9.5)": BalticWindConfig(),
    "Więcej wiatru (k=2.3, A=11.0)": BalticWindConfig(weibull_k=2.3, weibull_A=11.0),
    "Mniej wiatru (k=1.8, A=7.5)": BalticWindConfig(weibull_k=1.8, weibull_A=7.5),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ["#1e5c3a", "#c8531a", "#534AB7"]

for ax, (name, cfg), color in zip(axes, configs.items(), colors):
    temp_loader = WindDataLoader()
    temp_loader.generate_mock_data(config=cfg, seed=42)
    temp_stats = temp_loader.summary()

    ax.hist(
        temp_loader.wind_speeds, bins=50, density=True,
        alpha=0.7, color=color, edgecolor="white", linewidth=0.3,
    )
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Prędkość [m/s]")
    ax.set_ylabel("Gęstość")
    ax.axvline(
        temp_stats["wind_speed"]["mean"], color="black",
        linestyle="--", linewidth=1.5, label=f"Śr. = {temp_stats['wind_speed']['mean']:.1f} m/s"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Porównanie rozkładów — różne parametry Weibulla", fontsize=14, fontweight="500")
plt.tight_layout()
plt.savefig("../outputs/figures/weibull_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Quick test — FLORIS integration
# Jeśli FLORIS zainstalowany, uruchamiamy prostą symulację z 3 turbinami.

# %%
if FLORIS_AVAILABLE:
    from floris import FlorisModel
    from pathlib import Path
    import floris

    yaml_path = Path(floris.__file__).parent / "default_inputs.yaml"
    fmodel = FlorisModel(str(yaml_path))

    D = 240.0
    layout_x = [0.0, 7 * D, 14 * D]
    layout_y = [0.0, 0.0, 0.0]

    fmodel.set(
        layout_x=layout_x,
        layout_y=layout_y,
        turbine_type=["iea_15MW"],
        wind_data=wr,
    )

    fmodel.run()

    aep = fmodel.get_farm_AEP() / 1e9
    powers = fmodel.get_turbine_powers()

    print(f"AEP farmy (3 turbiny IEA 15MW): {aep:.2f} GWh")
    print(f"Moce turbin (przykładowy warunek): {powers[0] / 1e6} MW")

# %% [markdown]
# ## Następne kroki
# - [ ] Notebook 02: Definiowanie layoutów farmy (grid, staggered, circular)
# - [ ] Notebook 03: Porównanie modeli wake (Jensen vs GCH vs TurbOPark)
# - [ ] Podłączenie prawdziwych danych ERA5 po spotkaniu z prowadzącym
