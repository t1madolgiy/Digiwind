# %% [markdown]
# # 04 — Pełna symulacja farmy wiatrowej offshore
# **Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**
# **Bałtyk Południowy (~54.5°N, 16.5°E)**
#
# Kompletna analiza:
# 1. Realistyczne dane wiatrowe (Weibull offshore, sezonowość)
# 2. Farma 5x5 (25 turbin IEA 15MW = 375 MW)
# 3. Porównanie 3 layoutów (grid, staggered, zoptymalizowany)
# 4. Porównanie modeli wake (Jensen, GCH, TurbOPark, CC)
# 5. Porównanie turbin (5MW, 10MW, 15MW, 22MW)
# 6. Optymalizacja layoutu (Scipy)
# 7. Optymalizacja yaw (wake steering)
# 8. Analiza AEP: roczna, sezonowa, miesięczna
# 9. Analiza niepewności
# 10. Pełny eksport danych dla grupy

# %% Importy
import sys
sys.path.insert(0, "..")

import warnings
warnings.filterwarnings("ignore")  # FLORIS warnings spam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.wind_data import WindDataLoader, BalticWindConfig
from src.farm_model import FarmModel, WAKE_MODELS, TURBINE_LIBRARY
from src.optimizer import Optimizer, OptimizationResult
from src.aep_calculator import AEPCalculator

print("Moduły załadowane.")
print(f"Modele wake: {list(WAKE_MODELS.keys())}")
print(f"Turbiny: {list(TURBINE_LIBRARY.keys())}")

# %% [markdown]
# ---
# ## 1. Dane wiatrowe — Bałtyk Południowy
# Parametry Weibulla z literatury dla polskiej strefy offshore.
# Zimą silniejsze wiatry z SW/W, latem słabsze z NW.

# %%
config = BalticWindConfig(
    weibull_k=2.1,            # typowy kształt offshore
    weibull_A=9.5,            # skala — ok. 9-10 m/s na 150m
    dominant_direction=240.0,  # SW dominujący
    direction_spread=60.0,
    ti_mean=0.06,             # niski TI offshore
    ti_std=0.015,
    hub_height=150.0,         # IEA 15MW
    n_hours=8760 * 3,         # 3 lata danych
)

loader = WindDataLoader()
loader.generate_mock_data(config=config, seed=42)

stats = loader.summary()
print(f"Dane: {stats['n_records']} godzin ({stats['n_records']/8760:.0f} lat)")
print(f"Prędkość:  śr={stats['wind_speed']['mean']:.1f} m/s, "
      f"Weibull k={stats['wind_speed']['weibull_k_est']:.2f}, "
      f"A={stats['wind_speed']['weibull_A_est']:.1f} m/s")
print(f"Kierunek:  dominujący {stats['wind_direction']['dominant']:.0f}°")
print(f"TI:        śr={stats['turbulence_intensity']['mean']:.4f}")

# %%
# WindRose — dwie rozdzielczości
wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)   # optymalizacja
wr_fine = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)     # finalne AEP

# Wizualizacje danych
fig = loader.plot_wind_rose(title="Róża wiatrów — Bałtyk Pd. (3 lata)")
plt.savefig("../outputs/figures/00_wind_rose.png", dpi=150, bbox_inches="tight")
plt.show()

fig = loader.plot_time_series(title="Dane wiatrowe — Bałtyk Pd. (3 lata)")
plt.savefig("../outputs/figures/00_time_series.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 2. Definicja farmy — 25 turbin IEA 15MW
# Obszar: ~11 km × 11 km (rozstaw 7D = 1680m, 5 rzędów)
# Moc zainstalowana: 25 × 15 MW = 375 MW

# %%
D = 240.0  # średnica IEA 15MW
SPACING_D = 7.0
N_ROWS, N_COLS = 5, 5
N_TURBINES = N_ROWS * N_COLS

farm_area_km2 = ((N_COLS - 1) * SPACING_D * D / 1000) * ((N_ROWS - 1) * SPACING_D * D / 1000)

print(f"Turbina: IEA 15 MW (D={D}m, hub={150}m)")
print(f"Layout: {N_ROWS}x{N_COLS} = {N_TURBINES} turbin")
print(f"Rozstaw: {SPACING_D}D = {SPACING_D * D:.0f} m")
print(f"Moc zainstalowana: {N_TURBINES * 15} MW")
print(f"Obszar farmy: ~{farm_area_km2:.1f} km²")

# %% [markdown]
# ---
# ## 3. Porównanie 3 layoutów

# %%
farm = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wr_fine)

layouts = {}

# Layout 1: Grid regularny
farm.set_layout_grid(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D)
farm.run()
layouts["Grid regularny"] = {
    "x": farm.layout_x.copy(),
    "y": farm.layout_y.copy(),
    "aep": farm.get_aep_gwh(),
    "wake_loss": farm.get_wake_losses_percent(),
}

# Layout 2: Staggered
farm.set_layout_staggered(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D, offset=0.5)
farm.set_wind_data(wr_fine)
farm.run()
layouts["Staggered"] = {
    "x": farm.layout_x.copy(),
    "y": farm.layout_y.copy(),
    "aep": farm.get_aep_gwh(),
    "wake_loss": farm.get_wake_losses_percent(),
}

# Layout 3: Circular (2 pierścienie + centrum)
farm.set_layout_circular(n_turbines=12, radius_D=12.0, n_rings=2)
farm.set_wind_data(wr_fine)
farm.run()
layouts["Circular"] = {
    "x": farm.layout_x.copy(),
    "y": farm.layout_y.copy(),
    "aep": farm.get_aep_gwh(),
    "wake_loss": farm.get_wake_losses_percent(),
}

# Tabela porównawcza
print(f"\n{'Layout':<20} {'N turbin':<10} {'AEP [GWh]':<12} {'Wake loss [%]':<14}")
print("=" * 56)
for name, data in layouts.items():
    n = len(data["x"])
    print(f"{name:<20} {n:<10} {data['aep']:<12.2f} {data['wake_loss']:<14.1f}")

# %%
# Wizualizacja layoutów
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (name, data) in zip(axes, layouts.items()):
    ax.scatter(data["x"], data["y"], s=60, c="#1e5c3a", edgecolors="white", linewidths=1)
    ax.set_title(f"{name}\nAEP={data['aep']:.1f} GWh | Wake loss={data['wake_loss']:.1f}%")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
fig.suptitle("Porównanie layoutów — IEA 15MW | GCH", fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/01_layout_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 4. Flow field — wizualizacja wake

# %%
# Grid regularny — wiatr z SW (dominujący)
farm.set_layout_grid(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

farm.plot_flow_field(wind_direction=240.0, wind_speed=9.0, ax=axes[0],
                     title="Wiatr z SW (240°) — dominujący")
farm.plot_flow_field(wind_direction=270.0, wind_speed=9.0, ax=axes[1],
                     title="Wiatr z W (270°)")

fig.suptitle("Flow field — 5x5 grid IEA 15MW @ 7D | GCH", fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/02_flow_field.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5. Porównanie modeli wake

# %%
farm.set_layout_grid(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D)
farm.set_wind_data(wr_fine)

wake_results = {}
for model_name in ["jensen", "gch", "turbopark", "cc"]:
    farm.switch_wake_model(model_name)
    farm.run()
    wake_results[model_name] = {
        "aep": farm.get_aep_gwh(),
        "wake_loss": farm.get_wake_losses_percent(),
    }

# Przywróć GCH
farm.switch_wake_model("gch")

print(f"\n{'Model':<18} {'AEP [GWh]':<12} {'Wake loss [%]':<14}")
print("=" * 44)
for name, data in wake_results.items():
    print(f"{name.upper():<18} {data['aep']:<12.2f} {data['wake_loss']:<14.1f}")

# %%
# Wykres porównawczy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
names = list(wake_results.keys())
aeps = [d["aep"] for d in wake_results.values()]
losses = [d["wake_loss"] for d in wake_results.values()]

colors = ["#534AB7", "#1e5c3a", "#c8531a", "#7a3b00"]
axes[0].bar([n.upper() for n in names], aeps, color=colors)
axes[0].set_ylabel("AEP [GWh]")
axes[0].set_title("AEP per model wake")
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar([n.upper() for n in names], losses, color=colors)
axes[1].set_ylabel("Wake losses [%]")
axes[1].set_title("Straty wake per model")
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Porównanie modeli wake — {N_TURBINES} × IEA 15MW @ {SPACING_D}D",
             fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/03_wake_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6. Porównanie turbin

# %%
farm.switch_wake_model("gch")
turbine_results = farm.compare_turbines(
    turbines=["nrel_5MW", "iea_10MW", "iea_15MW", "iea_22MW"],
    spacing_D=SPACING_D,
    n_rows=N_ROWS,
    n_cols=N_COLS,
)

print(f"\n{'Turbina':<15} {'D [m]':<8} {'Moc [MW]':<10} {'AEP [GWh]':<12} "
      f"{'Wake loss':<12} {'CF [%]':<8}")
print("=" * 70)
for name, data in turbine_results.items():
    if "error" not in data:
        print(
            f"{data['name']:<15} {data['diameter']:<8.0f} "
            f"{data['rated_power_mw']:<10.0f} {data['aep_gwh']:<12.2f} "
            f"{data['wake_loss_pct']:<12.1f} {data['capacity_factor']:<8.1f}"
        )

# %%
# Wykres
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
valid = {k: v for k, v in turbine_results.items() if "error" not in v}
names = [v["name"] for v in valid.values()]
aeps = [v["aep_gwh"] for v in valid.values()]
losses = [v["wake_loss_pct"] for v in valid.values()]
cfs = [v["capacity_factor"] for v in valid.values()]
colors = ["#534AB7", "#1e5c3a", "#c8531a", "#7a3b00"]

axes[0].barh(names, aeps, color=colors[:len(names)])
axes[0].set_xlabel("AEP [GWh]")
axes[0].set_title("Roczna produkcja energii")

axes[1].barh(names, losses, color=colors[:len(names)])
axes[1].set_xlabel("Wake losses [%]")
axes[1].set_title("Straty wake")

axes[2].barh(names, cfs, color=colors[:len(names)])
axes[2].set_xlabel("Capacity factor [%]")
axes[2].set_title("Współczynnik wykorzystania")

fig.suptitle(f"Porównanie turbin — {N_ROWS}x{N_COLS} grid @ {SPACING_D}D | GCH",
             fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/04_turbine_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Spacing sweep — optymalna odległość

# %%
farm.switch_turbine("iea_15MW")
farm.set_wind_data(wr_fine)

sweep = farm.spacing_sweep(
    spacings_D=np.arange(5.0, 12.5, 1.0),
    n_rows=N_ROWS,
    n_cols=N_COLS,
)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(sweep["spacings_D"], sweep["aep_gwh"], "o-", color="#1e5c3a", linewidth=2, label="AEP")
ax1.set_xlabel("Rozstaw [×D]")
ax1.set_ylabel("AEP [GWh]", color="#1e5c3a")

ax2 = ax1.twinx()
ax2.plot(sweep["spacings_D"], sweep["wake_loss_pct"], "s--", color="#c8531a", linewidth=2, label="Wake losses")
ax2.set_ylabel("Wake losses [%]", color="#c8531a")

fig.suptitle(f"Spacing sweep — {N_TURBINES} × IEA 15MW | GCH", fontsize=14, fontweight="500")
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
ax1.grid(True, alpha=0.3)
plt.savefig("../outputs/figures/05_spacing_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

best_idx = np.argmax(sweep["aep_gwh"])
print(f"Optymalny rozstaw: {sweep['spacings_D'][best_idx]:.0f}D")
print(f"  AEP: {sweep['aep_gwh'][best_idx]:.2f} GWh")
print(f"  Wake losses: {sweep['wake_loss_pct'][best_idx]:.1f}%")

# %% [markdown]
# ---
# ## 8. Optymalizacja layoutu (Scipy)
# Startujemy z gridu 4x4 (16 turbin) na grubej WindRose — ~2-5 min.
# Potem liczymy finalne AEP na dokładnej WindRose.

# %%
# 4x4 = 16 turbin — kompromis między realizmem a czasem
farm_opt = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wr_coarse)
farm_opt.set_layout_grid(n_rows=4, n_cols=4, spacing_D=SPACING_D)
farm_opt.run()
print(f"Przed optymalizacją (coarse): {farm_opt.get_aep_gwh():.2f} GWh")

opt = Optimizer(farm_opt)
opt.set_boundaries_from_layout(margin_D=3.0)
opt.set_min_distance(min_dist_D=3.0)

# %%
result = opt.optimize_layout_scipy(maxiter=30)
print(f"\nScipy: {result.initial_aep_gwh:.2f} -> {result.optimized_aep_gwh:.2f} GWh "
      f"(+{result.aep_improvement_pct:.2f}%) | {result.elapsed_seconds:.1f}s")

# Finalne AEP na dokładnej WindRose
opt.apply_result(result)
farm_opt.set_wind_data(wr_fine)
farm_opt.run()
aep_optimized_fine = farm_opt.get_aep_gwh()
print(f"AEP po optymalizacji (fine): {aep_optimized_fine:.2f} GWh")

fig = opt.plot_optimization_result(result)
plt.savefig("../outputs/figures/06_optimization.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 9. Yaw optimization (wake steering)

# %%
farm_yaw = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wr_coarse)
farm_yaw.set_layout_grid(n_rows=4, n_cols=4, spacing_D=SPACING_D)

opt_yaw = Optimizer(farm_yaw)
opt_yaw.set_boundaries_from_layout(margin_D=3.0)

try:
    result_yaw = opt_yaw.optimize_yaw(yaw_min=0.0, yaw_max=25.0)
    print(f"Yaw: {result_yaw.initial_aep_gwh:.2f} -> {result_yaw.optimized_aep_gwh:.2f} GWh "
          f"(+{result_yaw.aep_improvement_pct:.2f}%) | {result_yaw.elapsed_seconds:.1f}s")

    fig = opt_yaw.plot_yaw_result(result_yaw)
    plt.savefig("../outputs/figures/07_yaw_optimization.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Yaw error: {e}")
    result_yaw = None

# %% [markdown]
# ---
# ## 10. Analiza AEP — sezonowa i miesięczna

# %%
farm.set_layout_grid(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D)
farm.switch_wake_model("gch")
farm.switch_turbine("iea_15MW")
farm.set_wind_data(wr_fine)
farm.run()

calc = AEPCalculator(farm)
summary = calc.compute_aep_summary()

print(f"{'='*50}")
print(f"PODSUMOWANIE ROCZNE")
print(f"{'='*50}")
print(f"Turbina:            {summary['turbine']}")
print(f"Liczba turbin:      {summary['n_turbines']}")
print(f"Moc zainstalowana:  {summary['total_capacity_mw']:.0f} MW")
print(f"Model wake:         {summary['wake_model'].upper()}")
print(f"AEP:                {summary['farm_aep_gwh']:.2f} GWh")
print(f"Capacity factor:    {summary['capacity_factor_pct']:.1f}%")
print(f"Wake losses:        {summary['wake_losses_pct']:.1f}%")
print(f"{'='*50}")

# %%
fig = calc.plot_aep_breakdown(loader)
plt.savefig("../outputs/figures/08_aep_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 11. Analiza niepewności

# %%
uncertainty_df = calc.uncertainty_sweep(wd_stds=[1.0, 3.0, 5.0, 7.0, 10.0])

print(f"\n{'σ [°]':<8} {'AEP det. [GWh]':<18} {'AEP unc. [GWh]':<18} {'Różnica [%]'}")
print("=" * 62)
for _, row in uncertainty_df.iterrows():
    print(f"{row['wd_std_deg']:<8.0f} {row['aep_deterministic_gwh']:<18.2f} "
          f"{row['aep_uncertain_gwh']:<18.2f} {row['difference_pct']:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(uncertainty_df["wd_std_deg"], uncertainty_df["aep_uncertain_gwh"],
        "o-", color="#534AB7", linewidth=2, markersize=8)
ax.axhline(uncertainty_df["aep_deterministic_gwh"].iloc[0],
           color="#c8531a", linestyle="--", linewidth=1.5, label="AEP deterministyczne")
ax.set_xlabel("Niepewność kierunku wiatru σ [°]")
ax.set_ylabel("AEP [GWh]")
ax.set_title("Wpływ niepewności kierunku wiatru na AEP")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("../outputs/figures/09_uncertainty.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 12. Porównanie scenariuszy

# %%
scenarios = {
    "Bazowy (grid 7D)":       {"layout_type": "grid",      "spacing_D": 7.0},
    "Gęsty (grid 5D)":        {"layout_type": "grid",      "spacing_D": 5.0},
    "Rzadki (grid 10D)":      {"layout_type": "grid",      "spacing_D": 10.0},
    "Staggered 7D":           {"layout_type": "staggered",  "spacing_D": 7.0},
    "Staggered 10D":          {"layout_type": "staggered",  "spacing_D": 10.0},
}

scenario_df = calc.compare_scenarios(scenarios)

print(f"\n{'Scenariusz':<22} {'AEP [GWh]':<12} {'Wake loss [%]':<14} {'CF [%]':<8}")
print("=" * 56)
for _, row in scenario_df.iterrows():
    print(f"{row['scenario']:<22} {row['aep_gwh']:<12.2f} "
          f"{row['wake_loss_pct']:<14.1f} {row['capacity_factor_pct']:<8.1f}")

fig = calc.plot_scenario_comparison(scenario_df)
plt.savefig("../outputs/figures/10_scenarios.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 13. Eksport danych dla grupy

# %%
# Layout CSV — dla Tematu 3 i 4
farm.set_layout_grid(n_rows=N_ROWS, n_cols=N_COLS, spacing_D=SPACING_D)
farm.export_layout_csv("../outputs/exports/layout.csv")

# AEP CSV — dla Tematu 5
farm.set_wind_data(wr_fine)
export_df = calc.export_for_team5("../outputs/exports/aep_timeseries.csv", loader=loader)

print("Pliki wyeksportowane:")
print("  outputs/exports/layout.csv         — dla Tematu 3 i 4")
print("  outputs/exports/aep_timeseries.csv — dla Tematu 5")
print(f"\nPodgląd layout.csv:")
print(pd.read_csv("../outputs/exports/layout.csv").head())
print(f"\nPodgląd aep_timeseries.csv:")
print(export_df.head())

# %% [markdown]
# ---
# ## 14. Podsumowanie wyników
#
# Poniżej zebrane najważniejsze wyniki z całej symulacji.

# %%
print("=" * 70)
print("PODSUMOWANIE SYMULACJI — FARMA WIATROWA OFFSHORE BAŁTYK PD.")
print("=" * 70)
print(f"\nLOKALIZACJA: ~54.5°N, 16.5°E (Bałtyk Południowy)")
print(f"DANE:        {stats['n_records']} godzin ({stats['n_records']/8760:.0f} lat)")
print(f"             Weibull k={stats['wind_speed']['weibull_k_est']:.2f}, "
      f"A={stats['wind_speed']['weibull_A_est']:.1f} m/s")
print(f"             Dominujący kierunek: {stats['wind_direction']['dominant']:.0f}°")

print(f"\nFARMA:")
print(f"  Turbina:        IEA 15 MW (D={D:.0f}m)")
print(f"  Layout:         {N_ROWS}x{N_COLS} grid @ {SPACING_D}D = {N_TURBINES} turbin")
print(f"  Moc:            {N_TURBINES * 15} MW")
print(f"  Obszar:         ~{farm_area_km2:.1f} km²")

print(f"\nWYNIKI (GCH wake model):")
print(f"  AEP:            {summary['farm_aep_gwh']:.2f} GWh")
print(f"  Capacity factor: {summary['capacity_factor_pct']:.1f}%")
print(f"  Wake losses:    {summary['wake_losses_pct']:.1f}%")

print(f"\nOPTYMALIZACJA (4x4 grid):")
print(f"  Scipy:          +{result.aep_improvement_pct:.2f}% AEP ({result.elapsed_seconds:.0f}s)")
if result_yaw is not None:
    print(f"  Yaw steering:   +{result_yaw.aep_improvement_pct:.2f}% AEP ({result_yaw.elapsed_seconds:.0f}s)")

print(f"\nMODELE WAKE:")
for name, data in wake_results.items():
    print(f"  {name.upper():<15} AEP={data['aep']:.1f} GWh, wake_loss={data['wake_loss']:.1f}%")

print(f"\nEKSPORT:")
print(f"  layout.csv         -> Temat 3 (wake steering), Temat 4 (HAWC obciążenia)")
print(f"  aep_timeseries.csv -> Temat 5 (analiza ekonomiczna)")
print("=" * 70)
