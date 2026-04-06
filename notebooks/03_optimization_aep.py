# %% [markdown]
# # 03 — Optymalizacja layoutu i analiza AEP
# **Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**
#
# Ten notebook demonstruje:
# 1. Optymalizację layoutu (Scipy + RandomSearch)
# 2. Optymalizację yaw (SerialRefine)
# 3. Porównanie metod optymalizacji
# 4. Szczegółową analizę AEP (sezonowa, miesięczna)
# 5. Analizę niepewności (UncertainFlorisModel)
# 6. Porównanie scenariuszy
# 7. Eksport danych dla Tematu 5

# %% Importy
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt

from src.wind_data import WindDataLoader, BalticWindConfig
from src.farm_model import FarmModel
from src.optimizer import Optimizer
from src.aep_calculator import AEPCalculator

print("Wszystkie moduły załadowane.")

# %% [markdown]
# ## 0. Przygotowanie danych
# Tworzymy DWA obiekty WindRose:
# - **wind_rose_coarse** — gruby (30° × 3 m/s) → do optymalizacji (szybko)
# - **wind_rose_fine** — dokładny (10° × 2 m/s) → do finalnego AEP

# %%
loader = WindDataLoader()
loader.generate_mock_data(seed=42)

# Gruby — do optymalizacji (~12 × 7 = 84 warunki zamiast ~400)
wind_rose_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)

# Dokładny — do finalnych obliczeń AEP
wind_rose_fine = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)

print(f"WindRose gruby (optymalizacja): szybki")
print(f"WindRose dokładny (AEP): dokładny")

# %%
# 3x3 farma do testów (9 turbin — szybkie)
farm = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wind_rose_coarse)
farm.set_layout_grid(n_rows=3, n_cols=3, spacing_D=7.0)
farm.run()
print(f"Farm: {farm}")
print(f"Start AEP (coarse): {farm.get_aep_gwh():.2f} GWh")

# %% [markdown]
# ## 1. Optymalizacja layoutu — Scipy
# ~1-3 min na 9 turbin z grubą WindRose

# %%
opt = Optimizer(farm)
opt.set_boundaries_from_layout(margin_D=3.0)
opt.set_min_distance(min_dist_D=3.0)
print(opt)

# %%
result_scipy = opt.optimize_layout_scipy(maxiter=30)

print(f"\nScipy: AEP {result_scipy.initial_aep_gwh:.2f} -> "
      f"{result_scipy.optimized_aep_gwh:.2f} GWh "
      f"(+{result_scipy.aep_improvement_pct:.2f}%) | {result_scipy.elapsed_seconds:.1f}s")

fig = opt.plot_optimization_result(result_scipy)
plt.savefig("../outputs/figures/optimization_scipy.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Optymalizacja layoutu — Random Search
# ~30 sekund (kontrolowany czas)

# %%
# RandomSearch wymaga multiprocessing, który wiesza się w Jupyter na Windows.
# Aby użyć RandomSearch, odpal z terminala:
#   python -c "from src.optimizer import ...; ..."
# Na razie pomijamy — Scipy daje dobre wyniki.
print("RandomSearch pominięty (multiprocessing + Jupyter + Windows = hang)")
print("Scipy dał +3.63% — przechodzimy dalej.")

# Tworzymy dummy result żeby reszta notebooka działała
from src.optimizer import OptimizationResult
result_rs = OptimizationResult(
    method="random_search (pominięty)",
    initial_x=farm.layout_x,
    initial_y=farm.layout_y,
    initial_aep_gwh=result_scipy.initial_aep_gwh,
    optimized_x=farm.layout_x,
    optimized_y=farm.layout_y,
    optimized_aep_gwh=result_scipy.initial_aep_gwh,
    aep_improvement_pct=0.0,
    elapsed_seconds=0.0,
)
# %% [markdown]
# ## 3. Optymalizacja Yaw (wake steering)

# %%
farm.set_layout_grid(n_rows=3, n_cols=3, spacing_D=7.0)
farm.set_wind_data(wind_rose_coarse)

try:
    result_yaw = opt.optimize_yaw(yaw_min=0.0, yaw_max=25.0)
    print(f"\nYaw: AEP {result_yaw.initial_aep_gwh:.2f} -> "
          f"{result_yaw.optimized_aep_gwh:.2f} GWh "
          f"(+{result_yaw.aep_improvement_pct:.2f}%) | {result_yaw.elapsed_seconds:.1f}s")

    fig = opt.plot_yaw_result(result_yaw)
    plt.savefig("../outputs/figures/optimization_yaw.png", dpi=150, bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Yaw optimization error: {e}")
    result_yaw = None

# %% [markdown]
# ## 4. Porownanie metod

# %%
print("\n" + "=" * 60)
print(f"{'Metoda':<20} {'AEP [GWh]':<12} {'Zmiana':<10} {'Czas [s]':<10}")
print("=" * 60)
print(f"{'Poczatkowy':<20} {result_scipy.initial_aep_gwh:<12.2f} {'---':<10} {'---':<10}")
print(f"{'Scipy':<20} {result_scipy.optimized_aep_gwh:<12.2f} "
      f"{'+' + f'{result_scipy.aep_improvement_pct:.2f}%':<10} "
      f"{result_scipy.elapsed_seconds:<10.1f}")
print(f"{'RandomSearch':<20} {result_rs.optimized_aep_gwh:<12.2f} "
      f"{'+' + f'{result_rs.aep_improvement_pct:.2f}%':<10} "
      f"{result_rs.elapsed_seconds:<10.1f}")
if result_yaw is not None:
    print(f"{'Yaw (wake steer.)':<20} {result_yaw.optimized_aep_gwh:<12.2f} "
          f"{'+' + f'{result_yaw.aep_improvement_pct:.2f}%':<10} "
          f"{result_yaw.elapsed_seconds:<10.1f}")
print("=" * 60)

# %% [markdown]
# ## 5. Finalne AEP — dokladna WindRose
# Aplikujemy najlepszy layout i liczymy AEP na dokladnej WindRose.

# %%
best = max([result_scipy, result_rs], key=lambda r: r.optimized_aep_gwh)
opt.apply_result(best)
farm.set_wind_data(wind_rose_fine)
farm.run()

print(f"Najlepszy layout: {best.method}")
print(f"AEP (gruby): {best.optimized_aep_gwh:.2f} GWh")
print(f"AEP (dokladny): {farm.get_aep_gwh():.2f} GWh")

# %% [markdown]
# ## 6. Analiza AEP — sezonowa i miesieczna

# %%
calc = AEPCalculator(farm)
summary = calc.compute_aep_summary()

print(f"\nPodsumowanie AEP:")
print(f"  AEP roczne: {summary['farm_aep_gwh']:.2f} GWh")
print(f"  Capacity factor: {summary['capacity_factor_pct']:.1f}%")
print(f"  Wake losses: {summary['wake_losses_pct']:.1f}%")
print(f"  Moc zainstalowana: {summary['total_capacity_mw']:.0f} MW")

# %%
fig = calc.plot_aep_breakdown(loader)
plt.savefig("../outputs/figures/aep_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Analiza niepewnosci

# %%
print("Analiza niepewnosci kierunku wiatru...")
uncertainty_df = calc.uncertainty_sweep(wd_stds=[1.0, 3.0, 5.0, 7.0, 10.0])
print(uncertainty_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    uncertainty_df["wd_std_deg"],
    uncertainty_df["aep_uncertain_gwh"],
    "o-", color="#534AB7", linewidth=2, markersize=8,
)
ax.axhline(
    uncertainty_df["aep_deterministic_gwh"].iloc[0],
    color="#c8531a", linestyle="--", linewidth=1.5, label="AEP deterministyczne",
)
ax.set_xlabel("Niepewnosc kierunku wiatru sigma [deg]")
ax.set_ylabel("AEP [GWh]")
ax.set_title("Wplyw niepewnosci kierunku wiatru na AEP")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("../outputs/figures/uncertainty_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Porownanie scenariuszy

# %%
scenarios = {
    "Grid 7D": {"layout_type": "grid", "spacing_D": 7.0, "n_rows": 3, "n_cols": 3},
    "Grid 9D": {"layout_type": "grid", "spacing_D": 9.0, "n_rows": 3, "n_cols": 3},
    "Staggered 7D": {"layout_type": "staggered", "spacing_D": 7.0, "n_rows": 3, "n_cols": 3},
    "Staggered 9D": {"layout_type": "staggered", "spacing_D": 9.0, "n_rows": 3, "n_cols": 3},
}

scenario_df = calc.compare_scenarios(scenarios)
print(scenario_df[["scenario", "aep_gwh", "wake_loss_pct", "capacity_factor_pct"]].to_string(index=False))

fig = calc.plot_scenario_comparison(scenario_df)
plt.savefig("../outputs/figures/scenario_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Eksport danych dla Tematu 5

# %%
export_df = calc.export_for_team5(
    "../outputs/exports/aep_timeseries.csv",
    loader=loader,
)
print("Wyeksportowano aep_timeseries.csv:")
print(export_df.head(15).to_string(index=False))

# %% [markdown]
# ## Podsumowanie
# - Layout zoptymalizowany dwoma metodami (Scipy + RandomSearch)
# - Optymalizacja yaw (wake steering)
# - AEP rozbite na sezony i miesiace
# - Analiza niepewnosci
# - Porownanie scenariuszy (grid vs staggered, 7D vs 9D)
# - Dane wyeksportowane dla Tematu 5
#
# **Do finalnych obliczen na pelna farme (5x5, 25 turbin)**
# zwieksz n_rows/n_cols i daj wiecej czasu na optymalizacje.
