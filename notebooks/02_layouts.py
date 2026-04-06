# %% [markdown]
# # 02 — Layouty farmy i porównania modeli
# **Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**
#
# Ten notebook demonstruje:
# 1. Tworzenie layoutów (grid, staggered, circular)
# 2. Wizualizację flow field (heatmapy wake)
# 3. Porównanie modeli wake (Jensen vs GCH vs TurbOPark vs CC)
# 4. Porównanie turbin (NREL 5MW vs IEA 10/15/22MW)
# 5. Spacing sweep (optymalna odległość między turbinami)
# 6. Eksport layoutu CSV dla grupy

# %% Importy
import sys
sys.path.insert(0, "..")

import numpy as np
import matplotlib.pyplot as plt

from src.wind_data import WindDataLoader, BalticWindConfig
from src.farm_model import FarmModel, WAKE_MODELS, TURBINE_LIBRARY

print("Dostępne modele wake:", list(WAKE_MODELS.keys()))
print("Dostępne turbiny:", list(TURBINE_LIBRARY.keys()))

# %% [markdown]
# ## 0. Przygotowanie danych wiatrowych

# %%
loader = WindDataLoader()
loader.generate_mock_data(seed=42)
wind_rose = loader.to_wind_rose(wd_step=10.0, ws_step=2.0)
print(f"WindRose gotowa: {type(wind_rose).__name__}")

# %% [markdown]
# ## 1. Tworzenie FarmModel i layoutów

# %%
# Inicjalizacja — GCH + IEA 15MW
farm = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wind_rose)
print(farm)

# %% Layout: siatka regularna 5x5
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
fig = farm.plot_layout()
plt.savefig("../outputs/figures/layout_grid.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Layout: siatka przesunięta 5x5
farm.set_layout_staggered(n_rows=5, n_cols=5, spacing_D=7.0, offset=0.5)
fig = farm.plot_layout()
plt.savefig("../outputs/figures/layout_staggered.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Layout: kołowy
farm.set_layout_circular(n_turbines=12, radius_D=10.0, n_rings=2)
fig = farm.plot_layout()
plt.savefig("../outputs/figures/layout_circular.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Wizualizacja flow field

# %% Heatmapa — siatka regularna, wiatr z zachodu
farm.set_layout_grid(n_rows=3, n_cols=5, spacing_D=7.0)
fig = farm.plot_flow_field(
    wind_direction=270.0,
    wind_speed=9.0,
    ti=0.06,
)
plt.savefig("../outputs/figures/flow_field_grid_270.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Heatmapa — wiatr z SW (240°, dominujący na Bałtyku)
fig = farm.plot_flow_field(
    wind_direction=240.0,
    wind_speed=9.0,
    ti=0.06,
)
plt.savefig("../outputs/figures/flow_field_grid_240.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Cross-section — przekrój 5D za pierwszą turbiną
fig = farm.plot_cross_section(
    downstream_D=5.0,
    wind_direction=270.0,
    wind_speed=9.0,
)
plt.savefig("../outputs/figures/cross_section_5D.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Porównanie modeli wake

# %%
farm.set_layout_grid(n_rows=3, n_cols=5, spacing_D=7.0)
farm.set_wind_data(wind_rose)

fig, results = farm.compare_wake_models(
    models=["jensen", "gch", "turbopark", "cc"],
    wind_direction=270.0,
    wind_speed=9.0,
)
plt.savefig("../outputs/figures/wake_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nAEP per model wake:")
for model, aep in results.items():
    if aep is not None:
        print(f"  {model:20s}: {aep:.2f} GWh")
    else:
        print(f"  {model:20s}: BŁĄD")

# %% [markdown]
# ## 4. Porównanie turbin

# %%
turbine_results = farm.compare_turbines(
    turbines=["nrel_5MW", "iea_10MW", "iea_15MW", "iea_22MW"],
    spacing_D=7.0,
    n_rows=5,
    n_cols=5,
)

print(f"\n{'Turbina':<15} {'D [m]':<8} {'N':<5} {'AEP [GWh]':<12} "
      f"{'Wake loss':<12} {'CF [%]':<8}")
print("=" * 65)
for name, data in turbine_results.items():
    if "error" not in data:
        print(
            f"{data['name']:<15} {data['diameter']:<8.0f} "
            f"{data['n_turbines']:<5d} {data['aep_gwh']:<12.2f} "
            f"{data['wake_loss_pct']:<12.1f} {data['capacity_factor']:<8.1f}"
        )

# %%
# Wykres porównawczy turbin
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

names = [d["name"] for d in turbine_results.values() if "error" not in d]
aeps = [d["aep_gwh"] for d in turbine_results.values() if "error" not in d]
losses = [d["wake_loss_pct"] for d in turbine_results.values() if "error" not in d]
cfs = [d["capacity_factor"] for d in turbine_results.values() if "error" not in d]

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

fig.suptitle("Porównanie turbin — 5x5 grid @ 7D", fontsize=14, fontweight="500")
fig.tight_layout()
plt.savefig("../outputs/figures/turbine_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Spacing sweep — optymalna odległość

# %%
farm = FarmModel(wake_model="gch", turbine="iea_15MW", wind_data=wind_rose)
sweep = farm.spacing_sweep(
    spacings_D=np.arange(4.0, 12.5, 0.5),
    n_rows=5,
    n_cols=5,
)

fig, ax1 = plt.subplots(figsize=(10, 6))

color_aep = "#1e5c3a"
color_loss = "#c8531a"

ax1.plot(sweep["spacings_D"], sweep["aep_gwh"], "o-", color=color_aep, linewidth=2, label="AEP")
ax1.set_xlabel("Rozstaw [×D]")
ax1.set_ylabel("AEP [GWh]", color=color_aep)
ax1.tick_params(axis="y", labelcolor=color_aep)

ax2 = ax1.twinx()
ax2.plot(sweep["spacings_D"], sweep["wake_loss_pct"], "s--", color=color_loss, linewidth=2, label="Wake losses")
ax2.set_ylabel("Wake losses [%]", color=color_loss)
ax2.tick_params(axis="y", labelcolor=color_loss)

fig.suptitle(
    f"Spacing sweep — {farm.turbine_info['name']} | 5x5 grid | GCH",
    fontsize=14, fontweight="500",
)
fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
ax1.grid(True, alpha=0.3)
plt.savefig("../outputs/figures/spacing_sweep.png", dpi=150, bbox_inches="tight")
plt.show()

# Optymalny rozstaw
best_idx = np.argmax(sweep["aep_gwh"])
print(f"\nOptymalny rozstaw: {sweep['spacings_D'][best_idx]:.1f}D")
print(f"  AEP: {sweep['aep_gwh'][best_idx]:.2f} GWh")
print(f"  Wake losses: {sweep['wake_loss_pct'][best_idx]:.1f}%")

# %% [markdown]
# ## 6. Eksport layoutu CSV

# %%
farm.set_layout_grid(n_rows=5, n_cols=5, spacing_D=7.0)
farm.export_layout_csv("../outputs/exports/layout.csv")
print("Layout wyeksportowany!")

# Podgląd
import pandas as pd
df = pd.read_csv("../outputs/exports/layout.csv")
print(df.head(10))

# %% [markdown]
# ## Następne kroki
# - [ ] Notebook 03: Optymalizacja layoutu (LayoutOptimizationScipy + RandomSearch)
# - [ ] Notebook 04: Analiza AEP (sezonowa, dobowa, scenariusze klimatyczne)
# - [ ] Dodanie analizy niepewności (UncertainFlorisModel)
