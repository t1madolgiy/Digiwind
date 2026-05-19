# Digiwind
# Wind Farm Layout Optimization

**Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**

Narzędzie do optymalizacji rozmieszczenia turbin wiatrowych offshore na Bałtyku Południowym. Wykorzystuje bibliotekę [FLORIS](https://github.com/NREL/floris) (NREL) do modelowania śladów aerodynamicznych i maksymalizacji rocznej produkcji energii (AEP).

🔗 **[Streamlit Dashboard](https://digiwind-fyhbp3zaekx3dabefrappyk.streamlit.app/)** — interaktywna aplikacja webowa

---

## Funkcje

- **5 modeli wake** — Jensen, GCH (Gauss), TurbOPark, Empirical Gauss, CC
- **4 turbiny referencyjne** — NREL 5MW, IEA 10MW, IEA 15MW, IEA 22MW (+ custom YAML + floating)
- **4 typy layoutów** — siatka regularna, siatka przesunięta (staggered), kołowy, równoległobok (Malisani 2025)
- **Źródła danych wiatrowych** — Mock (Weibull) lub ERA5 (Copernicus CDS)
- **Optymalizacja layoutu** — Scipy gradient + Random Search (FLORIS) na pełnej róży wiatrów
- **🧪 Lab algorytmów** — porównanie 6 algorytmów (Scipy, Random Search, GA, SA, PSO, Greedy) na wąskim binie wiatrowym, każdy w osobnym pliku → łatwe dodawanie nowych
- **🤝 Zakładka Grupy 3** — baza pod wake steering: yaw + yaw_max sweep, curtailment (disable/derating), Helix (AWM), eksport yaw_schedule.csv
- **Analiza AEP** — roczna, sezonowa, miesięczna, z analizą niepewności
- **Porównania** — modele wake, turbiny, rozstawy (spacing sweep), scenariusze
- **Benchmark** — porównanie wielu konfiguracji naraz (wake × layout × spacing)
- **Zaawansowane** — AVP, derating turbin, Helix (AWM), analiza wrażliwości parametrów
- **Eksport danych** — `layout.csv` i `aep_timeseries.csv` dla innych tematów w grupie
- **Streamlit Dashboard** — interaktywna aplikacja z 14 zakładkami

## Wyniki (mock data — Bałtyk Pd.)

| Parametr | Wartość |
|---|---|
| Lokalizacja | ~54.5°N, 16.5°E (Bałtyk Południowy) |
| Turbina | IEA 15 MW (D=240m, hub=150m) |
| Layout | 5×5 grid @ 7D = 25 turbin |
| Moc zainstalowana | 375 MW |
| AEP | ~1669 GWh |
| Capacity Factor | ~50.8% |
| Wake losses (GCH) | ~5.6% |
| Optymalizacja Scipy | +3.8% AEP |
| Yaw steering | +0.8% AEP |

## Instalacja

```bash
# Klonuj repozytorium
git clone https://github.com/t1madolgiy/Digiwind.git
cd Digiwind

# Stwórz środowisko conda
conda create -n windfarm python=3.11 -y
conda activate windfarm

# Zainstaluj zależności
pip install -r requirements.txt
```

### ERA5 (opcjonalnie — dane rzeczywiste z Copernicus)

Aby używać prawdziwych danych wiatrowych zamiast mock:

1. Załóż darmowe konto na [Copernicus CDS](https://cds.climate.copernicus.eu/)
2. Pobierz klucz API i zapisz w `~/.cdsapirc` ([instrukcja](https://cds.climate.copernicus.eu/api-how-to))
3. W sidebarze Streamlit wybierz "Źródło danych: ERA5 (Copernicus)"
4. Bez klucza appka automatycznie zrobi fallback do mock data

## Uruchomienie

### Streamlit Dashboard
```bash
streamlit run app.py
```
Otwiera się przeglądarka na `http://localhost:8501`.

### Notebooki (VS Code)
Otwórz pliki z `notebooks/` w VS Code — komentarze `# %%` tworzą komórki Jupyter.

## Struktura projektu

```
Digiwind/
├── app.py                          # Streamlit dashboard (14 zakładek)
├── requirements.txt                # Zależności Python
├── src/                            # Moduły logiki
│   ├── wind_data.py                # WindDataLoader — Mock / ERA5 / CSV
│   ├── farm_model.py               # FarmModel — symulacja FLORIS
│   ├── optimizer.py                # Optimizer — Scipy + RandomSearch + YawSR
│   ├── aep_calculator.py           # AEPCalculator — analiza produkcji energii
│   ├── report_generator.py         # ReportGenerator — PDF
│   └── algorithms/                 # 🧪 Lab algorytmów (każdy w osobnym pliku)
│       ├── base.py                 # LayoutAlgorithm ABC + AlgorithmResult + helpery
│       ├── scipy_gradient.py       # wrapper na FLORIS LayoutOptimizationScipy
│       ├── random_search.py        # własna implementacja losowych prób
│       ├── genetic.py              # GA z BLX-α + gaussian mutation + elitism
│       ├── simulated_annealing.py  # SA z geometric cooling
│       ├── pso.py                  # Particle Swarm Optimization
│       └── greedy.py               # Greedy Relocate (turbina po turbinie)
├── notebooks/                      # Notebooki Jupyter (VS Code)
│   ├── 01_eda_wind.py              # Eksploracja danych wiatrowych
│   ├── 02_layouts.py               # Layouty i porównania modeli
│   ├── 03_optimization_aep.py      # Optymalizacja i analiza AEP
│   ├── 04_full_simulation.py       # Pełna symulacja 375 MW
│   └── 05_advanced_floris.py       # AVP, derating, helix, sensitivity
├── data/                           # Dane wejściowe
│   ├── raw/                        # Surowe dane (ERA5 .nc, CSV)
│   ├── processed/                  # Przetworzone dane
│   └── turbines/                   # Pliki YAML z custom turbinami
└── outputs/                        # Wyniki
    ├── figures/                    # Wykresy PNG
    ├── exports/                    # CSV dla innych tematów
    └── logs/                       # Logi
```

## Streamlit — zakładki

| # | Zakładka | Opis |
|---|---|---|
| 1 | 📊 Przegląd | Layout farmy, kluczowe metryki (AEP, CF, moc) |
| 2 | 🌬️ Wiatr | Róża wiatrów, statystyki Weibulla, time series |
| 3 | 🌊 Flow field | Wizualizacja pola przepływu (wake) |
| 4 | ⚖️ Porównania | Modele wake, turbiny, spacing sweep |
| 5 | 🏆 Benchmark | Wiele konfiguracji naraz (wake × layout × spacing) |
| 6 | 🎯 Optymalizacja | Scipy gradient + Random Search (pełna róża) |
| 7 | 🧪 Lab algorytmów | **Porównanie 6 algorytmów na wąskim binie** |
| 8 | 🤝 Grupa 3 | **Baza pod Temat 3** (yaw, curtailment, helix) |
| 9 | ⚡ AEP | Analiza sezonowa, miesięczna, niepewność |
| 10 | 🔧 Turbiny | Custom turbiny (YAML), podgląd biblioteki |
| 11 | ✏️ Edytor | Ręczna edycja współrzędnych turbin |
| 12 | 🌐 3D | Wizualizacje Plotly (layout, profil wiatru, mapa mocy) |
| 13 | 📄 Raport | Generowanie PDF |
| 14 | 📁 Eksport | layout.csv, aep_timeseries.csv |

## 🧪 Lab algorytmów — dodanie własnego algorytmu

Każdy algorytm to osobny plik w `src/algorithms/`. Aby dodać własny:

1. Stwórz `src/algorithms/moj_algo.py`:
```python
from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep

class MojAlgorytm(LayoutAlgorithm):
    name = "Mój Algorytm"
    description = "Opis"

    def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params):
        # ... twoja logika ...
        return AlgorithmResult(
            name=self.name,
            initial_x=..., initial_y=..., initial_aep=...,
            final_x=..., final_y=..., final_aep=...,
            elapsed_s=..., n_evaluations=..., history=[...],
        )
```

2. Zarejestruj w `src/algorithms/__init__.py`:
```python
from .moj_algo import MojAlgorytm
ALGORITHMS["moj"] = MojAlgorytm
```

3. Algorytm pojawi się automatycznie w multiselectie zakładki **🧪 Lab algorytmów**.

## Technologie

- **FLORIS v4.6** — silnik wake modeling (NREL)
- **Python 3.11** — język programowania
- **Streamlit** — interfejs webowy
- **Matplotlib + Plotly** — wizualizacje 2D/3D
- **NumPy / Pandas / SciPy** — obliczenia
- **cdsapi + xarray + netcdf4** — dane ERA5 (opcjonalnie)

## Kontekst projektu

Projekt badawczy na studia — analiza produktywności i optymalizacja rozmieszczenia turbin wiatrowych w farmie offshore dla różnych zestawów danych wiatrowych z wykorzystaniem modeli analitycznych śladu aerodynamicznego (wake effect).

### Zależności z grupą
- **Temat 1** → dostarcza dane wiatrowe (ERA5 / klimatyczne) → mamy wbudowane `WindDataLoader.from_era5()`
- **Temat 2** → **(ten projekt)** optymalizacja layoutu farmy + Lab algorytmów
- **Temat 3** → wake steering — zakładka `🤝 Grupa 3` (yaw, curtailment, helix — baza gotowa, pełną implementację robi Grupa 3); wejście: `layout.csv`
- **Temat 4** → obciążenia łopat HAWC; wejście: `layout.csv`
- **Temat 5** → analiza ekonomiczna; wejście: `aep_timeseries.csv`

## Roadmap

- [x] Modularny kod (5 modułów + `src/algorithms/`)
- [x] 5 notebooków z analizami
- [x] Streamlit dashboard (14 zakładek)
- [x] Porównanie modeli wake i turbin
- [x] Optymalizacja layoutu (Scipy, Random Search) i yaw (SerialRefine)
- [x] 🧪 Lab algorytmów — 6 algorytmów do porównania na wąskim binie
- [x] 🤝 Zakładka Grupa 3 — baza pod wake steering
- [x] ERA5 jako alternatywa dla mock data
- [x] Custom turbiny (YAML)
- [x] Turbiny pływające (floating offshore)
- [x] Wizualizacje 3D (Plotly)
- [x] Automatyczne generowanie raportu PDF
- [x] Analiza AEP (sezonowa, miesięczna, niepewność)
- [x] Zaawansowane: AVP, derating, helix, analiza wrażliwości
- [x] Eksport danych dla grupy
- [x] Deployment na Streamlit Cloud

