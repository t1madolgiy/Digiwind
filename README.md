# 🌊 Digiwind — Wind Farm Layout Optimization

**Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**

Narzędzie do analizy produktywności i optymalizacji rozmieszczenia turbin wiatrowych offshore na Bałtyku Południowym. Wykorzystuje bibliotekę [FLORIS v4.6](https://github.com/NREL/floris) (NREL) do modelowania śladów aerodynamicznych (wake effect) i maksymalizacji rocznej produkcji energii (AEP).

🔗 **[Streamlit Dashboard (live)](https://digiwind-fyhbp3zaekx3dabefrappyk.streamlit.app/)** — interaktywna aplikacja webowa

---

## Spis treści

- [Funkcje](#funkcje)
- [Szybki start](#szybki-start)
- [Logowanie i role](#logowanie-i-role)
- [Struktura projektu](#struktura-projektu)
- [Architektura](#architektura-jak-to-działa)
- [Streamlit — zakładki](#streamlit--zakładki)
- [Lab algorytmów](#-lab-algorytmów--dodanie-własnego-algorytmu)
- [Dane wiatrowe (Mock / ERA5 / CSV)](#dane-wiatrowe)
- [Wyniki (mock — Bałtyk Pd.)](#wyniki-mock-data--bałtyk-pd)
- [Zależności z grupą](#zależności-z-grupą)
- [Rozwiązywanie problemów](#rozwiązywanie-problemów)
- [Technologie](#technologie)

---

## Funkcje

- **5 modeli wake** — Jensen, GCH (Gauss), TurbOPark, Empirical Gauss, CC
- **4 turbiny referencyjne** — NREL 5 MW, IEA 10/15/22 MW (+ custom YAML + floating IEA 15 MW)
- **4 typy layoutów** — siatka regularna, przesunięta (staggered), kołowy, równoległobok (Malisani 2025)
- **2 źródła danych wiatrowych** — Mock (Weibull + von Mises z sezonowością) lub ERA5 (Copernicus CDS)
- **Globalny tryb wiatru obliczeniowego** — jeden wybór (wąski bin *lub* pełna róża) spójny dla wszystkich modułów
- **Optymalizacja layoutu** — Scipy gradient + Random Search (FLORIS) na pełnej róży wiatrów
- **🧪 Lab algorytmów** — porównanie **7 algorytmów** (Scipy, Random Search, GA, Differential Evolution, Simulated Annealing, PSO, Greedy) na wąskim binie, każdy w osobnym pliku → łatwe dodawanie nowych
- **🤝 Zakładka Grupa 3** — baza pod wake steering: yaw + yaw_max sweep, curtailment (disable/derating), Helix (AWM), eksport `yaw_schedule.csv`
- **💰 Analiza ekonomiczna (Grupa 5)** — CAPEX/OPEX, LCOE, NPV, IRR, payback, cena energii, dyskonto, eskalacja, degradacja
- **Analiza AEP** — roczna, sezonowa, miesięczna, z analizą niepewności (UncertainFlorisModel)
- **Porównania** — modele wake, turbiny, rozstawy (spacing sweep), scenariusze
- **Benchmark** — wiele konfiguracji naraz (wake × layout × spacing) + porównanie flow field
- **Wizualizacje 3D** — Plotly (layout, profil wiatru, mapa mocy)
- **Eksport danych** — `layout.csv` i `aep_timeseries.csv` dla innych tematów w grupie
- **Raport PDF** — automatyczne generowanie (reportlab, z polskimi znakami)
- **🔐 Logowanie z rolami** — `admin` (pełny dostęp + kontrola widoczności zakładek) / `viewer` (zakładki podstawowe)
- **🛡️ Odporność na środowisko** — sprawdzanie wymaganych bibliotek przy starcie + fallback bez `pyarrow` (komputery firmowe z WDAC/AppLocker)

---

## Szybki start

```bash
# Klonuj repozytorium
git clone https://github.com/t1madolgiy/Digiwind.git
cd Digiwind

# Stwórz środowisko conda
conda create -n windfarm python=3.11 -y
conda activate windfarm

# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom dashboard
streamlit run app.py
```

Dashboard otworzy się na `http://localhost:8501`.

> **Sprawdzanie bibliotek przy starcie** — aplikacja sama weryfikuje, czy zainstalowane są wszystkie wymagane pakiety. Jeśli któregoś brakuje, pokaże gotowe polecenie `pip install ...` i zatrzyma się; brakujące biblioteki opcjonalne (ERA5) dają tylko ostrzeżenie.

### Notebooki (VS Code)

Pliki w `notebooks/` używają komentarzy `# %%`, które VS Code interpretuje jako komórki Jupyter. `06_scipy_optimization.ipynb` to natywny notebook `.ipynb`.

---

## Logowanie i role

Aplikacja wymaga logowania. Domyślne konta (do zmiany w `.streamlit/secrets.toml` na produkcji):

| Login   | Hasło          | Rola     | Dostęp                                                        |
|---------|----------------|----------|--------------------------------------------------------------|
| `admin` | `digiwind2025` | `admin`  | wszystkie zakładki + panel sterowania widocznością zakładek  |
| `gość`  | `gosc`         | `viewer` | tylko podstawowe: 🌬️ Wiatr, 🔧 Turbiny, 📐 Layout, ⚡ AEP   |

Aby nadpisać konta, dodaj sekcję `[users]` w `.streamlit/secrets.toml`:

```toml
[users.admin]
password = "twoje_haslo"
role = "admin"

[users.student]
password = "haslo123"
role = "viewer"
```

---

## Struktura projektu

```
Digiwind/
├── app.py                          # Streamlit dashboard (logowanie + 12 zakładek)
├── requirements.txt                # Zależności Python
├── test_parfloris.py               # Test layoutu równoległoboku
├── src/                            # Moduły logiki
│   ├── wind_data.py                # WindDataLoader — Mock / ERA5 / CSV → obiekty FLORIS
│   ├── farm_model.py               # FarmModel — wrapper FlorisModel + layouty + wake + flow field
│   ├── optimizer.py                # Optimizer — Scipy + RandomSearch + YawOptimizationSR
│   ├── aep_calculator.py           # AEPCalculator — sezonowa/miesięczna/niepewność + eksport
│   ├── report_generator.py         # ReportGenerator — raport PDF (reportlab)
│   └── algorithms/                 # 🧪 Lab algorytmów (każdy w osobnym pliku)
│       ├── base.py                 # LayoutAlgorithm (ABC) + AlgorithmResult + helpery
│       ├── scipy_gradient.py       # wrapper na FLORIS LayoutOptimizationScipy
│       ├── random_search.py        # własna implementacja losowych prób
│       ├── genetic.py              # GA: BLX-α crossover + gaussian mutation + elitism
│       ├── differential_evolution.py # DE: mutacja różnicowa + crossover
│       ├── simulated_annealing.py  # SA z geometric cooling
│       ├── pso.py                  # Particle Swarm Optimization
│       └── greedy.py               # Greedy Relocate (turbina po turbinie)
├── notebooks/                      # Analizy (VS Code # %% / Jupyter)
│   ├── 01_eda_wind.py              # Eksploracja danych wiatrowych
│   ├── 02_layouts.py               # Layouty i porównania modeli
│   ├── 03_optimization_aep.py      # Optymalizacja i analiza AEP
│   ├── 04_full_simulation.py       # Pełna symulacja 375 MW
│   ├── 05_advanced_floris.py       # AVP, derating, helix, sensitivity
│   └── 06_scipy_optimization.ipynb # Optymalizacja Scipy (natywny notebook)
├── data/                           # Dane wejściowe
│   ├── raw/                        # Surowe (ERA5 .nc cache, CSV)
│   ├── processed/                  # Przetworzone
│   └── turbines/                   # Custom turbiny (YAML)
└── outputs/                        # Wyniki (figures/, exports/, logs/)
```

---

## Architektura (jak to działa)

Przepływ danych jest jednokierunkowy: **dane wiatrowe → model farmy → analiza/optymalizacja → eksport**.

```
WindDataLoader              FarmModel                    Analiza / Optymalizacja
─────────────               ─────────                    ───────────────────────
Mock (Weibull)  ─┐          buduje FlorisModel           Optimizer (Scipy/RS/Yaw)
ERA5 (cdsapi)   ─┼─► TimeSeries / WindRose ─► (wake + turbina + layout) ─┬─► AEPCalculator
CSV             ─┘                                                       ├─► algorithms/* (Lab)
                                                                         └─► ReportGenerator (PDF)
```

**Kluczowe klasy:**

| Klasa | Plik | Odpowiedzialność |
|---|---|---|
| `WindDataLoader` | `src/wind_data.py` | Ładowanie/generowanie wiatru, konwersja do `TimeSeries`/`WindRose`/`WindTIRose`, statystyki, róża wiatrów, podział sezonowy |
| `FarmModel` | `src/farm_model.py` | Wrapper na `FlorisModel`: przełączanie modeli wake/turbin, generatory layoutów, flow field, straty wake, eksport layoutu |
| `Optimizer` | `src/optimizer.py` | `LayoutOptimizationScipy`, `LayoutOptimizationRandomSearch`, `YawOptimizationSR` + wizualizacja przed/po |
| `AEPCalculator` | `src/aep_calculator.py` | AEP roczne/sezonowe/miesięczne, analiza niepewności, porównanie scenariuszy, eksport dla Gr5 |
| `ReportGenerator` | `src/report_generator.py` | Raport PDF (reportlab, font z polskimi znakami) |
| `LayoutAlgorithm` | `src/algorithms/base.py` | Wspólny interfejs algorytmów + `AlgorithmResult` + helpery (feasibility, repair, random layout) |

**Wydajność w Streamlit:** budowa `FlorisModel` (ładowanie YAML) jest cache'owana (`@st.cache_resource`), a AEP przeliczane tylko gdy zmieni się konfiguracja (klucz w `session_state`). Dzięki temu kliknięcia w UI nie wywołują kosztownych przeliczeń FLORIS.

---

## Streamlit — zakładki

Kolejność i widoczność zależą od roli. Admin może ukrywać/pokazywać karty w panelu bocznym.

| # | Zakładka | Opis | Rola |
|---|---|---|---|
| 1 | 🌬️ **Wiatr** | Źródło danych (Mock/ERA5), parametry Weibulla, **globalny tryb wiatru obliczeniowego** (wąski bin / pełna róża), róża wiatrów, statystyki | viewer |
| 2 | 🔧 **Turbiny** | Wybór turbiny aktywnej + parametry fal (floating) | viewer |
| 3 | 📐 **Layout** | Model wake + typ layoutu (grid/staggered/circular/parallelogram) + podgląd + edytor współrzędnych/CSV | viewer |
| 4 | ⚖️ **Porównania** | Modele wake, turbiny, spacing sweep | admin |
| 5 | 🏆 **Benchmark** | Wiele konfiguracji naraz (wake × layout × spacing) + flow field obok siebie | admin |
| 6 | 🎯 **Optymalizacja** | Scipy gradient + Random Search (pełna róża) **oraz 🧪 Lab algorytmów** (7 algorytmów na wąskim binie) | admin |
| 7 | 🤝 **Grupa 3** | Baza pod Temat 3: yaw, yaw_max sweep, curtailment, Helix, porównanie strategii, eksport | admin |
| 8 | ⚡ **AEP** | Analiza sezonowa, miesięczna, niepewność | viewer |
| 9 | 💰 **Ekonomia (Gr5)** | CAPEX/OPEX, LCOE, NPV, IRR, payback, dyskonto, eskalacja, degradacja | admin |
| 10 | 🌐 **3D** | Wizualizacje Plotly (layout, profil wiatru, mapa mocy) | admin |
| 11 | 📄 **Raport / Eksport** | Generowanie PDF + `layout.csv` / `aep_timeseries.csv` | admin |
| 12 | 🗑️ **Śmietnik** | Sekcje wycofane z nawigacji, ale działające (Przegląd farmy, Flow field 1-bin) | admin |

---

## 🧪 Lab algorytmów — dodanie własnego algorytmu

Każdy algorytm to osobny plik w `src/algorithms/`, dziedziczący po `LayoutAlgorithm`. Algorytm dostaje `FarmModel` z ustawionym wiatrem (wąski bin), prostokątne `bounds_rect`, `min_dist` i `eval_budget`, a zwraca standardowy `AlgorithmResult` (z historią zbieżności do wykresu).

**1. Stwórz `src/algorithms/moj_algo.py`:**
```python
from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep

class MojAlgorytm(LayoutAlgorithm):
    name = "Mój Algorytm"
    description = "Krótki opis działania"
    # Opcjonalnie: parametry → UI wygeneruje suwaki
    PARAMS = [
        {"key": "alpha", "label": "Alpha", "type": "float",
         "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.1},
    ]

    def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params):
        # ... twoja logika; ewaluuj layout przez evaluate_aep(farm, x, y) ...
        return AlgorithmResult(
            name=self.name,
            initial_x=..., initial_y=..., initial_aep=...,
            final_x=..., final_y=..., final_aep=...,
            elapsed_s=..., n_evaluations=..., history=[...],
        )
```

**2. Zarejestruj w `src/algorithms/__init__.py`:**
```python
from .moj_algo import MojAlgorytm
ALGORITHMS["moj"] = MojAlgorytm
```

**3. Algorytm pojawi się automatycznie** w multiselectie zakładki **🧪 Lab algorytmów**.

Helpery dostępne w `base.py`: `evaluate_aep`, `is_feasible`, `repair_min_dist`, `random_feasible_layout`, `bounds_from_layout`.

---

## Dane wiatrowe

`WindDataLoader` obsługuje trzy źródła i zwraca natywne obiekty FLORIS (`TimeSeries`, `WindRose`, `WindTIRose`):

- **Mock (Weibull)** — prędkość z rozkładu Weibulla (k≈2.1, A≈9.5 m/s), kierunek z von Mises z sezonowością (zimą SW/W, latem NW), TI offshere 0.02–0.20. Powtarzalne (`seed`).
- **CSV** — `WindDataLoader.from_csv(...)` z konfigurowalnymi nazwami kolumn (działa z IMGW, ERA5 wyeksportowanym do CSV itd.).
- **ERA5 (Copernicus CDS)** — `WindDataLoader.from_era5(...)`, pobiera składowe wiatru na 100 m, ekstrapoluje logarytmicznie do hub height, cache w `data/raw/`.

### ERA5 — konfiguracja (opcjonalna)

Aby używać prawdziwych danych zamiast mock:

1. Załóż darmowe konto na [Copernicus CDS](https://cds.climate.copernicus.eu/)
2. Zapisz klucz API w `~/.cdsapirc` ([instrukcja](https://cds.climate.copernicus.eu/api-how-to))
3. W zakładce 🌬️ Wiatr wybierz "Źródło danych: ERA5 (Copernicus)"
4. Bez klucza appka automatycznie zrobi fallback do mock data

---

## Wyniki (mock data — Bałtyk Pd.)

| Parametr | Wartość |
|---|---|
| Lokalizacja | ~54.5°N, 16.5°E (Bałtyk Południowy) |
| Turbina | IEA 15 MW (D=240 m, hub=150 m) |
| Layout | 5×5 grid @ 7D = 25 turbin |
| Moc zainstalowana | 375 MW |
| AEP | ~1669 GWh |
| Capacity Factor | ~50.8% |
| Wake losses (GCH) | ~5.6% |
| Optymalizacja Scipy | +3.8% AEP |
| Yaw steering | +0.8% AEP |

---

## Zależności z grupą

Projekt badawczy na studia — analiza i optymalizacja rozmieszczenia turbin w farmie offshore dla różnych zestawów danych wiatrowych, z wykorzystaniem analitycznych modeli śladu aerodynamicznego.

- **Temat 1** → dostarcza dane wiatrowe (ERA5 / klimatyczne) → wbudowane `WindDataLoader.from_era5()`
- **Temat 2** → **(ten projekt)** optymalizacja layoutu farmy + Lab algorytmów
- **Temat 3** → wake steering — zakładka 🤝 Grupa 3 (yaw, curtailment, helix; baza gotowa); wejście: `layout.csv`
- **Temat 4** → obciążenia łopat HAWC; wejście: `layout.csv`
- **Temat 5** → analiza ekonomiczna — zakładka 💰 Ekonomia; wejście: `aep_timeseries.csv`

---

## Rozwiązywanie problemów

**`ImportError: DLL load failed while importing lib` (pyarrow) — "Zasady kontroli aplikacji zablokowały ten plik"**
Komputer (zwykle firmowy) blokuje DLL biblioteki `pyarrow` przez politykę WDAC/AppLocker. Aplikacja wykrywa to przy starcie i automatycznie przełącza tabele w **tryb awaryjny (HTML)** — działa dalej, choć interaktywny edytor tabel zastępuje edytowalne pole CSV. Trwałe rozwiązanie: poproś IT o whitelist `pyarrow` albo zainstaluj Pythona poza folderem użytkownika (np. `C:\ProgramData\miniconda3`).

**Brakujące biblioteki przy starcie**
Aplikacja pokaże dokładne polecenie `pip install ...`. Wymagane: floris, numpy, pandas, scipy, matplotlib, streamlit, plotly, reportlab. Opcjonalne (ERA5): cdsapi, xarray, netcdf4.

**Ostrzeżenia FLORIS "Computing AEP with uniform frequencies"**
Nieszkodliwe — pojawiają się na wąskim binie (1 warunek wiatrowy). Są wyciszane w logach.

**ERA5 nie pobiera danych**
Sprawdź `~/.cdsapirc` i akceptację warunków datasetu na portalu CDS. Bez konfiguracji appka i tak działa na mock data.

---

## Technologie

- **FLORIS v4.6** — silnik wake modeling (NREL)
- **Python 3.11**
- **Streamlit** — interfejs webowy
- **Matplotlib + Plotly** — wizualizacje 2D/3D
- **NumPy / Pandas / SciPy** — obliczenia
- **reportlab** — raport PDF
- **cdsapi + xarray + netcdf4** — dane ERA5 (opcjonalnie)

---

## Roadmap

- [x] Modularny kod (5 modułów + `src/algorithms/`)
- [x] 6 notebooków z analizami
- [x] Streamlit dashboard (logowanie + role + 12 zakładek)
- [x] Porównanie modeli wake i turbin + benchmark
- [x] Optymalizacja layoutu (Scipy, Random Search) i yaw (SerialRefine)
- [x] 🧪 Lab algorytmów — 7 algorytmów do porównania na wąskim binie
- [x] 🤝 Zakładka Grupa 3 — baza pod wake steering
- [x] 💰 Analiza ekonomiczna (Grupa 5) — LCOE / NPV
- [x] ERA5 jako alternatywa dla mock data
- [x] Custom turbiny (YAML) + turbiny pływające
- [x] Wizualizacje 3D (Plotly) + raport PDF
- [x] Analiza AEP (sezonowa, miesięczna, niepewność)
- [x] Eksport danych dla grupy
- [x] Sprawdzanie bibliotek przy starcie + fallback bez pyarrow
- [x] Deployment na Streamlit Cloud
</content>
</invoke>
