# Digiwind
# Wind Farm Layout Optimization

**Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych**

Narzędzie do optymalizacji rozmieszczenia turbin wiatrowych offshore na Bałtyku Południowym. Wykorzystuje bibliotekę [FLORIS](https://github.com/NREL/floris) (NREL) do modelowania śladów aerodynamicznych i maksymalizacji rocznej produkcji energii (AEP).

🔗 **[Streamlit Dashboard](https://digiwind-fyhbp3zaekx3dabefrappyk.streamlit.app/)** — interaktywna aplikacja webowa

---

## Funkcje

- **5 modeli wake** — Jensen, GCH (Gauss), TurbOPark, Empirical Gauss, CC
- **4 turbiny referencyjne** — NREL 5MW, IEA 10MW, IEA 15MW, IEA 22MW
- **3 typy layoutów** — siatka regularna, siatka przesunięta (staggered), kołowy
- **Optymalizacja layoutu** — Scipy (gradient-based) + optymalizacja yaw (wake steering)
- **Analiza AEP** — roczna, sezonowa, miesięczna, z analizą niepewności
- **Porównania** — modele wake, turbiny, rozstawy (spacing sweep), scenariusze
- **Zaawansowane** — AVP (Annual Value Production), derating turbin, Helix control (Active Wake Mixing), analiza wrażliwości parametrów
- **Eksport danych** — `layout.csv` i `aep_timeseries.csv` dla innych tematów w grupie
- **Streamlit Dashboard** — interaktywna aplikacja z 7 zakładkami

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
git clone https://github.com/your-username/wind-farm-optimization.git
cd wind-farm-optimization

# Stwórz środowisko conda
conda create -n windfarm python=3.11 -y
conda activate windfarm

# Zainstaluj zależności
pip install -r requirements.txt
```

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
wind_farm_project/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Zależności Python
├── src/                            # Moduły logiki
│   ├── wind_data.py                # WindDataLoader — dane wiatrowe
│   ├── farm_model.py               # FarmModel — symulacja FLORIS
│   ├── optimizer.py                # Optimizer — optymalizacja layoutu/yaw
│   └── aep_calculator.py           # AEPCalculator — analiza produkcji energii
├── notebooks/                      # Notebooki Jupyter (VS Code)
│   ├── 01_eda_wind.py              # Eksploracja danych wiatrowych
│   ├── 02_layouts.py               # Layouty i porównania modeli
│   ├── 03_optimization_aep.py      # Optymalizacja i analiza AEP
│   ├── 04_full_simulation.py       # Pełna symulacja 375 MW
│   └── 05_advanced_floris.py       # Zaawansowane funkcje FLORIS
├── data/                           # Dane wejściowe
│   ├── raw/                        # Surowe dane (ERA5, CSV)
│   ├── processed/                  # Przetworzone dane
│   └── turbines/                   # Pliki konfiguracyjne turbin
└── outputs/                        # Wyniki
    ├── figures/                    # Wykresy PNG
    └── exports/                    # CSV dla grupy
```

## Technologie

- **FLORIS v4.6** — silnik wake modeling (NREL)
- **Python 3.11** — język programowania
- **Streamlit** — interfejs webowy
- **Matplotlib** — wizualizacje
- **NumPy / Pandas / SciPy** — obliczenia

## Kontekst projektu

Projekt badawczy na studia — analiza produktywności i optymalizacja rozmieszczenia turbin wiatrowych w farmie offshore dla różnych zestawów danych wiatrowych z wykorzystaniem modeli analitycznych śladu aerodynamicznego (wake effect).

### Zależności z grupą
- **Temat 1** → dostarcza dane wiatrowe (ERA5 / klimatyczne)
- **Temat 2** → optymalizacja layoutu farmy
- **Temat 3** → wake steering (`layout.csv`)
- **Temat 4** → obciążenia łopat HAWC (`layout.csv`)
- **Temat 5** → analiza ekonomiczna (`aep_timeseries.csv`)

## Roadmap

- [x] Modularny kod (4 moduły w `src/`)
- [x] 5 notebooków z analizami
- [x] Streamlit dashboard
- [x] Porównanie modeli wake i turbin
- [x] Optymalizacja layoutu (Scipy) i yaw (SerialRefine)
- [x] Analiza AEP (sezonowa, miesięczna, niepewność)
- [x] Zaawansowane: AVP, derating, helix, analiza wrażliwości
- [x] Eksport danych dla grupy
- [ ] Wizualizacje 3D (Plotly / VTK)
- [ ] Turbiny pływające (floating offshore)
- [ ] Dodawanie własnych turbin przez Streamlit
- [ ] Automatyczne generowanie raportu
- [x] Deployment na Streamlit Cloud

