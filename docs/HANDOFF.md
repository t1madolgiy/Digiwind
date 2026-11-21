# Przekazanie pracy — Task 2 (stan 21.11.2026)

Ten plik jest dla nowej sesji Claude Code uruchomionej LOKALNIE na laptopie.
Pełny dokument przekazania (decyzje, dane, błędy, plan): `docs/przekazanie_task2.txt`.

## Stan repo
- Branch roboczy: `claude/wizardly-cori-g37ur1` = `Ver3_Gr2+3` (05.06) + `laptop1` (commit „ERA5”, 24.05), scalone bez konfliktów.
- `main` jest stary (V1, kwiecień) — nie używać.
- `.gitignore` blokuje duże pliki danych w `data/raw/` (zip, IMGW*/, ERA5*/).

## Pliki na dysku laptopa (poza repo), folder `Pulpit\DIGIWIND\`
- `IMGW_positive_version_wind_profile.zip` (2,7 GB) — dane IMGW od Klaudii (T1). Wg wiadomości Klaudii:
  bez 10-letnich przerw, kolumna źródła (historyczne / prognoza IMGW / jej model), profil wiatru na 150 m,
  alfa z prawa potęgowego. Brak nasłonecznienia itp.
- `Proga kabarowy laptop-20261121T12153...zip` (2,6 MB) — kopia folderu z drugiego laptopa. SPRAWDZIĆ czy jest tam kod nowszy niż repo i scalić.
- `ERA5_1940_2025_combined-001.zip` (7 GB) + `ERA5data-...zip` — ERA5 od Klaudii (tylko do testów).

## Cel na weekend: zamknąć Task 2
1. Faza 0 — poprawki błędów (szczegóły: sekcja 6.3 i 7.2 w przekazanie_task2.txt):
   róża od 0 m/s (AEP +9,2% zawyżone), AEP miesięczne/sezonowe jako energia (suma = rok),
   is_feasible na wynikach algorytmów (SLSQP daje 0,00D), random_feasible_layout różny per seed,
   eksporty zawsze z pełnej róży, hasła admin/digiwind2025 → secrets.toml, floris==4.6.4.
2. Dane IMGW: wyciągnąć punkt(y) Baltica 2 do `data/processed/` (mały plik, commit), `WindDataLoader.from_imgw()`
   (kalendarz noleap → cftime, krok 3 h, height == 150), alfa z profilu → `flow_field.wind_shear`, TI stałe 0,06.
3. Baltica 2: prostokąt z OOŚ 55°00'07,509"–55°06'51,280"N, 17°01'50,132"–17°27'20,251"E → UTM 33N (EPSG:32633),
   107 × IEA-15-240-RWT (hub 150 m), siatka obrócona do dominującego kierunku.
4. Baseline (M1): AEP, CF, wake loss — GCH vs TurbOPark vs Empirical Gauss; wrażliwość TI 0,04–0,10.
   Eksport: layout.csv (id, x/y UTM, lat, lon, typ), floris_config.yaml, aep_timeseries.csv (rok/miesiące/sezony), v(t) per turbina (T6).
5. Optymalizacja (M2/finał): ograniczenia wielokątne (shapely), GA vs SLSQP vs hybryda GA→SLSQP, 5 seedów,
   szukanie na róży 30°×3 m/s, ocena na 10°×2 m/s. Finalny layout + porównanie z baseline.
6. Wyniki do prezentacji / raportu.

## Uwagi
- Windows/PowerShell: nie łączyć komend przez `&&`. RandomSearch/ParFlorisModel uruchamiać ze skryptu z `if __name__ == "__main__"`.
- Commity pushować na `claude/wizardly-cori-g37ur1`.
