# Baseline AEP — Baltica 2 na wietrze IMGW (21.11.2026)

Layout bazowy: 107 × IEA-15-240-RWT (1605 MW), siatka 4,7 D obrócona do 260°,
w granicy Baltica 2 z raportu EIA (137 km²). Wiatr: IMGW (WRF + MBCn), 150 m,
1991–2010, krok 3 h. Róża 10° × 2 m/s od 0 m/s. AEP = tylko straty wake
(bez dostępności, strat elektrycznych itd.). Skrypt: `scripts/baseline_aep.py`.

## 1. Wybór punktów IMGW (GCH, TI 0,06, α z danych)

| Punkty | Średni wiatr | α | AEP [GWh] | CF | Straty wake |
|---|---|---|---|---|---|
| **P47, P62, P63, P35 (dookoła farmy) — domyślne** | 9,65 m/s | 0,059 | **7443** | 52,9% | 10,0% |
| P47, P33, P20, P35 (południe/zachód) | 9,58 m/s | 0,061 | 7351 | 52,3% | 10,3% |
| 6 najbliższych | 9,65 m/s | 0,058 | 7435 | 52,9% | 10,1% |

Średnia ważona 1/d² składowych u, v. Wagi: P35 0,29 · P62 0,27 · P47 0,25 · P63 0,19.

## 2. Modele wake (TI 0,06)

| Model | AEP [GWh] | CF | Straty wake |
|---|---|---|---|
| GCH | 7443 | 52,9% | 10,0% |
| Empirical Gauss | 7287 | 51,8% | 11,9% |
| TurbOPark | 6614 | 47,0% | 20,1% |

## 3. Wrażliwość na TI (GCH)

| TI | AEP [GWh] | Straty wake |
|---|---|---|
| 0,04 | 7304 | 11,7% |
| 0,06 | 7443 | 10,0% |
| 0,08 | 7551 | 8,7% |
| 0,10 | 7632 | 7,8% |

## 4. Wykładnik α (GCH)

α z danych (0,059): 7443 GWh · domyślne FLORIS 0,12: 7427 GWh (−0,2%).
