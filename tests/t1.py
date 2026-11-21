"""Błędy #1 i #2: róża vs szereg czasowy, suma miesięcy/sezonów vs rok.
Uruchomienie: python tests/t1.py <ścieżka_do_repo>
"""
import sys, logging
import numpy as np

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
logging.disable(logging.WARNING)

from src.wind_data import WindDataLoader
from src.farm_model import FarmModel
from src.aep_calculator import AEPCalculator

L = WindDataLoader(); ts = L.generate_mock_data()
print("mock mean ws", L.wind_speeds.mean().round(2),
      "frac<3:", (L.wind_speeds < 3).mean().round(3),
      "frac>25:", (L.wind_speeds > 25).mean().round(4))

wr = L.to_wind_rose()
print("WindRose freq sum:", wr.freq_table.sum().round(4), "ws centers", wr.wind_speeds)

farm = FarmModel("gch", "iea_15MW"); farm.set_layout_grid(5, 5, 7.0)
farm.set_wind_data(wr); farm.run(); a_rose = farm.get_aep_gwh()

# referencja: AEP z pełnego szeregu czasowego (każdy rekord waga 1/N)
farm.set_wind_data(ts); farm.run()
P = farm.fmodel.get_farm_power()
a_ts = np.nansum(P) / len(P) * 8760 / 1e9
print(f"AEP rose(default)={a_rose:.1f}  AEP timeseries exact={a_ts:.1f}  diff={100*(a_rose/a_ts-1):+.2f}%")

wr2 = L.to_wind_rose(ws_min=0.0, ws_max=36.0)
print("rose 0-36 freq sum", wr2.freq_table.sum().round(4))
farm.set_wind_data(wr2); farm.run(); print(f"AEP rose(0-36)={farm.get_aep_gwh():.1f}")

farm.set_wind_data(wr)
calc = AEPCalculator(farm)
s = calc.compute_seasonal_aep(L); m = calc.compute_monthly_aep(L)
print(s[["season", "n_hours", "fraction", "aep_gwh"]].round(3).to_string())
print("sum seasonal aep:", s.aep_gwh.sum().round(1), " sum monthly aep:", m.aep_gwh.sum().round(1),
      " annual:", round(a_rose, 1))
