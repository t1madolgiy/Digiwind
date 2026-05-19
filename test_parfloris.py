"""Test ParFlorisModel — porównanie czasu vs FlorisModel."""
import time
import numpy as np
from floris import FlorisModel, ParFlorisModel, WindRose

# Setup
fmodel = FlorisModel("inputs/gch.yaml")  # albo inny config
n_rows, n_cols = 5, 5
D = 240  # IEA 15MW
sp = 7 * D
xx, yy = np.meshgrid(np.arange(n_cols) * sp, np.arange(n_rows) * sp)
layout_x = xx.flatten()
layout_y = yy.flatten()

# WindRose 36 kierunków × 5 prędkości
wd = np.arange(0, 360, 10)
ws = np.array([5, 7, 9, 11, 13])
freq = np.ones((len(wd), len(ws))) / (len(wd) * len(ws))
wr = WindRose(wind_directions=wd, wind_speeds=ws, freq_table=freq, ti_table=0.06)

# --- Test 1: FlorisModel ---
fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wr)
t0 = time.perf_counter()
fmodel.run()
aep_seq = fmodel.get_farm_AEP()
t_seq = time.perf_counter() - t0
print(f"FlorisModel:    {t_seq:.2f}s, AEP={aep_seq/1e9:.2f} GWh")

# --- Test 2: ParFlorisModel ---
try:
    pfmodel = ParFlorisModel("inputs/gch.yaml", max_workers=8)
    pfmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=wr)
    t0 = time.perf_counter()
    pfmodel.run()
    aep_par = pfmodel.get_farm_AEP()
    t_par = time.perf_counter() - t0
    print(f"ParFlorisModel: {t_par:.2f}s, AEP={aep_par/1e9:.2f} GWh")
    print(f"Przyspieszenie: {t_seq/t_par:.1f}×")
except Exception as e:
    print(f"ParFlorisModel BŁĄD: {e}")