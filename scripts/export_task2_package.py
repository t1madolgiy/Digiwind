"""Paczka Task 2 dla innych grup: layout.csv, floris_config.yaml, wind_rose.csv.

  layout.csv          — 107 turbin: id, UTM 33N (EPSG:32633), lat/lon WGS84, x/y lokalne, typ
  floris_config.yaml  — pełna konfiguracja FLORIS 4.6.4 (GCH, IEA 15 MW, α z IMGW,
                        layout w metrach lokalnych = UTM minus origin z layout.csv)
  wind_rose.csv       — róża IMGW 1991–2010, 150 m, 10° × 2 m/s: wd, ws, freq, ti

Na końcu sprawdza, że YAML wczytuje się do FlorisModel i daje to samo AEP.
Uruchomienie:  python scripts/export_task2_package.py <katalog_docelowy>
"""
import sys, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.disable(logging.WARNING)

from floris import FlorisModel, WindRose
from src.wind_data import WindDataLoader
from src.farm_model import FarmModel
from src.site_baltica2 import baltica2_site, rotated_grid_layout, layout_table


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / "exports" / "Task2_pliki"
    out.mkdir(parents=True, exist_ok=True)

    site = baltica2_site()
    x, y, info = rotated_grid_layout(site, n_turbines=107, D=240.0)

    # 1. layout.csv
    lay = layout_table(site, x, y, turbine_type="IEA-15-240-RWT")
    lay.to_csv(out / "layout.csv", index=False)

    # 3. wind_rose.csv (przed YAML — potrzebna do kontroli AEP)
    L = WindDataLoader.from_imgw()
    rose = L.to_wind_rose(wd_step=10.0, ws_step=2.0)
    WD, WS = np.meshgrid(rose.wind_directions, rose.wind_speeds, indexing="ij")
    wr = pd.DataFrame({"wd_deg": WD.ravel(), "ws_ms": WS.ravel(), "freq": rose.freq_table.ravel()})
    wr = wr[wr.freq > 0].reset_index(drop=True)
    wr["ti"] = 0.06
    wr["freq"] = wr.freq.round(8)
    wr.to_csv(out / "wind_rose.csv", index=False)

    # 2. floris_config.yaml — model taki jak w naszych obliczeniach
    f = FarmModel("gch", "iea_15MW")
    f.set_layout_custom(x, y, name="baltica2_base")
    f.set_wind_shear(L.wind_shear)
    f.set_wind_data(rose); f.run()
    aep_ref = f.get_aep_gwh()
    # w YAML jeden reprezentatywny warunek zamiast całej róży (róża jest w CSV)
    f.fmodel.set(wind_directions=np.array([260.0]), wind_speeds=np.array([9.0]),
                 turbulence_intensities=np.array([0.06]))
    yaml_path = out / "floris_config.yaml"
    cfg = f.fmodel.core.as_dict()
    # lokalna ścieżka do biblioteki turbin (Path) — nieprzenośna; FLORIS i tak
    # domyślnie używa wbudowanej biblioteki, w której jest iea_15MW
    cfg["farm"].pop("turbine_library_path", None)
    import yaml
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    yaml_path.write_text(body, encoding="utf-8")
    header = (
        "# Task 2 - Baltica 2 baseline, 107 x IEA-15-240-RWT, FLORIS 4.6.4\n"
        "# Layout in local metres: x = UTM33N E - {oe:.0f}, y = UTM33N N - {on:.0f} (see layout.csv)\n"
        "# Wake model GCH; wind_shear from IMGW 150 m 1991-2010; full wind rose in wind_rose.csv\n"
    ).format(oe=site.origin_e, on=site.origin_n)
    yaml_path.write_text(header + yaml_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Kontrola: YAML + wind_rose.csv → to samo AEP
    fm = FlorisModel(str(yaml_path))
    w = pd.read_csv(out / "wind_rose.csv")
    wd_u, ws_u = np.sort(w.wd_deg.unique()), np.sort(w.ws_ms.unique())
    ft = np.zeros((len(wd_u), len(ws_u)))
    ft[np.searchsorted(wd_u, w.wd_deg), np.searchsorted(ws_u, w.ws_ms)] = w.freq
    fm.set(wind_data=WindRose(wind_directions=wd_u, wind_speeds=ws_u, freq_table=ft, ti_table=0.06))
    fm.run()
    aep_check = fm.get_farm_AEP() / 1e9

    print(f"layout.csv: {len(lay)} turbines, spacing {info['spacing_cross_D']:.2f} D")
    print(f"wind_rose.csv: {len(wr)} bins, freq sum {wr.freq.sum():.6f}")
    print(f"floris_config.yaml: wind_shear {L.wind_shear:.3f}")
    print(f"AEP check: our model {aep_ref:.1f} GWh, from exported files {aep_check:.1f} GWh "
          f"(diff {100 * (aep_check / aep_ref - 1):+.3f}%)")
    print("written to", out)


if __name__ == "__main__":
    main()
