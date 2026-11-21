"""Geometria Baltica 2: UTM w obie strony, granica, layout bazowy 107 turbin.
Uruchomienie: python tests/t3_site.py <ścieżka_do_repo>
"""
import sys
import numpy as np

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from scipy.spatial.distance import pdist
from src.site_baltica2 import (baltica2_site, latlon_to_utm, utm_to_latlon,
                               rotated_grid_layout, layout_table)

# Punkt kontrolny: 55°N 15°E (południk osiowy strefy 33) → E = 500000 dokładnie
e, n = latlon_to_utm(55.0, 15.0)
print(f"55N 15E -> E={e:.3f} N={n:.3f}  (oczekiwane E=500000.000, N≈6094791)")
lat = np.random.default_rng(0).uniform(54.9, 55.2, 1000)
lon = np.random.default_rng(1).uniform(16.9, 17.6, 1000)
la2, lo2 = utm_to_latlon(*latlon_to_utm(lat, lon))
err_m = np.hypot((la2 - lat) * 111200, (lo2 - lon) * 63800).max()
print(f"round-trip max error: {err_m*1000:.4f} mm")

site = baltica2_site()
print(f"area {site.area_km2:.1f} km², bounds (local m) {np.round(site.bounds_rect)}")
print(f"origin UTM E={site.origin_e} N={site.origin_n}")

x, y, info = rotated_grid_layout(site)
D = 240.0
print("layout:", len(x), "turbin", {k: round(v, 2) for k, v in info.items()})
print(f"min spacing {pdist(np.column_stack([x, y])).min()/D:.2f}D, all inside: {site.contains(x, y).all()}")
df = layout_table(site, x, y)
print(df.head(3).to_string(index=False))
assert len(x) == 107 and site.contains(x, y).all() and err_m < 1e-3
print("OK")
