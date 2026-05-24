"""
Pobieranie danych ERA5 do lokalnego cache — skrypt offline.
============================================================

Po co osobny skrypt?
    Pobranie wielu lat ERA5 trwa długo (kolejka CDS + transfer netcdf).
    Streamlit blokowałby się na cały ten czas. Lepiej raz pobrać do
    `data/raw/`, a aplikacja wczyta gotowy cache przez `from_era5_local()`.

Wymagania:
    1. pip install cdsapi xarray netcdf4
    2. Klucz API w ~/.cdsapirc (Linux/Mac) lub C:/Users/<ty>/.cdsapirc (Windows).
       Instrukcja: https://cds.climate.copernicus.eu/how-to-api
    3. Zaakceptowana licencja datasetu "ERA5 hourly data on single levels":
       https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels

Użycie:
    # Bałtyk Południowy, ostatnie 5 lat, domyślny cache data/raw/
    python scripts/download_era5.py --years 2020 2021 2022 2023 2024

    # Inna lokalizacja, jeden rok, mniejszy box (mniejszy cost requesta)
    python scripts/download_era5.py --lat 55.0 --lon 17.5 --years 2023 --area 0.05

    # Inny katalog cache
    python scripts/download_era5.py --years 2023 --output-dir data/era5_test

Skrypt jest BEZPIECZNY do wielokrotnego uruchomienia — pomija już pobrane pliki.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Pozwala uruchamiać skrypt z głównego katalogu projektu (`python scripts/download_era5.py`)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wind_data import WindDataLoader  # noqa: E402


def _format_size(n_bytes: int) -> str:
    """Czytelny rozmiar (KB/MB/GB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Czytelny czas (s/min/h)."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pobiera dane ERA5 do lokalnego cache (miesiąc po miesiącu).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przykład:\n"
            "  python scripts/download_era5.py --lat 54.5 --lon 16.5 "
            "--years 2022 2023 2024\n"
        ),
    )
    parser.add_argument("--lat", type=float, default=54.5, help="Szerokość [°] (default: 54.5 = Bałtyk Pd.)")
    parser.add_argument("--lon", type=float, default=16.5, help="Długość [°] (default: 16.5)")
    parser.add_argument(
        "--years", type=int, nargs="+", required=True,
        help="Lista lat do pobrania, np. --years 2022 2023 2024",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Katalog cache (default: data/raw)",
    )
    parser.add_argument(
        "--area", type=float, default=0.1, dest="area_halfwidth",
        help="Połowa szerokości boxa wokół (lat, lon) [°]. Mniejszy = tańszy request. (default: 0.1)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Liczba prób per miesiąc przy błędach sieciowych (default: 3)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Pełne logi (DEBUG zamiast INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # cdsapi sam loguje sporo INFO — w trybie nie-verbose tłumimy do WARNING
    if not args.verbose:
        logging.getLogger("cdsapi").setLevel(logging.WARNING)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_months = len(args.years) * 12
    print("=" * 60)
    print(f"ERA5 download — {len(args.years)} lat × 12 miesięcy = {total_months} requestów")
    print(f"  Lokalizacja:  lat={args.lat}°, lon={args.lon}°")
    print(f"  Box:          ±{args.area_halfwidth}° (≈{args.area_halfwidth * 111:.0f} km)")
    print(f"  Lata:         {args.years}")
    print(f"  Cache:        {out_dir}")
    print("=" * 60)

    def _progress(done, total, label):
        bar_w = 30
        filled = int(bar_w * done / total)
        bar = "█" * filled + "░" * (bar_w - filled)
        pct = 100.0 * done / total
        # \r żeby nadpisywać tę samą linię
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {label:40s}")
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")

    t0 = time.perf_counter()
    try:
        files, failed = WindDataLoader.download_era5_months(
            latitude=args.lat,
            longitude=args.lon,
            years=args.years,
            output_dir=out_dir,
            max_retries=args.max_retries,
            area_halfwidth=args.area_halfwidth,
            progress_callback=_progress,
        )
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Przerwane przez użytkownika. Już pobrane pliki zostały w cache.")
        return 130
    except Exception as e:
        print(f"\n\n  ❌  Niepowodzenie: {e}")
        return 1

    elapsed = time.perf_counter() - t0
    total_size = sum(f.stat().st_size for f in files if f.exists())

    print("\n" + "=" * 60)
    print("Podsumowanie:")
    print(f"  Pobrano:     {len(files)} / {total_months} plików")
    print(f"  Rozmiar:     {_format_size(total_size)}")
    print(f"  Czas:        {_format_duration(elapsed)}")
    if failed:
        print(f"  Błędy:       {len(failed)} miesięcy")
        for year, month, err in failed[:5]:
            print(f"    - {year}-{month:02d}: {err[:80]}")
        if len(failed) > 5:
            print(f"    ... i {len(failed) - 5} więcej (sprawdź logi)")
    print("=" * 60)
    print(f"  Cache:       {out_dir}")
    print("  W Streamlit:  użyj 'ERA5 (lokalny cache)' jako źródła danych.")
    print("=" * 60)

    return 0 if files else 1


if __name__ == "__main__":
    sys.exit(main())
