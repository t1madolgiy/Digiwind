"""
Wind Farm Layout Optimization — Streamlit Dashboard v3
=======================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Nowe w v3:
    - Custom turbiny (formularz → YAML → TURBINE_LIBRARY)
    - Turbiny pływające (IEA 15MW Floating + parametry fal)
    - Generowanie raportu PDF
    - Wizualizacja 3D (Plotly)

Uruchomienie:
    cd wind_farm_project
    streamlit run app.py
"""

import sys
import warnings
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from src.wind_data import WindDataLoader, BalticWindConfig
from src.farm_model import (
    FarmModel, WAKE_MODELS, TURBINE_LIBRARY, FLOATING_TURBINES,
    register_custom_turbine, load_custom_turbines_from_dir,
    generate_power_curve,
)
from src.optimizer import Optimizer
from src.aep_calculator import AEPCalculator
from src.report_generator import ReportGenerator
from src.algorithms import ALGORITHMS, bounds_from_layout

import floris
from floris import TimeSeries
from floris.utilities import load_yaml
import floris.layout_visualization as layoutviz

FLORIS_DIR = Path(floris.__file__).parent

# Załaduj custom turbiny z data/turbines/ przy starcie
load_custom_turbines_from_dir()


# =====================================================================
# HELPERS — zapis figury do session_state
# =====================================================================
def fig_to_bytes(fig):
    """Konwertuje matplotlib Figure do bytes (PNG) do przechowania w session_state."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="none")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def show_stored_fig(key, caption=None):
    """Wyświetla figurę z session_state jeśli istnieje."""
    if key in st.session_state:
        st.image(st.session_state[key], use_container_width=True, caption=caption)
        return True
    return False


def show_stored_df(key):
    """Wyświetla DataFrame z session_state jeśli istnieje."""
    if key in st.session_state:
        st.dataframe(st.session_state[key], use_container_width=True)
        return True
    return False


def show_stored_text(key):
    """Wyświetla tekst z session_state jeśli istnieje."""
    if key in st.session_state:
        st.info(st.session_state[key])
        return True
    return False


# =====================================================================
# KONFIGURACJA STRONY
# =====================================================================
st.set_page_config(
    page_title="Wind Farm Optimizer — Temat 2",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.title("⚙️ Konfiguracja")

    st.header("Turbina")
    turbine_name = st.selectbox(
        "Model turbiny",
        options=list(TURBINE_LIBRARY.keys()),
        index=2,
        format_func=lambda x: (
            f"{TURBINE_LIBRARY[x]['name']} (D={TURBINE_LIBRARY[x]['diameter']:.0f}m)"
            + (" 🌊" if TURBINE_LIBRARY[x].get("floating") else "")
            + (" ⭐" if TURBINE_LIBRARY[x].get("custom_yaml") else "")
        ),
    )
    turbine_info = TURBINE_LIBRARY[turbine_name]
    is_floating = turbine_name in FLOATING_TURBINES or turbine_info.get("floating", False)

    # Floating — parametry fal
    wave_period = 2.0
    wave_height = 1.0
    if is_floating:
        st.caption("🌊 Turbina pływająca — parametry fal")
        wave_period = st.select_slider("Okres fali Tp [s]", options=[2, 4], value=2)
        wave_height = st.select_slider("Wysokość fali Hs [m]", options=[1, 5], value=1)

    st.header("Model wake")
    wake_model = st.selectbox(
        "Model śladu aerodynamicznego",
        options=list(WAKE_MODELS.keys()),
        index=1,
        format_func=lambda x: x.upper(),
    )

    st.header("Layout farmy")
    layout_type = st.selectbox(
        "Typ layoutu",
        options=["grid", "staggered", "circular", "parallelogram"],
        format_func=lambda x: {
            "grid": "Siatka regularna",
            "staggered": "Siatka przesunięta",
            "circular": "Kołowy",
            "parallelogram": "Równoległobok (Malisani 2025)",
        }[x],
    )

    if layout_type in ["grid", "staggered"]:
        col1, col2 = st.columns(2)
        n_rows = col1.number_input("Rzędy", 2, 10, 5)
        n_cols = col2.number_input("Kolumny", 2, 10, 5)
    else:
        n_rows, n_cols = 4, 4

    spacing_D = st.slider("Rozstaw [×D]", 4.0, 15.0, 7.0, 0.5)

    if layout_type == "staggered":
        stagger_offset = st.slider("Offset", 0.0, 1.0, 0.5, 0.1)
    else:
        stagger_offset = 0.5

    if layout_type == "circular":
        n_ring_turbines = st.number_input("Turbin na pierścień", 4, 20, 12)
        n_rings = st.number_input("Pierścienie", 1, 3, 2)
    else:
        n_ring_turbines, n_rings = 12, 2

    # Parallelogram params (Malisani et al. 2025)
    if layout_type == "parallelogram":
        st.caption("Parametryzacja wg Malisani et al. (2025)")
        para_n = st.number_input("Liczba turbin", 4, 100, 25, key="para_n")
        col_p1, col_p2 = st.columns(2)
        para_r1 = col_p1.slider("r₁ [×D]", 2.0, 10.0, 7.0, 0.5, key="para_r1")
        para_r2 = col_p2.slider("r₂ [×D]", 2.0, 10.0, 7.0, 0.5, key="para_r2")
        para_t1 = col_p1.slider("θ₁ [°]", -89, 89, 88, 1, key="para_t1")
        para_t2 = col_p2.slider("θ₂ [°]", -89, 89, 18, 1, key="para_t2")
    else:
        para_n, para_r1, para_r2, para_t1, para_t2 = 25, 7.0, 7.0, 90, 18

    st.header("Dane wiatrowe")
    data_source = st.radio(
        "Źródło danych",
        ["Mock (Weibull)", "ERA5 (Copernicus)"],
        horizontal=True,
        help="ERA5 wymaga cdsapi+xarray+netcdf4 i klucza w ~/.cdsapirc. "
             "Lokalizacja domyślna: Bałtyk Południowy.",
    )

    if data_source == "Mock (Weibull)":
        weibull_A = st.slider("Weibull A [m/s]", 6.0, 14.0, 9.5, 0.5)
        weibull_k = st.slider("Weibull k", 1.5, 3.0, 2.1, 0.1)
        n_years = st.selectbox("Lata danych", [1, 2, 3], index=0)
        era5_lat = 54.5
        era5_lon = 16.5
        era5_years = (2023,)
        era5_hub_input = 150.0
    else:
        st.caption(
            "Pobranie z Copernicus CDS. Pierwsze ściągnięcie trwa kilka minut, "
            "kolejne są z lokalnego cache w `data/raw/`."
        )
        col_lat, col_lon = st.columns(2)
        era5_lat = col_lat.number_input("Szerokość [°N]", 50.0, 60.0, 54.5, 0.1)
        era5_lon = col_lon.number_input("Długość [°E]", 10.0, 20.0, 16.5, 0.1)
        era5_years = tuple(sorted(st.multiselect(
            "Lata", list(range(2015, 2025)), default=[2023],
        )))
        era5_hub_input = float(st.slider("Hub height [m]", 80, 200, 150, 10))
        # Wartości "ghost" — potrzebne dalej do klucza cache i fallbacku
        weibull_A = 9.5
        weibull_k = 2.1
        n_years = 1

    wr_resolution = st.selectbox(
        "Rozdzielczość WindRose",
        ["Gruby (30°)", "Dokładny (10°)", "Precyzyjny (5°)"],
        index=1,
    )


# =====================================================================
# INICJALIZACJA DANYCH
# =====================================================================
@st.cache_data
def generate_wind_data(weibull_A, weibull_k, n_years):
    config = BalticWindConfig(
        weibull_A=weibull_A, weibull_k=weibull_k,
        n_hours=8760 * n_years,
    )
    loader = WindDataLoader()
    loader.generate_mock_data(config=config, seed=42)
    return loader


@st.cache_data(show_spinner="Pobieram ERA5 (może chwilę potrwać przy pierwszym uruchomieniu)...")
def load_era5_wind(lat, lon, years_tuple, hub_height):
    """Próbuje pobrać dane ERA5. Zwraca (loader, error_msg). Loader=None przy błędzie."""
    try:
        loader = WindDataLoader.from_era5(
            latitude=float(lat),
            longitude=float(lon),
            years=list(years_tuple),
            hub_height=float(hub_height),
        )
        return loader, None
    except ImportError as e:
        return None, (
            f"Brak bibliotek do ERA5 ({e}). "
            "Zainstaluj: `pip install cdsapi xarray netcdf4`"
        )
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


era5_active = False
era5_error = None

if data_source == "Mock (Weibull)":
    loader = generate_wind_data(weibull_A, weibull_k, n_years)
elif not era5_years:
    st.sidebar.warning("⚠️ Wybierz co najmniej jeden rok ERA5. Używam mock.")
    loader = generate_wind_data(weibull_A, weibull_k, n_years)
else:
    loader, era5_error = load_era5_wind(era5_lat, era5_lon, era5_years, era5_hub_input)
    if loader is None:
        st.sidebar.error(f"❌ ERA5: {era5_error}\n\nUżywam mock data.")
        loader = generate_wind_data(weibull_A, weibull_k, n_years)
    else:
        era5_active = True
        st.sidebar.success(
            f"✅ ERA5: {len(era5_years)} lat, "
            f"{len(loader.wind_speeds)} rekordów, "
            f"@{era5_hub_input:.0f}m"
        )

wd_step = 30.0 if "30" in wr_resolution else (5.0 if "5" in wr_resolution else 10.0)
ws_step = 3.0 if "30" in wr_resolution else (1.0 if "5" in wr_resolution else 2.0)
wind_rose = loader.to_wind_rose(wd_step=wd_step, ws_step=ws_step)

farm = FarmModel(
    wake_model=wake_model,
    turbine=turbine_name,
    wind_data=wind_rose,
    wave_period=wave_period,
    wave_height=wave_height,
)

if layout_type == "grid":
    farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)
elif layout_type == "staggered":
    farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D, offset=stagger_offset)
elif layout_type == "circular":
    farm.set_layout_circular(n_turbines=n_ring_turbines, radius_D=spacing_D, n_rings=n_rings)
elif layout_type == "parallelogram":
    farm.set_layout_parallelogram(
        n_turbines=para_n, r1_D=para_r1, r2_D=para_r2,
        theta1_deg=float(para_t1), theta2_deg=float(para_t2),
    )

farm.set_wind_data(wind_rose)
farm.run()
aep = farm.get_aep_gwh()
rated_total = turbine_info["rated_power"] * farm.n_turbines
cf = aep / (rated_total * 8.76) * 100 if rated_total > 0 else 0

# --- Config key do czyszczenia session_state ---
_data_part = (
    f"era5_{era5_lat}_{era5_lon}_{era5_years}_{era5_hub_input}"
    if era5_active else f"mock_{weibull_A}_{weibull_k}_{n_years}"
)
_config_key = f"{turbine_name}_{wake_model}_{layout_type}_{spacing_D}_{n_rows}_{n_cols}_{_data_part}"
if st.session_state.get("_prev_config") != _config_key:
    # Konfiguracja się zmieniła — wyczyść stare wyniki
    keys_to_clear = [k for k in st.session_state.keys()
                     if k.startswith("fig_") or k.startswith("df_") or k.startswith("txt_")
                     or k in ("opt_result", "opt_final_aep", "yaw_result", "yaw_error",
                              "ed_results", "report_pdf", "report_ready")]
    for k in keys_to_clear:
        del st.session_state[k]
    st.session_state["_prev_config"] = _config_key


# =====================================================================
# TABS
# =====================================================================
(tab_overview, tab_wind, tab_flow, tab_compare, tab_benchmark, tab_optimize,
 tab_lab, tab_group3, tab_aep, tab_turbines, tab_editor, tab_3d, tab_report, tab_export) = st.tabs([
    "📊 Przegląd", "🌬️ Wiatr", "🌊 Flow field",
    "⚖️ Porównania", "🏆 Benchmark", "🎯 Optymalizacja",
    "🧪 Lab algorytmów", "🤝 Grupa 3",
    "⚡ AEP", "🔧 Turbiny", "✏️ Edytor", "🌐 3D", "📄 Raport", "📁 Eksport",
])


# =====================================================================
# TAB 1: PRZEGLĄD
# =====================================================================
with tab_overview:
    st.header("Przegląd farmy wiatrowej")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Turbiny", f"{farm.n_turbines}")
    col2.metric("Moc zainstalowana", f"{rated_total:.0f} MW")
    col3.metric("AEP", f"{aep:.1f} GWh")
    col4.metric("Capacity Factor", f"{cf:.1f}%")

    if is_floating:
        st.info(f"🌊 Turbina pływająca — Tp={wave_period}s, Hs={wave_height}m")

    col_layout, col_info = st.columns([2, 1])

    with col_layout:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(farm.layout_x, farm.layout_y, s=100, c="#1e5c3a",
                   edgecolors="white", linewidths=1.5, zorder=5)
        for i, (x, y) in enumerate(zip(farm.layout_x, farm.layout_y)):
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(8, 8), fontsize=8, color="#4a4a45")
        for x, y in zip(farm.layout_x, farm.layout_y):
            ax.add_patch(plt.Circle((x, y), turbine_info["diameter"] / 2,
                                    fill=False, color="#d8d5cc", linewidth=0.5, linestyle="--"))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title(f"Layout — {turbine_info['name']} | {layout_type} @ {spacing_D}D")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col_info:
        st.subheader("Parametry")
        params = {
            "Turbina": turbine_info["name"],
            "Średnica": f"{turbine_info['diameter']:.0f} m",
            "Hub height": f"{turbine_info['hub_height']:.0f} m",
            "Moc znamionowa": f"{turbine_info['rated_power']} MW",
            "Model wake": wake_model.upper(),
            "Layout": layout_type,
            "Rozstaw": f"{spacing_D}D = {spacing_D * turbine_info['diameter']:.0f} m",
            "Liczba turbin": farm.n_turbines,
        }
        if is_floating:
            params["Typ"] = "🌊 Pływająca"
            params["Fale"] = f"Tp={wave_period}s, Hs={wave_height}m"
        if turbine_info.get("custom_yaml"):
            params["Źródło"] = "⭐ Custom YAML"
        for k, v in params.items():
            st.write(f"**{k}:** {v}")


# =====================================================================
# TAB 2: DANE WIATROWE
# =====================================================================
with tab_wind:
    st.header("Dane wiatrowe")

    col_rose, col_stats = st.columns([2, 1])
    with col_rose:
        fig = loader.plot_wind_rose(title="Róża wiatrów")
        st.pyplot(fig)
        plt.close()
    with col_stats:
        stats = loader.summary()
        st.subheader("Statystyki")
        for k, v in [
            ("Rekordów", stats['n_records']),
            ("Prędkość średnia", f"{stats['wind_speed']['mean']:.1f} m/s"),
            ("Weibull k", f"{stats['wind_speed']['weibull_k_est']:.2f}"),
            ("Weibull A", f"{stats['wind_speed']['weibull_A_est']:.1f} m/s"),
            ("Kierunek dominujący", f"{stats['wind_direction']['dominant']:.0f}°"),
            ("TI średnia", f"{stats['turbulence_intensity']['mean']:.4f}"),
        ]:
            st.write(f"**{k}:** {v}")

    fig = loader.plot_time_series()
    st.pyplot(fig)
    plt.close()


# =====================================================================
# TAB 3: FLOW FIELD
# =====================================================================
with tab_flow:
    st.header("Wizualizacja pola przepływu")

    col1, col2, col3 = st.columns(3)
    ff_wd = col1.slider("Kierunek [°]", 0.0, 350.0, 240.0, 10.0, key="ff_wd")
    ff_ws = col2.slider("Prędkość [m/s]", 3.0, 20.0, 9.0, 0.5, key="ff_ws")
    ff_ti = col3.slider("TI", 0.02, 0.15, 0.06, 0.01, key="ff_ti")

    if st.button("🌊 Generuj flow field", key="gen_ff"):
        with st.spinner("Obliczam..."):
            fig = farm.plot_flow_field(
                wind_direction=ff_wd, wind_speed=ff_ws, ti=ff_ti, figsize=(14, 6),
            )
            st.session_state["fig_flow"] = fig_to_bytes(fig)
            farm.set_wind_data(wind_rose)

    show_stored_fig("fig_flow")

    st.divider()
    st.subheader("Waking directions")

    if st.button("Pokaż kierunki wake", key="gen_waking"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, wd in zip(axes, [ff_wd - 30, ff_wd, ff_wd + 30]):
            farm.fmodel.set(
                wind_directions=[wd % 360], wind_speeds=[ff_ws],
                turbulence_intensities=[ff_ti],
            )
            layoutviz.plot_turbine_points(farm.fmodel, ax=ax)
            layoutviz.plot_waking_directions(farm.fmodel, ax=ax, limit_dist_D=20)
            ax.set_title(f"WD={wd % 360:.0f}°")
            ax.set_aspect("equal")
        fig.suptitle("Które turbiny widzą ślad których?", fontsize=14)
        fig.tight_layout()
        st.session_state["fig_waking"] = fig_to_bytes(fig)
        farm.set_wind_data(wind_rose)

    show_stored_fig("fig_waking")


# =====================================================================
# TAB 4: PORÓWNANIA
# =====================================================================
with tab_compare:
    st.header("Porównania")

    compare_type = st.radio(
        "Co porównać?",
        ["Modele wake", "Turbiny", "Spacing sweep"],
        horizontal=True,
    )

    if compare_type == "Modele wake":
        models_to_compare = st.multiselect(
            "Wybierz modele", list(WAKE_MODELS.keys()),
            default=["jensen", "gch", "cc"],
            format_func=lambda x: x.upper(),
        )
        if st.button("Porównaj modele", key="cmp_wake"):
            with st.spinner("Obliczam..."):
                results = {}
                for m in models_to_compare:
                    farm.switch_wake_model(m)
                    farm.set_wind_data(wind_rose)
                    farm.run()
                    results[m.upper()] = {
                        "AEP [GWh]": farm.get_aep_gwh(),
                        "Wake losses [%]": farm.get_wake_losses_percent(),
                    }
                farm.switch_wake_model(wake_model)
                farm.set_wind_data(wind_rose)

                df = pd.DataFrame(results).T
                st.session_state["df_wake_cmp"] = df

                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                colors = ["#534AB7", "#1e5c3a", "#c8531a", "#7a3b00", "#3B8BD4"]
                axes[0].bar(df.index, df["AEP [GWh]"], color=colors[:len(df)])
                axes[0].set_ylabel("AEP [GWh]")
                axes[0].set_title("AEP per model")
                axes[1].bar(df.index, df["Wake losses [%]"], color=colors[:len(df)])
                axes[1].set_ylabel("Wake losses [%]")
                axes[1].set_title("Straty wake")
                fig.tight_layout()
                st.session_state["fig_wake_cmp"] = fig_to_bytes(fig)

        show_stored_df("df_wake_cmp")
        show_stored_fig("fig_wake_cmp")

    elif compare_type == "Turbiny":
        # Filtruj non-floating turbiny do porównania (floating wymaga rebuild)
        comparable = [k for k in TURBINE_LIBRARY.keys()
                      if k not in FLOATING_TURBINES and not TURBINE_LIBRARY[k].get("floating")]
        turbines_to_compare = st.multiselect(
            "Wybierz turbiny", comparable,
            default=[k for k in ["nrel_5MW", "iea_10MW", "iea_15MW", "iea_22MW"] if k in comparable],
            format_func=lambda x: TURBINE_LIBRARY[x]["name"],
        )
        if st.button("Porównaj turbiny", key="cmp_turb"):
            with st.spinner("Obliczam..."):
                results = farm.compare_turbines(
                    turbines=turbines_to_compare,
                    spacing_D=spacing_D, n_rows=n_rows, n_cols=n_cols,
                )
                farm.switch_turbine(turbine_name)
                if layout_type == "grid":
                    farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)
                elif layout_type == "staggered":
                    farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D, offset=stagger_offset)
                elif layout_type == "parallelogram":
                    farm.set_layout_parallelogram(n_turbines=para_n, r1_D=para_r1, r2_D=para_r2, theta1_deg=float(para_t1), theta2_deg=float(para_t2))
                farm.set_wind_data(wind_rose)

                rows = [v for v in results.values() if "error" not in v]
                st.session_state["df_turb_cmp"] = pd.DataFrame(rows)

        show_stored_df("df_turb_cmp")

    elif compare_type == "Spacing sweep":
        sp_min, sp_max = st.slider("Zakres [×D]", 3.0, 15.0, (5.0, 12.0), 0.5)
        sp_step = st.selectbox("Krok", [0.5, 1.0, 2.0], index=1)

        if st.button("Skanuj rozstawy", key="cmp_spacing"):
            with st.spinner("Obliczam..."):
                sweep = farm.spacing_sweep(
                    spacings_D=np.arange(sp_min, sp_max + sp_step, sp_step),
                    n_rows=n_rows, n_cols=n_cols,
                )

                fig, ax1 = plt.subplots(figsize=(10, 6))
                ax1.plot(sweep["spacings_D"], sweep["aep_gwh"], "o-",
                         color="#1e5c3a", linewidth=2, label="AEP")
                ax1.set_xlabel("Rozstaw [×D]")
                ax1.set_ylabel("AEP [GWh]", color="#1e5c3a")
                ax2 = ax1.twinx()
                ax2.plot(sweep["spacings_D"], sweep["wake_loss_pct"], "s--",
                         color="#c8531a", linewidth=2, label="Wake losses")
                ax2.set_ylabel("Wake losses [%]", color="#c8531a")
                fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
                ax1.grid(True, alpha=0.3)
                fig.suptitle(f"Spacing sweep — {turbine_info['name']} | {wake_model.upper()}")
                st.session_state["fig_spacing"] = fig_to_bytes(fig)

                best = np.argmax(sweep["aep_gwh"])
                st.session_state["txt_spacing"] = (
                    f"Optymalny rozstaw: **{sweep['spacings_D'][best]:.1f}D** "
                    f"(AEP={sweep['aep_gwh'][best]:.1f} GWh, "
                    f"wake losses={sweep['wake_loss_pct'][best]:.1f}%)"
                )

                # Przywróć
                if layout_type == "grid":
                    farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)
                elif layout_type == "staggered":
                    farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D, offset=stagger_offset)
                elif layout_type == "parallelogram":
                    farm.set_layout_parallelogram(n_turbines=para_n, r1_D=para_r1, r2_D=para_r2, theta1_deg=float(para_t1), theta2_deg=float(para_t2))
                farm.set_wind_data(wind_rose)

        show_stored_fig("fig_spacing")
        if "txt_spacing" in st.session_state:
            st.success(st.session_state["txt_spacing"])


# =====================================================================
# TAB 5: BENCHMARK — ALL-IN-ONE
# =====================================================================
with tab_benchmark:
    st.header("🏆 Benchmark — porównanie konfiguracji")
    st.caption(
        "Porównanie wielu konfiguracji naraz: modele wake × layouty × spacing. "
        "Parametry wiatru ustawiasz tutaj (niezależnie od sidebara)."
    )

    # --- Kontrolki na górze ---
    col_bw1, col_bw2, col_bw3, col_bw4 = st.columns(4)
    bm_weibull_A = col_bw1.slider("Weibull A [m/s]", 6.0, 14.0, 9.5, 0.5, key="bm_wa")
    bm_weibull_k = col_bw2.slider("Weibull k", 1.5, 3.0, 2.1, 0.1, key="bm_wk")
    bm_n_turb = col_bw3.number_input("Turbin (siatka)", 4, 100, 25, key="bm_nt")
    bm_turbine = col_bw4.selectbox(
        "Turbina",
        [k for k in TURBINE_LIBRARY.keys() if k not in FLOATING_TURBINES],
        index=2,
        format_func=lambda x: TURBINE_LIBRARY[x]["name"],
        key="bm_turb",
    )

    st.divider()

    col_bm_left, col_bm_right = st.columns(2)

    with col_bm_left:
        bm_models = st.multiselect(
            "Modele wake",
            list(WAKE_MODELS.keys()),
            default=["jensen", "gch", "cc"],
            format_func=lambda x: x.upper(),
            key="bm_models",
        )
    with col_bm_right:
        bm_layouts = st.multiselect(
            "Layouty",
            ["grid", "staggered", "parallelogram"],
            default=["grid", "staggered"],
            format_func=lambda x: {"grid": "Siatka", "staggered": "Przesunięta", "parallelogram": "Równoległobok"}[x],
            key="bm_layouts",
        )

    bm_spacings = st.slider("Zakres spacing [×D]", 4.0, 12.0, (5.0, 9.0), 1.0, key="bm_sp")

    if st.button("🏆 Uruchom benchmark", key="run_benchmark", type="primary"):
        with st.spinner("Benchmark w toku..."):
            # Wygeneruj dane wiatrowe dla benchmarku
            bm_config = BalticWindConfig(weibull_A=bm_weibull_A, weibull_k=bm_weibull_k)
            bm_loader = WindDataLoader()
            bm_loader.generate_mock_data(config=bm_config, seed=42)
            bm_wr = bm_loader.to_wind_rose(wd_step=30.0, ws_step=3.0)

            D_bm = TURBINE_LIBRARY[bm_turbine]["diameter"]
            n_bm = bm_n_turb
            n_c = int(np.ceil(np.sqrt(n_bm)))
            n_r = int(np.ceil(n_bm / n_c))

            spacings_list = np.arange(bm_spacings[0], bm_spacings[1] + 1.0, 1.0)

            results = []
            total_runs = len(bm_models) * len(bm_layouts) * len(spacings_list)
            progress = st.progress(0)
            run_i = 0

            for wake_m in bm_models:
                for layout_t in bm_layouts:
                    for sp in spacings_list:
                        try:
                            f = FarmModel(wake_model=wake_m, turbine=bm_turbine)

                            if layout_t == "grid":
                                f.set_layout_grid(n_rows=n_r, n_cols=n_c, spacing_D=sp)
                            elif layout_t == "staggered":
                                f.set_layout_staggered(n_rows=n_r, n_cols=n_c, spacing_D=sp, offset=0.5)
                            elif layout_t == "parallelogram":
                                f.set_layout_parallelogram(n_turbines=n_bm, r1_D=sp, r2_D=sp,
                                                           theta1_deg=88.0, theta2_deg=18.0)

                            f.set_wind_data(bm_wr)
                            f.run()
                            a = f.get_aep_gwh()
                            wl = f.get_wake_losses_percent()

                            rated = TURBINE_LIBRARY[bm_turbine]["rated_power"] * f.n_turbines
                            cf_val = a / (rated * 8.76) * 100 if rated > 0 else 0

                            results.append({
                                "Wake model": wake_m.upper(),
                                "Layout": layout_t,
                                "Spacing [×D]": sp,
                                "Turbin": f.n_turbines,
                                "AEP [GWh]": round(a, 1),
                                "Wake losses [%]": round(wl, 1),
                                "CF [%]": round(cf_val, 1),
                            })
                        except Exception as e:
                            results.append({
                                "Wake model": wake_m.upper(),
                                "Layout": layout_t,
                                "Spacing [×D]": sp,
                                "Turbin": "?",
                                "AEP [GWh]": f"ERR: {str(e)[:30]}",
                                "Wake losses [%]": "-",
                                "CF [%]": "-",
                            })

                        run_i += 1
                        progress.progress(run_i / total_runs)

            progress.empty()
            bm_df = pd.DataFrame(results)
            st.session_state["bm_results"] = bm_df

            # Wykres
            try:
                numeric_df = bm_df[bm_df["AEP [GWh]"].apply(lambda x: isinstance(x, (int, float)))].copy()
                if len(numeric_df) > 0:
                    fig_bm, axes_bm = plt.subplots(1, 2, figsize=(14, 6))

                    # AEP per konfiguracja
                    labels = numeric_df.apply(
                        lambda r: f"{r['Wake model']}\n{r['Layout']}\n{r['Spacing [×D]']}D", axis=1
                    )
                    colors_bm = []
                    cmap = plt.cm.Set2(np.linspace(0, 1, len(bm_models)))
                    model_color = {m.upper(): cmap[i] for i, m in enumerate(bm_models)}
                    for _, r in numeric_df.iterrows():
                        colors_bm.append(model_color.get(r["Wake model"], "#888888"))

                    axes_bm[0].barh(range(len(numeric_df)), numeric_df["AEP [GWh]"],
                                    color=colors_bm, edgecolor="white", linewidth=0.5)
                    axes_bm[0].set_yticks(range(len(numeric_df)))
                    axes_bm[0].set_yticklabels(labels, fontsize=7)
                    axes_bm[0].set_xlabel("AEP [GWh]")
                    axes_bm[0].set_title("AEP per konfiguracja")
                    axes_bm[0].grid(True, alpha=0.3, axis="x")

                    # Wake losses
                    wl_vals = numeric_df["Wake losses [%]"].astype(float)
                    axes_bm[1].barh(range(len(numeric_df)), wl_vals,
                                    color=colors_bm, edgecolor="white", linewidth=0.5)
                    axes_bm[1].set_yticks(range(len(numeric_df)))
                    axes_bm[1].set_yticklabels(labels, fontsize=7)
                    axes_bm[1].set_xlabel("Wake losses [%]")
                    axes_bm[1].set_title("Straty wake")
                    axes_bm[1].grid(True, alpha=0.3, axis="x")

                    fig_bm.suptitle(
                        f"Benchmark — {TURBINE_LIBRARY[bm_turbine]['name']} | "
                        f"Weibull A={bm_weibull_A} k={bm_weibull_k}",
                        fontsize=13,
                    )
                    fig_bm.tight_layout()
                    st.session_state["fig_benchmark"] = fig_to_bytes(fig_bm)
            except Exception:
                pass

    # Wyświetl wyniki
    if "bm_results" in st.session_state:
        bm_df = st.session_state["bm_results"]

        # Najlepsza konfiguracja
        try:
            numeric_df = bm_df[bm_df["AEP [GWh]"].apply(lambda x: isinstance(x, (int, float)))].copy()
            if len(numeric_df) > 0:
                best_idx = numeric_df["AEP [GWh]"].astype(float).idxmax()
                best = numeric_df.loc[best_idx]
                st.success(
                    f"🏆 **Najlepsza konfiguracja:** {best['Wake model']} | "
                    f"{best['Layout']} @ {best['Spacing [×D]']}D → "
                    f"**AEP = {best['AEP [GWh]']} GWh** | "
                    f"Wake losses = {best['Wake losses [%]']}%"
                )
        except Exception:
            pass

        st.dataframe(bm_df, use_container_width=True, hide_index=True)
        show_stored_fig("fig_benchmark")

        st.download_button(
            "⬇️ Pobierz benchmark CSV",
            bm_df.to_csv(index=False),
            file_name="benchmark_results.csv", mime="text/csv",
        )

    # --- Wizualizacja flow field obok siebie ---
    st.divider()
    st.subheader("🌊 Porównanie wake — flow field")
    st.caption(
        "Wizualizacja pola przepływu dla wybranych konfiguracji obok siebie. "
        "Używa tych samych parametrów wiatru co benchmark powyżej."
    )

    col_ff1, col_ff2, col_ff3 = st.columns(3)
    ff_bm_wd = col_ff1.slider("Kierunek wiatru [°]", 0.0, 350.0, 240.0, 10.0, key="ff_bm_wd")
    ff_bm_ws = col_ff2.slider("Prędkość [m/s]", 3.0, 20.0, 9.0, 0.5, key="ff_bm_ws")
    ff_bm_sp = col_ff3.slider("Spacing [×D]", 4.0, 12.0, 7.0, 0.5, key="ff_bm_sp")

    col_ff4, col_ff5 = st.columns(2)
    ff_bm_models = col_ff4.multiselect(
        "Modele wake do porównania",
        list(WAKE_MODELS.keys()),
        default=["jensen", "gch", "turbopark", "cc"],
        format_func=lambda x: x.upper(),
        key="ff_bm_models",
    )
    ff_bm_layout = col_ff5.selectbox(
        "Layout",
        ["grid", "staggered", "parallelogram"],
        format_func=lambda x: {"grid": "Siatka", "staggered": "Przesunięta", "parallelogram": "Równoległobok"}[x],
        key="ff_bm_layout",
    )

    if st.button("🌊 Generuj porównanie flow field", key="gen_ff_benchmark"):
        if len(ff_bm_models) < 1:
            st.warning("Wybierz co najmniej 1 model wake.")
        else:
            with st.spinner(f"Generuję flow field dla {len(ff_bm_models)} modeli..."):
                n_models = len(ff_bm_models)
                # Max 4 kolumny, potem nowe rzędy
                n_cols_fig = min(n_models, 4)
                n_rows_fig = int(np.ceil(n_models / n_cols_fig))

                fig_ff, axes_ff = plt.subplots(
                    n_rows_fig, n_cols_fig,
                    figsize=(5 * n_cols_fig, 5 * n_rows_fig),
                    squeeze=False,
                )

                # Turbina i layout
                bm_turb_sel = st.session_state.get("bm_turb", turbine_name)
                bm_nt_sel = st.session_state.get("bm_nt", 25)
                D_ff = TURBINE_LIBRARY.get(bm_turb_sel, turbine_info)["diameter"]
                n_c_ff = int(np.ceil(np.sqrt(bm_nt_sel)))
                n_r_ff = int(np.ceil(bm_nt_sel / n_c_ff))

                for idx, wm in enumerate(ff_bm_models):
                    row_i = idx // n_cols_fig
                    col_i = idx % n_cols_fig
                    ax = axes_ff[row_i][col_i]

                    try:
                        f = FarmModel(wake_model=wm, turbine=bm_turb_sel)
                        if ff_bm_layout == "grid":
                            f.set_layout_grid(n_rows=n_r_ff, n_cols=n_c_ff, spacing_D=ff_bm_sp)
                        elif ff_bm_layout == "staggered":
                            f.set_layout_staggered(n_rows=n_r_ff, n_cols=n_c_ff, spacing_D=ff_bm_sp, offset=0.5)
                        elif ff_bm_layout == "parallelogram":
                            f.set_layout_parallelogram(
                                n_turbines=bm_nt_sel, r1_D=ff_bm_sp, r2_D=ff_bm_sp,
                                theta1_deg=88.0, theta2_deg=18.0,
                            )

                        f.plot_flow_field(
                            wind_direction=ff_bm_wd,
                            wind_speed=ff_bm_ws,
                            ti=0.06,
                            ax=ax,
                            title=f"{wm.upper()}",
                            show_labels=False,
                            show_wind_arrow=(idx == 0),  # strzałka tylko na pierwszym
                        )

                    except Exception as e:
                        ax.text(0.5, 0.5, f"Błąd:\n{str(e)[:60]}",
                                transform=ax.transAxes, ha="center", va="center",
                                fontsize=9, wrap=True)
                        ax.set_title(f"{wm.upper()} — BŁĄD")

                # Ukryj puste subploty
                for idx in range(n_models, n_rows_fig * n_cols_fig):
                    row_i = idx // n_cols_fig
                    col_i = idx % n_cols_fig
                    axes_ff[row_i][col_i].set_visible(False)

                fig_ff.suptitle(
                    f"Porównanie wake — {ff_bm_layout} @ {ff_bm_sp}D | "
                    f"WD={ff_bm_wd}° WS={ff_bm_ws} m/s | "
                    f"{TURBINE_LIBRARY.get(bm_turb_sel, turbine_info)['name']}",
                    fontsize=14, fontweight="500",
                )
                fig_ff.tight_layout()
                st.session_state["fig_ff_benchmark"] = fig_to_bytes(fig_ff)

    show_stored_fig("fig_ff_benchmark")

    # --- Porównanie layoutów przy stałym modelu wake ---
    st.divider()
    st.subheader("📐 Porównanie layoutów — flow field")

    col_fl1, col_fl2 = st.columns(2)
    fl_wake = col_fl1.selectbox(
        "Model wake",
        list(WAKE_MODELS.keys()),
        index=1,
        format_func=lambda x: x.upper(),
        key="fl_wake",
    )
    fl_layouts = col_fl2.multiselect(
        "Layouty do porównania",
        ["grid", "staggered", "parallelogram"],
        default=["grid", "staggered", "parallelogram"],
        format_func=lambda x: {"grid": "Siatka", "staggered": "Przesunięta", "parallelogram": "Równoległobok"}[x],
        key="fl_layouts",
    )

    if st.button("📐 Generuj porównanie layoutów", key="gen_fl_benchmark"):
        if len(fl_layouts) < 1:
            st.warning("Wybierz co najmniej 1 layout.")
        else:
            with st.spinner(f"Generuję flow field dla {len(fl_layouts)} layoutów..."):
                n_lay = len(fl_layouts)
                fig_fl, axes_fl = plt.subplots(1, n_lay, figsize=(5 * n_lay, 5), squeeze=False)

                bm_turb_sel = st.session_state.get("bm_turb", turbine_name)
                bm_nt_sel = st.session_state.get("bm_nt", 25)
                n_c_fl = int(np.ceil(np.sqrt(bm_nt_sel)))
                n_r_fl = int(np.ceil(bm_nt_sel / n_c_fl))

                for idx, lay in enumerate(fl_layouts):
                    ax = axes_fl[0][idx]
                    try:
                        f = FarmModel(wake_model=fl_wake, turbine=bm_turb_sel)
                        if lay == "grid":
                            f.set_layout_grid(n_rows=n_r_fl, n_cols=n_c_fl, spacing_D=ff_bm_sp)
                        elif lay == "staggered":
                            f.set_layout_staggered(n_rows=n_r_fl, n_cols=n_c_fl, spacing_D=ff_bm_sp, offset=0.5)
                        elif lay == "parallelogram":
                            f.set_layout_parallelogram(
                                n_turbines=bm_nt_sel, r1_D=ff_bm_sp, r2_D=ff_bm_sp,
                                theta1_deg=88.0, theta2_deg=18.0,
                            )

                        lay_label = {"grid": "Siatka", "staggered": "Przesunięta", "parallelogram": "Równoległobok"}[lay]
                        f.plot_flow_field(
                            wind_direction=ff_bm_wd,
                            wind_speed=ff_bm_ws,
                            ti=0.06,
                            ax=ax,
                            title=f"{lay_label}",
                            show_labels=False,
                            show_wind_arrow=(idx == 0),
                        )
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Błąd:\n{str(e)[:60]}",
                                transform=ax.transAxes, ha="center", va="center",
                                fontsize=9, wrap=True)

                fig_fl.suptitle(
                    f"Porównanie layoutów — {fl_wake.upper()} @ {ff_bm_sp}D | "
                    f"WD={ff_bm_wd}° WS={ff_bm_ws} m/s",
                    fontsize=14, fontweight="500",
                )
                fig_fl.tight_layout()
                st.session_state["fig_fl_benchmark"] = fig_to_bytes(fig_fl)

    show_stored_fig("fig_fl_benchmark")


# =====================================================================
# TAB 6: OPTYMALIZACJA (layout) — Yaw przeniesiony do zakładki Grupa 3
# =====================================================================
with tab_optimize:
    st.header("Optymalizacja layoutu")
    st.caption(
        "Optymalizacja FLORIS dla pełnej róży wiatrów. "
        "Do porównywania algorytmów na wąskim binie użyj zakładki **🧪 Lab algorytmów**. "
        "Optymalizacja yaw (wake steering) jest w zakładce **🤝 Grupa 3**."
    )

    opt_method = st.radio(
        "Metoda",
        ["Scipy (gradient)", "Random Search (FLORIS)"],
        horizontal=True,
    )

    if opt_method == "Scipy (gradient)":
        opt_maxiter = st.slider("Max iteracji", 10, 100, 30, 10)
        opt_margin = st.slider("Margines granic [×D]", 1.0, 5.0, 3.0, 0.5)

        if st.button("🎯 Optymalizuj layout", key="opt_scipy"):
            with st.spinner(f"Optymalizacja Scipy ({opt_maxiter} iteracji)..."):
                wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                farm.set_wind_data(wr_coarse)

                opt = Optimizer(farm)
                opt.set_boundaries_from_layout(margin_D=opt_margin)
                opt.set_min_distance(min_dist_D=3.0)

                result = opt.optimize_layout_scipy(maxiter=opt_maxiter)

                st.session_state["opt_result"] = {
                    "before": result.initial_aep_gwh,
                    "after": result.optimized_aep_gwh,
                    "pct": result.aep_improvement_pct,
                    "time": result.elapsed_seconds,
                }

                fig = opt.plot_optimization_result(result)
                st.session_state["fig_opt"] = fig_to_bytes(fig)

                opt.apply_result(result)
                farm.set_wind_data(wind_rose)
                farm.run()
                st.session_state["opt_final_aep"] = farm.get_aep_gwh()

        if "opt_result" in st.session_state:
            r = st.session_state["opt_result"]
            col1, col2, col3 = st.columns(3)
            col1.metric("AEP przed", f"{r['before']:.1f} GWh")
            col2.metric("AEP po", f"{r['after']:.1f} GWh", f"+{r['pct']:.2f}%")
            col3.metric("Czas", f"{r['time']:.0f}s")
        show_stored_fig("fig_opt")
        if "opt_final_aep" in st.session_state:
            st.info(f"AEP na dokładnej WindRose: **{st.session_state['opt_final_aep']:.1f} GWh**")

    else:  # Random Search (FLORIS)
        rs_seconds = st.slider("Budżet czasowy [s]", 10, 300, 60, 10)
        rs_margin = st.slider("Margines granic [×D]", 1.0, 5.0, 3.0, 0.5, key="rs_margin")

        if st.button("🎯 Optymalizuj layout (RS)", key="opt_rs"):
            with st.spinner(f"Random Search ({rs_seconds}s)..."):
                wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                farm.set_wind_data(wr_coarse)

                opt = Optimizer(farm)
                opt.set_boundaries_from_layout(margin_D=rs_margin)
                opt.set_min_distance(min_dist_D=3.0)

                try:
                    result = opt.optimize_layout_random_search(seconds=rs_seconds)
                    st.session_state["opt_rs_result"] = {
                        "before": result.initial_aep_gwh,
                        "after": result.optimized_aep_gwh,
                        "pct": result.aep_improvement_pct,
                        "time": result.elapsed_seconds,
                    }
                    fig = opt.plot_optimization_result(result)
                    st.session_state["fig_opt_rs"] = fig_to_bytes(fig)
                    opt.apply_result(result)
                    farm.set_wind_data(wind_rose)
                    farm.run()
                except Exception as e:
                    st.session_state["opt_rs_error"] = str(e)

        if "opt_rs_result" in st.session_state:
            r = st.session_state["opt_rs_result"]
            col1, col2, col3 = st.columns(3)
            col1.metric("AEP przed", f"{r['before']:.1f} GWh")
            col2.metric("AEP po", f"{r['after']:.1f} GWh", f"+{r['pct']:.2f}%")
            col3.metric("Czas", f"{r['time']:.0f}s")
        show_stored_fig("fig_opt_rs")
        if "opt_rs_error" in st.session_state:
            st.error(st.session_state["opt_rs_error"])


# =====================================================================
# TAB: LAB ALGORYTMÓW — porównanie algorytmów optymalizacji layoutu
# =====================================================================
with tab_lab:
    st.header("🧪 Lab algorytmów")
    st.caption(
        "Porównaj wiele algorytmów optymalizacji layoutu na **tej samej farmie** "
        "i **tym samym wąskim binie** wiatrowym (1 kierunek + 1 prędkość = "
        "szybka ewaluacja, dziesiątki wywołań w kilka sekund). "
        "Algorytmy startują z siatki regularnej i próbują znaleźć lepszy layout. "
        "Każdy algorytm to osobny plik w `src/algorithms/`."
    )

    # --- Konfiguracja warunków ---
    st.subheader("1. Warunki testowe (wąski bin)")
    col_wd, col_ws, col_ti = st.columns(3)
    lab_wd = col_wd.slider("Kierunek WD [°]", 0.0, 350.0, 270.0, 10.0, key="lab_wd")
    lab_ws = col_ws.slider("Prędkość WS [m/s]", 4.0, 18.0, 9.0, 0.5, key="lab_ws")
    lab_ti = col_ti.slider("TI", 0.02, 0.15, 0.06, 0.01, key="lab_ti")

    st.subheader("2. Farma startowa")
    col_r, col_c, col_sp = st.columns(3)
    lab_n_rows = int(col_r.number_input("Rzędy", 2, 6, 3, 1, key="lab_nr"))
    lab_n_cols = int(col_c.number_input("Kolumny", 2, 6, 3, 1, key="lab_nc"))
    lab_spacing_D = col_sp.slider("Spacing startowy [×D]", 4.0, 12.0, 7.0, 0.5, key="lab_sp")

    st.subheader("3. Ograniczenia + budżet")
    col_m, col_d, col_b, col_s = st.columns(4)
    lab_margin_D = col_m.slider("Margines boundaries [×D]", 1.0, 5.0, 3.0, 0.5, key="lab_margin")
    lab_min_dist_D = col_d.slider("Min odległość [×D]", 2.0, 5.0, 3.0, 0.5, key="lab_mindist")
    lab_eval_budget = int(col_b.number_input("Eval budget per algo", 20, 1000, 100, 10, key="lab_eb"))
    lab_seed = int(col_s.number_input("Seed", 1, 9999, 42, 1, key="lab_seed"))

    st.subheader("4. Algorytmy do porównania")
    lab_selected = st.multiselect(
        "Wybierz algorytmy (każdy w osobnym pliku src/algorithms/)",
        list(ALGORITHMS.keys()),
        default=["scipy", "random_search", "genetic", "simulated_annealing"],
        format_func=lambda k: ALGORITHMS[k].name,
        key="lab_sel",
    )

    if st.button("🚀 Uruchom porównanie", key="lab_run", type="primary",
                 disabled=len(lab_selected) == 0):
        progress = st.progress(0)
        status = st.empty()
        results = []

        for idx, algo_key in enumerate(lab_selected):
            algo_cls = ALGORITHMS[algo_key]
            algo = algo_cls()
            status.write(f"⏳ {idx + 1}/{len(lab_selected)}: {algo.name}...")

            try:
                # Świeża farma per algorytm — czyste startowe warunki
                lab_farm = FarmModel(
                    wake_model=wake_model, turbine=turbine_name,
                    wave_period=wave_period, wave_height=wave_height,
                )
                lab_farm.set_layout_grid(
                    n_rows=lab_n_rows, n_cols=lab_n_cols, spacing_D=lab_spacing_D,
                )

                ts = TimeSeries(
                    wind_directions=np.array([lab_wd]),
                    wind_speeds=np.array([lab_ws]),
                    turbulence_intensities=np.array([lab_ti]),
                )
                lab_farm.set_wind_data(ts)

                bounds = bounds_from_layout(
                    lab_farm.layout_x, lab_farm.layout_y, lab_margin_D * lab_farm.D,
                )
                min_dist = lab_min_dist_D * lab_farm.D

                res = algo.run(
                    lab_farm, bounds, min_dist, lab_eval_budget, seed=lab_seed,
                )
                results.append(res)
            except Exception as e:
                from src.algorithms import AlgorithmResult
                results.append(AlgorithmResult(
                    name=algo.name,
                    initial_x=np.array([]), initial_y=np.array([]), initial_aep=0.0,
                    final_x=np.array([]), final_y=np.array([]), final_aep=0.0,
                    elapsed_s=0.0, n_evaluations=0, error=str(e),
                ))

            progress.progress((idx + 1) / len(lab_selected))

        status.empty()
        progress.empty()
        st.session_state["lab_results"] = results

    # --- Wyniki ---
    if "lab_results" in st.session_state:
        results = st.session_state["lab_results"]

        # Tabela porównawcza
        table_data = []
        for r in results:
            table_data.append({
                "Algorytm": r.name,
                "AEP init [GWh]": round(r.initial_aep, 3),
                "AEP final [GWh]": round(r.final_aep, 3),
                "Δ [GWh]": round(r.improvement_gwh, 4),
                "Δ [%]": round(r.improvement_pct, 3),
                "Czas [s]": round(r.elapsed_s, 2),
                "Ewaluacje": r.n_evaluations,
                "Błąd": (r.error or "")[:60],
            })
        df_lab = pd.DataFrame(table_data)
        st.subheader("📊 Tabela porównawcza")
        st.dataframe(df_lab, hide_index=True, use_container_width=True)

        valid = [r for r in results if r.error is None and r.final_aep > 0]
        if valid:
            best = max(valid, key=lambda r: r.final_aep)
            fastest = min(valid, key=lambda r: r.elapsed_s)
            col_b1, col_b2 = st.columns(2)
            col_b1.success(
                f"🏆 **Najlepszy AEP:** {best.name} → "
                f"{best.final_aep:.3f} GWh (+{best.improvement_pct:.2f}%)"
            )
            col_b2.info(f"⚡ **Najszybszy:** {fastest.name} → {fastest.elapsed_s:.1f}s")

        # Wykres zbieżności
        st.subheader("📈 Zbieżność algorytmów")
        valid_hist = [r for r in results if len(r.history) > 1]
        if valid_hist:
            fig_conv, ax_conv = plt.subplots(figsize=(10, 5))
            cmap = plt.cm.tab10(np.linspace(0, 1, max(len(valid_hist), 1)))
            for r, color in zip(valid_hist, cmap):
                ax_conv.plot(r.history, label=r.name, color=color, linewidth=2, alpha=0.85)
            ax_conv.set_xlabel("Iteracja / krok")
            ax_conv.set_ylabel("Best AEP [GWh]")
            ax_conv.set_title("Krzywe zbieżności")
            ax_conv.legend(loc="lower right", fontsize=9)
            ax_conv.grid(True, alpha=0.3)
            st.pyplot(fig_conv)
            plt.close()

        # Wykres słupkowy improvement
        st.subheader("📊 Poprawa AEP")
        fig_imp, (ax_imp, ax_time) = plt.subplots(1, 2, figsize=(13, 5))
        names = [r.name for r in results]
        imps = [r.improvement_pct for r in results]
        times = [r.elapsed_s for r in results]
        colors = ["#1e5c3a" if i >= 0 else "#c8531a" for i in imps]

        ax_imp.barh(names, imps, color=colors)
        ax_imp.set_xlabel("Improvement [%]")
        ax_imp.set_title("Poprawa AEP względem startu")
        for i, v in enumerate(imps):
            ax_imp.text(v, i, f" {v:+.2f}%", va="center", fontsize=9)
        ax_imp.axvline(0, color="black", linewidth=0.5)
        ax_imp.grid(True, alpha=0.3, axis="x")

        ax_time.barh(names, times, color="#534AB7")
        ax_time.set_xlabel("Czas [s]")
        ax_time.set_title("Czas wykonania")
        for i, v in enumerate(times):
            ax_time.text(v, i, f" {v:.1f}s", va="center", fontsize=9)
        ax_time.grid(True, alpha=0.3, axis="x")

        fig_imp.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

        # Layout porównanie
        st.subheader("📍 Layouty — przed vs po (per algorytm)")
        n_show = len([r for r in results if len(r.final_x) > 0])
        if n_show > 0:
            n_cols_fig = min(n_show, 3)
            n_rows_fig = int(np.ceil(n_show / n_cols_fig))
            fig_lay, axes_lay = plt.subplots(
                n_rows_fig, n_cols_fig,
                figsize=(5 * n_cols_fig, 5 * n_rows_fig),
                squeeze=False,
            )
            shown = 0
            for r in results:
                if len(r.final_x) == 0:
                    continue
                ax = axes_lay[shown // n_cols_fig][shown % n_cols_fig]
                ax.scatter(r.initial_x, r.initial_y, s=60, c="#888888",
                           alpha=0.5, label="Start", marker="x")
                ax.scatter(r.final_x, r.final_y, s=80, c="#1e5c3a",
                           label="Final", edgecolors="white", linewidths=1)
                # strzałki
                for sx, sy, fx, fy in zip(r.initial_x, r.initial_y, r.final_x, r.final_y):
                    if abs(fx - sx) > 1 or abs(fy - sy) > 1:
                        ax.annotate(
                            "", xy=(fx, fy), xytext=(sx, sy),
                            arrowprops=dict(arrowstyle="->", color="#c8531a", alpha=0.4, lw=1),
                        )
                ax.set_title(f"{r.name}\n+{r.improvement_pct:.2f}% in {r.elapsed_s:.1f}s")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="upper right")
                shown += 1
            for idx in range(shown, n_rows_fig * n_cols_fig):
                axes_lay[idx // n_cols_fig][idx % n_cols_fig].set_visible(False)
            fig_lay.tight_layout()
            st.pyplot(fig_lay)
            plt.close()

        # Eksport
        st.divider()
        st.download_button(
            "⬇️ Eksport CSV (porównanie)",
            df_lab.to_csv(index=False),
            file_name="lab_algorithms.csv", mime="text/csv",
        )

        # Eksport layoutów per algo
        for r in results:
            if len(r.final_x) == 0:
                continue
            algo_layout_df = pd.DataFrame({
                "turbine_id": range(len(r.final_x)),
                "x_m": r.final_x,
                "y_m": r.final_y,
            })
            st.download_button(
                f"⬇️ Layout final — {r.name}",
                algo_layout_df.to_csv(index=False),
                file_name=f"layout_{r.name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv",
                mime="text/csv",
                key=f"dl_{r.name}",
            )

    # --- Dokumentacja jak dodać własny algorytm ---
    with st.expander("💡 Jak dodać własny algorytm?"):
        st.markdown("""
1. Stwórz nowy plik w `src/algorithms/moj_algo.py`
2. Zdefiniuj klasę dziedziczącą po `LayoutAlgorithm`:
```python
from .base import LayoutAlgorithm, AlgorithmResult, evaluate_aep

class MojAlgorytm(LayoutAlgorithm):
    name = "Mój Algorytm"
    description = "Opis"

    def run(self, farm, bounds_rect, min_dist, eval_budget, seed=42, **params):
        # ... twoja logika ...
        return AlgorithmResult(...)
```
3. Dodaj do `src/algorithms/__init__.py`:
```python
from .moj_algo import MojAlgorytm
ALGORITHMS["moj"] = MojAlgorytm
```
4. Algorytm pojawi się tutaj automatycznie.
        """)


# =====================================================================
# TAB: GRUPA 3 — WAKE STEERING (Yaw + Curtailment + Helix)
# =====================================================================
with tab_group3:
    st.header("🤝 Grupa 3 — Wake Steering")
    st.caption(
        "Temat 3 (sterowanie aerodynamiczne farmy): yaw, hamowanie/derating "
        "upstream, Active Wake Mixing. Wszystko bazuje na layoucie z naszej "
        "Optymalizacji (Temat 2). Tu są zarówno **gotowe analizy** dla Grupy 3, "
        "jak i **opis co jeszcze powinni dorobić**."
    )

    g3_section = st.radio(
        "Sekcja",
        ["📋 Co robi Grupa 3", "🎯 Yaw (wake steering)", "🛑 Curtailment (baza)",
         "🌀 Active Wake Mixing (Helix)", "📊 Porównanie strategii", "📁 Eksport"],
        horizontal=False,
    )

    # ----- A. Opis zadań Grupy 3 -----
    if g3_section == "📋 Co robi Grupa 3":
        st.subheader("Zakres tematu 3 — sterowanie aerodynamiczne farmy")
        st.markdown("""
**Wejście (od Tematu 2):** `layout.csv` z naszej zakładki **📁 Eksport**.

**Wyjście Tematu 3:** strategie sterowania farmą redukujące straty wake — eksport jako `yaw_schedule.csv`, `curtailment_schedule.csv`, `wake_steering_summary.csv`.

---

#### A. Wake steering — pogłębienie analizy yaw
Bazę macie gotową (`Optimizer.optimize_yaw()` w `src/optimizer.py`, sekcja "🎯 Yaw" poniżej). Co dorobić:

1. **Yaw schedule lookup table** — eksport tabeli yaw_kąt per (wd, ws, turbina_id). Sterownik turbiny musi to wczytać.
2. **Robust wake steering** — re-optymalizacja z niepewnością kierunku (σ_WD = 3°/5°/7°). Pokazuje jak "ostry" lub "miękki" jest optymalny kąt.
3. **Yaw_max sweep** (0/15/25/30°) — krzywa AEP(yaw_max), znajdźcie "sweet spot" akceptowany przez producenta turbiny.
4. **Per-single-bin diagnostyka** — yaw dla pojedynczego (WD, WS) + wizualizacja deflection wake.

#### B. Curtailment / hamowanie upstream
Baza w sekcji "🛑 Curtailment" poniżej. Co dorobić:

1. **Greedy curtailment** — algorytm: w każdej iteracji wybiera turbinę do zderatowania jeśli farma jako całość zyskuje (upstream produkuje < rated, downstream wychodzi z cienia).
2. **Per-direction strategy** — różne zestawy zderatowanych turbin dla różnych kierunków.
3. **Hybrid yaw + curtailment** — kombinacja obu strategii.

#### C. Active Wake Mixing (Helix)
Bazę macie w sekcji "🌀 Helix". Co dorobić:

1. Porównanie: brak / yaw / helix / yaw+helix dla różnych kierunków.
2. Analiza wrażliwości amplitudy modulacji helix.

#### D. Raport końcowy (Wasz output)
Tabela: brak / yaw / curtailment / helix / kombinacje — AEP, % zmiany AEP, czas obliczeń, ryzyko (mechaniczne obciążenia turbiny).

#### Pliki techniczne, którymi można manipulować
- `src/optimizer.py` → `optimize_yaw()` (SerialRefine) — można dodać `optimize_yaw_robust()`
- `farm.fmodel.set(disable_turbines=...)` — wyłączanie turbin
- `farm.fmodel.set_operation_model("simple-derating")` + `power_setpoints=...` — derating
- `enable_active_wake_mixing=True` w configu wake — helix
- Notebook `notebooks/05_advanced_floris.py` ma działające przykłady wszystkich powyższych
        """)

    # ----- B. Yaw (przeniesione z zakładki Optymalizacja) -----
    elif g3_section == "🎯 Yaw (wake steering)":
        st.subheader("Optymalizacja kątów yaw")
        st.caption(
            "FLORIS YawOptimizationSR (SerialRefine) — znajduje optymalne kąty yaw "
            "per kierunek wiatru. Eksport schedule poniżej."
        )

        col_y1, col_y2 = st.columns(2)
        yaw_max = col_y1.slider("Max kąt yaw [°]", 10.0, 35.0, 25.0, 5.0, key="yaw_max")
        yaw_min = col_y2.slider("Min kąt yaw [°]", -35.0, 0.0, 0.0, 5.0, key="yaw_min")

        if st.button("🎯 Optymalizuj yaw", key="g3_opt_yaw"):
            with st.spinner("Optymalizacja yaw..."):
                wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                farm.set_wind_data(wr_coarse)
                opt = Optimizer(farm)
                try:
                    result_yaw = opt.optimize_yaw(yaw_min=yaw_min, yaw_max=yaw_max)
                    st.session_state["g3_yaw_result"] = {
                        "before": result_yaw.initial_aep_gwh,
                        "after": result_yaw.optimized_aep_gwh,
                        "pct": result_yaw.aep_improvement_pct,
                        "time": result_yaw.elapsed_seconds,
                        "yaw_angles": result_yaw.yaw_angles,
                    }
                    fig = opt.plot_yaw_result(result_yaw)
                    st.session_state["g3_fig_yaw"] = fig_to_bytes(fig)
                except Exception as e:
                    st.session_state["g3_yaw_error"] = str(e)
                farm.set_wind_data(wind_rose)

        if "g3_yaw_result" in st.session_state:
            r = st.session_state["g3_yaw_result"]
            col1, col2, col3 = st.columns(3)
            col1.metric("AEP bez yaw", f"{r['before']:.1f} GWh")
            col2.metric("AEP z yaw", f"{r['after']:.1f} GWh", f"+{r['pct']:.2f}%")
            col3.metric("Czas", f"{r['time']:.0f}s")
        show_stored_fig("g3_fig_yaw")
        if "g3_yaw_error" in st.session_state:
            st.error(st.session_state["g3_yaw_error"])

        # Yaw max sweep
        st.divider()
        st.subheader("Yaw_max sweep — wpływ ograniczenia kąta na AEP")
        st.caption("Wykres AEP(yaw_max) dla różnych ograniczeń kąta yaw — pomocne dla Grupy 3.")
        if st.button("📈 Uruchom sweep yaw_max", key="g3_yaw_sweep"):
            with st.spinner("Sweep yaw_max..."):
                wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                farm.set_wind_data(wr_coarse)
                sweep_results = []
                yaw_maxes = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0]
                for ym in yaw_maxes:
                    try:
                        if ym == 0.0:
                            farm.run()
                            sweep_results.append({"yaw_max": 0.0, "aep": farm.get_aep_gwh()})
                        else:
                            opt = Optimizer(farm)
                            res = opt.optimize_yaw(yaw_min=0.0, yaw_max=ym)
                            sweep_results.append({"yaw_max": ym, "aep": res.optimized_aep_gwh})
                    except Exception as e:
                        sweep_results.append({"yaw_max": ym, "aep": None, "error": str(e)[:40]})
                farm.set_wind_data(wind_rose)
                st.session_state["g3_yaw_sweep"] = sweep_results

        if "g3_yaw_sweep" in st.session_state:
            sweep = st.session_state["g3_yaw_sweep"]
            df_sweep = pd.DataFrame(sweep)
            st.dataframe(df_sweep, hide_index=True, use_container_width=True)
            valid_sweep = [s for s in sweep if s.get("aep") is not None]
            if len(valid_sweep) > 1:
                fig_sw, ax_sw = plt.subplots(figsize=(8, 4))
                ax_sw.plot([s["yaw_max"] for s in valid_sweep],
                           [s["aep"] for s in valid_sweep],
                           "o-", color="#534AB7", linewidth=2)
                ax_sw.set_xlabel("Yaw max [°]")
                ax_sw.set_ylabel("AEP [GWh]")
                ax_sw.set_title("Wpływ ograniczenia yaw_max na AEP")
                ax_sw.grid(True, alpha=0.3)
                st.pyplot(fig_sw)
                plt.close()

    # ----- C. Curtailment baseline -----
    elif g3_section == "🛑 Curtailment (baza)":
        st.subheader("Hamowanie / wyłączanie turbin upstream")
        st.caption(
            "Baza pod algorytm Grupy 3: wyłączasz wybrane turbiny → "
            "downstream mają mniejszy ślad aerodynamiczny. Sprawdza czy farma "
            "jako całość zyskuje. Tu **ręczny wybór turbin**; Grupa 3 buduje na tym **algorytm**."
        )

        st.write(f"Farma ma **{farm.n_turbines}** turbin (indeksy 0..{farm.n_turbines - 1}).")
        curt_ids = st.multiselect(
            "Turbiny do wyłączenia",
            list(range(farm.n_turbines)),
            default=[],
            key="g3_curt_ids",
            format_func=lambda i: f"T{i}",
        )

        col_cm1, col_cm2 = st.columns(2)
        curt_mode = col_cm1.radio(
            "Tryb", ["Wyłączenie (disable)", "Derating do X MW"],
            horizontal=True, key="g3_curt_mode",
        )
        if curt_mode == "Derating do X MW":
            derate_mw = col_cm2.slider("Derating power [MW]", 1.0, turbine_info["rated_power"], 10.0, 0.5, key="g3_derate_mw")
        else:
            derate_mw = None

        if st.button("🛑 Symuluj curtailment", key="g3_run_curt"):
            with st.spinner("Obliczam..."):
                try:
                    # AEP bez curtailment
                    farm.fmodel.set(disable_turbines=None)
                    farm.fmodel.set_operation_model("simple")
                    farm.fmodel.reset_operation()
                    farm.set_wind_data(wind_rose)
                    farm.run()
                    aep_full = farm.get_aep_gwh()

                    n_findex = farm.fmodel.core.flow_field.n_findex
                    n_turb = farm.n_turbines

                    if curt_mode == "Wyłączenie (disable)":
                        disable_arr = np.full((n_findex, n_turb), False)
                        for tid in curt_ids:
                            disable_arr[:, tid] = True
                        farm.fmodel.set(disable_turbines=disable_arr)
                    else:  # derating
                        farm.fmodel.set(disable_turbines=np.full((n_findex, n_turb), False))
                        farm.fmodel.set_operation_model("simple-derating")
                        ps = np.full((n_findex, n_turb), None, dtype=object)
                        for tid in curt_ids:
                            ps[:, tid] = derate_mw * 1e6
                        farm.fmodel.set(power_setpoints=ps)

                    farm.run()
                    aep_curt = farm.get_aep_gwh()

                    # reset
                    farm.fmodel.set(disable_turbines=np.full((n_findex, n_turb), False))
                    farm.fmodel.set_operation_model("simple")
                    farm.fmodel.reset_operation()
                    farm.set_wind_data(wind_rose)

                    delta_pct = (aep_curt - aep_full) / aep_full * 100 if aep_full else 0
                    st.session_state["g3_curt_result"] = {
                        "full": aep_full, "curt": aep_curt, "delta_pct": delta_pct,
                        "ids": curt_ids, "mode": curt_mode,
                    }
                except Exception as e:
                    st.session_state["g3_curt_error"] = str(e)

        if "g3_curt_result" in st.session_state:
            r = st.session_state["g3_curt_result"]
            col1, col2, col3 = st.columns(3)
            col1.metric("AEP pełny", f"{r['full']:.1f} GWh")
            col2.metric("AEP z curtailment", f"{r['curt']:.1f} GWh", f"{r['delta_pct']:+.2f}%")
            col3.metric("Wyłączone", f"{len(r['ids'])} turbin")
            if r["delta_pct"] > 0:
                st.success("Curtailment poprawia AEP — wake-saving zadziałało!")
            else:
                st.warning(
                    "Curtailment obniża AEP. To **normalne** dla wielu konfiguracji — "
                    "Grupa 3 ma znaleźć kombinację (kierunek wiatru, zestaw turbin) "
                    "gdzie zysk z mniejszego cienia przewyższa stratę z wyłączonej turbiny."
                )
        if "g3_curt_error" in st.session_state:
            st.error(st.session_state["g3_curt_error"])

        st.divider()
        st.caption(
            "**Dla Grupy 3:** szablon algorytmu greedy_curtailment — sprawdzaj per kierunek wiatru, "
            "wybieraj iteracyjnie kandydatów do wyłączenia, akceptuj jeśli farma zyskuje. "
            "Patrz: `notebooks/05_advanced_floris.py` linie 160-205."
        )

    # ----- D. Helix (Active Wake Mixing) -----
    elif g3_section == "🌀 Active Wake Mixing (Helix)":
        st.subheader("Active Wake Mixing (Helix)")
        st.caption(
            "Modulacja pitch łopat powoduje 'helikoidalny' ślad który szybciej "
            "dyfunduje. Wymaga modelu wake `empirical_gauss`. Baza dla Grupy 3."
        )
        st.info(
            "Implementacja referencyjna w `notebooks/05_advanced_floris.py` linie 470-530. "
            "Grupa 3 buduje na tym własną analizę:\n"
            "1. Włącza `enable_active_wake_mixing=True` w configu wake\n"
            "2. Porównuje AEP: bez / z helix / z helix+yaw\n"
            "3. Analizuje wpływ amplitudy modulacji"
        )
        st.code('''# Szablon — uruchom w notebooku lub tu w UI:
from floris.utilities import load_yaml
from pathlib import Path
import floris

FLORIS_DIR = Path(floris.__file__).parent
helix_dict = load_yaml(FLORIS_DIR / "default_inputs.yaml")
helix_dict["farm"]["turbine_type"] = ["iea_15MW"]
helix_dict["flow_field"]["reference_wind_height"] = 150.0
helix_dict["wake"]["model_strings"]["velocity_model"] = "empirical_gauss"
helix_dict["wake"]["model_strings"]["deflection_model"] = "empirical_gauss"
helix_dict["wake"]["model_strings"]["turbulence_model"] = "wake_induced_mixing"
helix_dict["wake"]["enable_active_wake_mixing"] = True
# ... farm = FarmModel(custom_config=helix_dict) ...''', language="python")

    # ----- E. Porównanie strategii -----
    elif g3_section == "📊 Porównanie strategii":
        st.subheader("Porównanie strategii sterowania")
        st.caption(
            "Brak sterowania vs Yaw vs Curtailment vs kombinacje. "
            "Grupa 3 rozbuduje to o pełną macierz strategii."
        )

        if st.button("📊 Porównaj strategie", key="g3_compare"):
            with st.spinner("Liczę strategie..."):
                results = []
                try:
                    wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                    farm.set_wind_data(wr_coarse)
                    farm.run()
                    aep_base = farm.get_aep_gwh()
                    results.append({"Strategia": "Brak sterowania", "AEP [GWh]": round(aep_base, 2), "Δ [%]": 0.0})

                    # Yaw
                    try:
                        opt = Optimizer(farm)
                        r_yaw = opt.optimize_yaw()
                        results.append({
                            "Strategia": "Yaw (SerialRefine)",
                            "AEP [GWh]": round(r_yaw.optimized_aep_gwh, 2),
                            "Δ [%]": round(r_yaw.aep_improvement_pct, 2),
                        })
                    except Exception as e:
                        results.append({"Strategia": "Yaw (SerialRefine)", "AEP [GWh]": "ERR", "Δ [%]": str(e)[:30]})

                    farm.set_wind_data(wind_rose)
                    st.session_state["g3_compare"] = results
                except Exception as e:
                    st.session_state["g3_compare_error"] = str(e)

        if "g3_compare" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["g3_compare"]), hide_index=True, use_container_width=True)
            st.caption(
                "**TODO Grupy 3:** dodać wiersze: Curtailment (best greedy), "
                "Helix (AWM), Yaw+Helix, Yaw+Curtailment."
            )
        if "g3_compare_error" in st.session_state:
            st.error(st.session_state["g3_compare_error"])

    # ----- F. Eksport -----
    elif g3_section == "📁 Eksport":
        st.subheader("Eksport dla Grupy 3")

        # Yaw schedule
        st.write("**1. Yaw schedule** — uruchom najpierw optymalizację yaw w sekcji '🎯 Yaw'.")
        if "g3_yaw_result" in st.session_state and st.session_state["g3_yaw_result"].get("yaw_angles") is not None:
            yaw_arr = st.session_state["g3_yaw_result"]["yaw_angles"]
            # yaw_arr shape: (n_findex, n_turbines)
            rows = []
            for fi in range(yaw_arr.shape[0]):
                for ti in range(yaw_arr.shape[1]):
                    rows.append({"findex": fi, "turbine_id": ti, "yaw_deg": float(yaw_arr[fi, ti])})
            yaw_df = pd.DataFrame(rows)
            st.dataframe(yaw_df.head(20), hide_index=True, use_container_width=True)
            st.caption(f"Total: {len(yaw_df)} wierszy ({yaw_arr.shape[0]} warunków wiatrowych × {yaw_arr.shape[1]} turbin)")
            st.download_button(
                "⬇️ Pobierz yaw_schedule.csv",
                yaw_df.to_csv(index=False),
                file_name="yaw_schedule.csv", mime="text/csv",
            )
        else:
            st.info("Najpierw uruchom optymalizację yaw w sekcji '🎯 Yaw (wake steering)'.")

        st.divider()
        st.write("**2. Layout dla Grupy 3** (taki sam jak w zakładce Eksport, dla wygody):")
        layout_df_g3 = pd.DataFrame({
            "turbine_id": range(farm.n_turbines),
            "x_m": farm.layout_x,
            "y_m": farm.layout_y,
            "turbine_type": turbine_name,
        })
        st.dataframe(layout_df_g3, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Pobierz layout.csv (dla Grupy 3)",
            layout_df_g3.to_csv(index=False),
            file_name="layout.csv", mime="text/csv",
            key="g3_dl_layout",
        )


# =====================================================================
# TAB 6: ANALIZA AEP
# =====================================================================
with tab_aep:
    st.header("Analiza AEP")

    farm.set_wind_data(wind_rose)
    farm.run()
    calc = AEPCalculator(farm)
    summary = calc.compute_aep_summary()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AEP roczne", f"{summary['farm_aep_gwh']:.1f} GWh")
    col2.metric("Capacity Factor", f"{summary['capacity_factor_pct']:.1f}%")
    col3.metric("Wake Losses", f"{summary['wake_losses_pct']:.1f}%")
    col4.metric("Moc zainstalowana", f"{summary['total_capacity_mw']:.0f} MW")

    st.divider()

    # --- Analiza sezonowa/miesięczna ---
    if st.button("📅 Analiza sezonowa/miesięczna", key="aep_breakdown"):
        with st.spinner("Obliczam..."):
            fig = calc.plot_aep_breakdown(loader)
            st.session_state["fig_aep_breakdown"] = fig_to_bytes(fig)

    show_stored_fig("fig_aep_breakdown")

    st.divider()

    # --- Niepewność ---
    if st.button("📊 Analiza niepewności", key="aep_unc"):
        with st.spinner("Obliczam..."):
            unc_df = calc.uncertainty_sweep(wd_stds=[1.0, 3.0, 5.0, 7.0, 10.0])
            st.session_state["df_uncertainty"] = unc_df

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(unc_df["wd_std_deg"], unc_df["aep_uncertain_gwh"],
                    "o-", color="#534AB7", linewidth=2, markersize=8)
            ax.axhline(unc_df["aep_deterministic_gwh"].iloc[0],
                       color="#c8531a", linestyle="--", label="Deterministyczne")
            ax.set_xlabel("Niepewność σ [°]")
            ax.set_ylabel("AEP [GWh]")
            ax.set_title("Wpływ niepewności kierunku wiatru")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.session_state["fig_uncertainty"] = fig_to_bytes(fig)

    show_stored_df("df_uncertainty")
    show_stored_fig("fig_uncertainty")

    st.divider()

    # --- Scenariusze ---
    if st.button("🔄 Porównaj scenariusze", key="aep_scen"):
        with st.spinner("Obliczam..."):
            scenarios = {
                f"Grid {spacing_D}D": {"layout_type": "grid", "spacing_D": spacing_D},
                f"Grid {spacing_D + 2}D": {"layout_type": "grid", "spacing_D": spacing_D + 2},
                f"Staggered {spacing_D}D": {"layout_type": "staggered", "spacing_D": spacing_D},
                f"Staggered {spacing_D + 2}D": {"layout_type": "staggered", "spacing_D": spacing_D + 2},
            }
            scen_df = calc.compare_scenarios(scenarios)
            st.session_state["df_scenarios"] = scen_df[
                ["scenario", "aep_gwh", "wake_loss_pct", "capacity_factor_pct"]
            ]

            fig = calc.plot_scenario_comparison(scen_df)
            st.session_state["fig_scenarios"] = fig_to_bytes(fig)

            # Przywróć
            if layout_type == "grid":
                farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)
            elif layout_type == "staggered":
                farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D, offset=stagger_offset)
            farm.set_wind_data(wind_rose)

    show_stored_fig("fig_scenarios")
    show_stored_df("df_scenarios")


# =====================================================================
# TAB 7: TURBINY — Custom + Floating
# =====================================================================
with tab_turbines:
    st.header("🔧 Zarządzanie turbinami")

    turb_action = st.radio(
        "Akcja",
        ["Dodaj własną turbinę", "Podgląd biblioteki", "Podgląd krzywej mocy"],
        horizontal=True,
    )

    if turb_action == "Dodaj własną turbinę":
        st.subheader("Formularz nowej turbiny")
        st.caption(
            "Wypełnij parametry → wygenerujemy realistyczną krzywą mocy i Ct → "
            "plik YAML zostanie zapisany w `data/turbines/` i dodany do listy w sidebarze."
        )

        col1, col2 = st.columns(2)
        with col1:
            ct_name = st.text_input("Nazwa (bez spacji)", value="my_turbine_8MW")
            ct_power = st.number_input("Moc znamionowa [MW]", 1.0, 30.0, 8.0, 0.5)
            ct_diameter = st.number_input("Średnica rotora [m]", 50.0, 350.0, 180.0, 5.0)
            ct_hub = st.number_input("Hub height [m]", 50.0, 250.0, 120.0, 5.0)
        with col2:
            ct_cutin = st.number_input("Cut-in [m/s]", 2.0, 6.0, 3.0, 0.5)
            ct_rated = st.number_input("Rated speed [m/s]", 8.0, 16.0, 11.0, 0.5)
            ct_cutout = st.number_input("Cut-out [m/s]", 20.0, 35.0, 25.0, 1.0)
            ct_tsr = st.number_input("TSR", 4.0, 12.0, 8.0, 0.5)

        # Podgląd krzywej mocy
        if st.button("👀 Podgląd krzywej mocy", key="preview_curve"):
            ws, pw, ct_vals = generate_power_curve(
                rated_power_kw=ct_power * 1000,
                rotor_diameter=ct_diameter,
                cut_in=ct_cutin,
                rated_speed=ct_rated,
                cut_out=ct_cutout,
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(ws, np.array(pw) / 1000, "o-", color="#1e5c3a", markersize=3)
            ax1.set_xlabel("Prędkość wiatru [m/s]")
            ax1.set_ylabel("Moc [MW]")
            ax1.set_title(f"Krzywa mocy — {ct_name}")
            ax1.grid(True, alpha=0.3)
            ax1.axhline(ct_power, color="#c8531a", linestyle="--", alpha=0.5, label=f"Rated: {ct_power} MW")
            ax1.legend()

            ax2.plot(ws, ct_vals, "s-", color="#534AB7", markersize=3)
            ax2.set_xlabel("Prędkość wiatru [m/s]")
            ax2.set_ylabel("Ct [-]")
            ax2.set_title("Współczynnik ciągu")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            st.session_state["fig_custom_curve"] = fig_to_bytes(fig)

        show_stored_fig("fig_custom_curve")

        if st.button("💾 Zapisz turbinę", key="save_turbine"):
            try:
                key = register_custom_turbine(
                    name=ct_name,
                    rated_power_mw=ct_power,
                    rotor_diameter=ct_diameter,
                    hub_height=ct_hub,
                    cut_in=ct_cutin,
                    rated_speed=ct_rated,
                    cut_out=ct_cutout,
                    tsr=ct_tsr,
                )
                st.success(
                    f"Turbina **{ct_name}** zapisana jako `data/turbines/{ct_name}.yaml` "
                    f"i dodana do biblioteki jako **{key}**. "
                    f"Odśwież stronę (F5) żeby zobaczyć ją w sidebarze."
                )
            except Exception as e:
                st.error(f"Błąd: {e}")

    elif turb_action == "Podgląd biblioteki":
        st.subheader("Biblioteka turbin")
        lib_data = []
        for key, info in TURBINE_LIBRARY.items():
            lib_data.append({
                "ID": key,
                "Nazwa": info["name"],
                "D [m]": info["diameter"],
                "Hub [m]": info["hub_height"],
                "MW": info["rated_power"],
                "Floating": "🌊" if info.get("floating") else "",
                "Custom": "⭐" if info.get("custom_yaml") else "",
            })
        st.dataframe(pd.DataFrame(lib_data), use_container_width=True, hide_index=True)

    elif turb_action == "Podgląd krzywej mocy":
        st.subheader("Krzywa mocy aktualnej turbiny")
        st.caption(f"Turbina: **{turbine_info['name']}**")

        if st.button("📈 Pokaż krzywą mocy", key="show_power_curve"):
            with st.spinner("Obliczam..."):
                # Symuluj jedną turbinę przy różnych prędkościach
                single_farm = FarmModel(
                    wake_model=wake_model, turbine=turbine_name,
                    wave_period=wave_period, wave_height=wave_height,
                )
                ws_range = np.arange(3.0, 26.0, 0.5)
                powers = []
                for ws in ws_range:
                    single_farm.fmodel.set(
                        wind_directions=[270.0],
                        wind_speeds=[float(ws)],
                        turbulence_intensities=[0.06],
                    )
                    single_farm.fmodel.run()
                    p = single_farm.fmodel.get_turbine_powers() / 1e6  # MW
                    powers.append(float(p.flatten()[0]))

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ws_range, powers, "o-", color="#1e5c3a", markersize=3, linewidth=2)
                ax.set_xlabel("Prędkość wiatru [m/s]")
                ax.set_ylabel("Moc [MW]")
                ax.set_title(f"Krzywa mocy — {turbine_info['name']}")
                ax.axhline(turbine_info["rated_power"], color="#c8531a",
                           linestyle="--", alpha=0.5, label=f"Rated: {turbine_info['rated_power']} MW")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.session_state["fig_power_curve"] = fig_to_bytes(fig)

        show_stored_fig("fig_power_curve")


# =====================================================================
# TAB 8: EDYTOR LAYOUTU
# =====================================================================
with tab_editor:
    st.header("✏️ Edytor layoutu")
    st.caption(
        "Wybierz liczbę turbin i startowy układ — współrzędne wygenerują się automatycznie. "
        "Edytuj wartości w tabeli, potem kliknij **Oblicz AEP**."
    )

    col_ctrl, col_viz = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Ustawienia")

        ed_n_turbines = st.number_input("Liczba turbin", 2, 100, 25, key="ed_n")
        ed_init_layout = st.selectbox(
            "Układ startowy",
            ["grid", "staggered", "circular", "parallelogram"],
            format_func=lambda x: {
                "grid": "Siatka regularna",
                "staggered": "Siatka przesunięta",
                "circular": "Kołowy",
                "parallelogram": "Równoległobok",
            }[x],
            key="ed_layout",
        )
        ed_spacing = st.slider("Rozstaw [×D]", 4.0, 15.0, 7.0, 0.5, key="ed_sp")

        # Generuj współrzędne automatycznie (na podstawie ustawień)
        D = turbine_info["diameter"]
        n = ed_n_turbines

        if ed_init_layout == "grid":
            n_c = int(np.ceil(np.sqrt(n)))
            n_r = int(np.ceil(n / n_c))
            sp = ed_spacing * D
            xg, yg = np.meshgrid(np.arange(n_c) * sp, np.arange(n_r) * sp)
            ed_xs = xg.flatten()[:n]
            ed_ys = yg.flatten()[:n]

        elif ed_init_layout == "staggered":
            n_c = int(np.ceil(np.sqrt(n)))
            n_r = int(np.ceil(n / n_c))
            sp = ed_spacing * D
            ed_xs, ed_ys = [], []
            for row in range(n_r):
                for col in range(n_c):
                    if len(ed_xs) >= n:
                        break
                    ed_xs.append(col * sp + (0.5 * sp if row % 2 == 1 else 0))
                    ed_ys.append(row * sp)
            ed_xs = np.array(ed_xs)
            ed_ys = np.array(ed_ys)

        elif ed_init_layout == "circular":
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            r = ed_spacing * D
            ed_xs = r * np.cos(angles)
            ed_ys = r * np.sin(angles)

        elif ed_init_layout == "parallelogram":
            r1 = ed_spacing * D
            r2 = ed_spacing * D
            t1, t2 = np.radians(88), np.radians(18)
            v1 = np.array([r1 * np.cos(t1), r1 * np.sin(t1)])
            v2 = np.array([r2 * np.cos(t2), r2 * np.sin(t2)])
            ns = int(np.ceil(np.sqrt(n))) + 2
            pts_x, pts_y = [], []
            for i in range(-1, ns + 1):
                for j in range(-1, ns + 1):
                    pos = i * v1 + j * v2
                    pts_x.append(pos[0])
                    pts_y.append(pos[1])
            pts_x, pts_y = np.array(pts_x), np.array(pts_y)
            pts_x -= pts_x.min()
            pts_y -= pts_y.min()
            cx, cy = pts_x.mean(), pts_y.mean()
            dists = np.sqrt((pts_x - cx)**2 + (pts_y - cy)**2)
            idx = np.argsort(dists)[:n]
            ed_xs = pts_x[idx] - pts_x[idx].min()
            ed_ys = pts_y[idx] - pts_y[idx].min()

        # Załaduj do session_state (tylko jeśli parametry się zmieniły)
        editor_key = f"{ed_init_layout}_{ed_n_turbines}_{ed_spacing}"
        if st.session_state.get("_editor_key") != editor_key:
            st.session_state["editor_df"] = pd.DataFrame({
                "turbine_id": range(len(ed_xs)),
                "x_m": np.round(ed_xs, 1),
                "y_m": np.round(ed_ys, 1),
            })
            st.session_state["_editor_key"] = editor_key

        st.divider()
        st.subheader("Format CSV")
        st.code("turbine_id,x_m,y_m\n0,0.0,0.0\n1,1680.0,0.0\n...", language="csv")

        # Upload CSV
        uploaded = st.file_uploader("📂 Wczytaj layout CSV", type=["csv"], key="ed_upload")
        if uploaded is not None:
            try:
                up_df = pd.read_csv(uploaded)
                if "x_m" in up_df.columns and "y_m" in up_df.columns:
                    up_df["turbine_id"] = range(len(up_df))
                    st.session_state["editor_df"] = up_df[["turbine_id", "x_m", "y_m"]]
                    st.session_state["_editor_key"] = "uploaded"
                    st.success(f"Wczytano {len(up_df)} turbin.")
                    st.rerun()
                else:
                    st.error("CSV musi mieć kolumny: x_m, y_m")
            except Exception as e:
                st.error(f"Błąd: {e}")

    with col_viz:
        # Edytor tabeli
        st.subheader(f"Współrzędne ({len(st.session_state.get('editor_df', []))} turbin)")
        if "editor_df" in st.session_state:
            edited_df = st.data_editor(
                st.session_state["editor_df"],
                num_rows="fixed",
                use_container_width=True,
                column_config={
                    "turbine_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                    "x_m": st.column_config.NumberColumn("X [m]", min_value=-50000, max_value=50000, step=10.0, format="%.1f"),
                    "y_m": st.column_config.NumberColumn("Y [m]", min_value=-50000, max_value=50000, step=10.0, format="%.1f"),
                },
                key="ed_table",
                height=300,
            )
            st.session_state["editor_df"] = edited_df

    # Podgląd layoutu
    st.divider()
    col_plot, col_results = st.columns([2, 1])

    with col_plot:
        if "editor_df" in st.session_state:
            df_ed = st.session_state["editor_df"]
            fig_ed, ax_ed = plt.subplots(figsize=(8, 7))
            ax_ed.scatter(df_ed["x_m"], df_ed["y_m"], s=120, c="#1e5c3a",
                          edgecolors="white", linewidths=1.5, zorder=5)
            for _, row in df_ed.iterrows():
                ax_ed.annotate(str(int(row["turbine_id"])), (row["x_m"], row["y_m"]),
                               textcoords="offset points", xytext=(8, 8),
                               fontsize=8, color="#4a4a45")
                ax_ed.add_patch(plt.Circle(
                    (row["x_m"], row["y_m"]), turbine_info["diameter"] / 2,
                    fill=False, color="#d8d5cc", linewidth=0.5, linestyle="--"))

            # Minimalna odległość
            from scipy.spatial.distance import pdist
            coords = df_ed[["x_m", "y_m"]].values
            if len(coords) > 1:
                dists = pdist(coords)
                min_dist = dists.min()
                min_dist_D = min_dist / turbine_info["diameter"]
                color_dist = "#c8531a" if min_dist_D < 3.0 else "#1e5c3a"
                ax_ed.set_title(
                    f"Edytor layoutu — {len(df_ed)} turbin | "
                    f"Min. odl.: {min_dist_D:.1f}D ({min_dist:.0f}m)",
                    color=color_dist,
                )

            ax_ed.set_xlabel("X [m]")
            ax_ed.set_ylabel("Y [m]")
            ax_ed.set_aspect("equal")
            ax_ed.grid(True, alpha=0.3)
            st.pyplot(fig_ed)
            plt.close()

    with col_results:
        st.subheader("Wynik")

        # Walidacja minimalnej odległości
        ed_valid = True
        if "editor_df" in st.session_state:
            df_check = st.session_state["editor_df"]
            coords_check = df_check[["x_m", "y_m"]].values
            if len(coords_check) > 1:
                from scipy.spatial.distance import pdist
                dists_check = pdist(coords_check)
                min_d = dists_check.min()
                min_d_D = min_d / turbine_info["diameter"]

                if min_d < 1.0:  # < 1 metr = nakładają się
                    st.error(f"❌ Turbiny nachodzą na siebie! Min odl.: {min_d:.0f}m")
                    ed_valid = False
                elif min_d_D < 2.0:
                    st.warning(f"⚠️ Turbiny za blisko: {min_d_D:.1f}D ({min_d:.0f}m). Min. zalecane: 3D.")
                elif min_d_D < 3.0:
                    st.info(f"Min. odl.: {min_d_D:.1f}D ({min_d:.0f}m) — poniżej zalecanego 3D.")

        if st.button("⚡ Oblicz AEP", key="ed_calc", type="primary", disabled=not ed_valid):
            with st.spinner("Obliczam..."):
                df_ed = st.session_state["editor_df"]
                ed_farm = FarmModel(
                    wake_model=wake_model, turbine=turbine_name,
                    wave_period=wave_period, wave_height=wave_height,
                )
                ed_farm.set_layout_custom(
                    df_ed["x_m"].values, df_ed["y_m"].values, name="editor",
                )
                ed_farm.set_wind_data(wind_rose)
                ed_farm.run()
                ed_aep = ed_farm.get_aep_gwh()
                ed_rated = turbine_info["rated_power"] * len(df_ed)
                ed_cf = ed_aep / (ed_rated * 8.76) * 100 if ed_rated > 0 else 0
                try:
                    ed_wl = ed_farm.get_wake_losses_percent()
                except Exception:
                    ed_wl = 0.0

                st.session_state["ed_results"] = {
                    "aep": ed_aep, "cf": ed_cf, "wl": ed_wl,
                    "n": len(df_ed), "mw": ed_rated,
                }

        if "ed_results" in st.session_state:
            r = st.session_state["ed_results"]
            st.metric("AEP", f"{r['aep']:.1f} GWh")
            st.metric("Capacity Factor", f"{r['cf']:.1f}%")
            st.metric("Wake Losses", f"{r['wl']:.1f}%")
            st.metric("Moc zainstalowana", f"{r['mw']:.0f} MW")

        st.divider()
        if "editor_df" in st.session_state:
            st.download_button(
                "⬇️ Pobierz layout CSV",
                st.session_state["editor_df"].to_csv(index=False),
                file_name="custom_layout.csv", mime="text/csv",
            )


# =====================================================================
# TAB 9: 3D VISUALIZATION (Plotly)
# =====================================================================
with tab_3d:
    st.header("🌐 Wizualizacja 3D")

    viz_type = st.radio(
        "Typ wizualizacji",
        ["Layout 3D", "Profil wiatru 3D", "Mapa mocy 3D"],
        horizontal=True,
    )

    if viz_type == "Layout 3D":
        st.caption("Widok 3D rozmieszczenia turbin z proporcjonalnymi rotorami.")

        if st.button("🌐 Generuj widok 3D", key="gen_3d_layout"):
            try:
                import plotly.graph_objects as go

                x = farm.layout_x
                y = farm.layout_y
                D = turbine_info["diameter"]
                hh = turbine_info["hub_height"]

                fig3d = go.Figure()

                # Wieże (linie pionowe)
                for i in range(len(x)):
                    fig3d.add_trace(go.Scatter3d(
                        x=[x[i], x[i]], y=[y[i], y[i]], z=[0, hh],
                        mode="lines",
                        line=dict(color="#888888", width=4),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # Hub points
                fig3d.add_trace(go.Scatter3d(
                    x=x, y=y, z=np.full_like(x, hh),
                    mode="markers+text",
                    marker=dict(size=8, color="#1e5c3a", symbol="circle"),
                    text=[f"T{i}" for i in range(len(x))],
                    textposition="top center",
                    textfont=dict(size=9),
                    name="Turbiny",
                    hovertemplate="Turbina %{text}<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<extra></extra>",
                ))

                # Koła rotorów (przybliżone)
                theta = np.linspace(0, 2 * np.pi, 36)
                for i in range(len(x)):
                    rx = x[i] + np.zeros_like(theta)
                    ry = y[i] + (D / 2) * np.cos(theta)
                    rz = hh + (D / 2) * np.sin(theta)
                    fig3d.add_trace(go.Scatter3d(
                        x=rx, y=ry, z=rz,
                        mode="lines",
                        line=dict(color="#1e5c3a", width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # Podłoże (morze)
                x_range = [x.min() - 2 * D, x.max() + 2 * D]
                y_range = [y.min() - 2 * D, y.max() + 2 * D]
                fig3d.add_trace(go.Mesh3d(
                    x=[x_range[0], x_range[1], x_range[1], x_range[0]],
                    y=[y_range[0], y_range[0], y_range[1], y_range[1]],
                    z=[0, 0, 0, 0],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#1a5276",
                    opacity=0.3,
                    name="Morze",
                    hoverinfo="skip",
                ))

                fig3d.update_layout(
                    scene=dict(
                        xaxis_title="X [m]",
                        yaxis_title="Y [m]",
                        zaxis_title="Z [m]",
                        aspectmode="data",
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=0.8),
                        ),
                    ),
                    title=f"Farma 3D — {turbine_info['name']} | {farm.n_turbines} turbin",
                    height=700,
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.session_state["fig_3d_layout"] = fig3d

            except ImportError:
                st.error("Plotly nie jest zainstalowany. Uruchom: `pip install plotly`")

        if "fig_3d_layout" in st.session_state:
            st.plotly_chart(st.session_state["fig_3d_layout"], use_container_width=True)

    elif viz_type == "Profil wiatru 3D":
        st.caption("Rozkład prędkości wiatru w 3D — heatmap z wysokością = prędkość.")

        col1, col2 = st.columns(2)
        wd_3d = col1.slider("Kierunek [°]", 0.0, 350.0, 240.0, 10.0, key="wd_3d")
        ws_3d = col2.slider("Prędkość [m/s]", 5.0, 15.0, 9.0, 0.5, key="ws_3d")

        if st.button("🌐 Generuj profil 3D", key="gen_3d_wind"):
            try:
                import plotly.graph_objects as go

                farm.fmodel.set(
                    wind_directions=[wd_3d],
                    wind_speeds=[ws_3d],
                    turbulence_intensities=[0.06],
                )

                D = turbine_info["diameter"]
                x_bounds = (
                    float(farm.layout_x.min() - 3 * D),
                    float(farm.layout_x.max() + 8 * D),
                )
                y_bounds = (
                    float(farm.layout_y.min() - 3 * D),
                    float(farm.layout_y.max() + 3 * D),
                )

                hp = farm.fmodel.calculate_horizontal_plane(
                    height=turbine_info["hub_height"],
                    x_resolution=80,
                    y_resolution=40,
                    x_bounds=x_bounds,
                    y_bounds=y_bounds,
                )

                df_flow = hp.df
                u_col = "u" if "u" in df_flow.columns else df_flow.columns[-1]

                # Zamień NaN/inf
                df_flow[u_col] = df_flow[u_col].replace([np.inf, -np.inf], np.nan)
                df_flow[u_col] = df_flow[u_col].fillna(df_flow[u_col].median())

                # Pivot do 2D grid
                Z = df_flow.pivot_table(
                    values=u_col, index="x2", columns="x1",
                    aggfunc="mean",
                ).values
                x_unique = np.sort(df_flow["x1"].unique())
                y_unique = np.sort(df_flow["x2"].unique())

                # Wypełnij ewentualne NaN w Z
                Z = np.nan_to_num(Z, nan=float(np.nanmedian(Z)))

                fig3d = go.Figure()

                fig3d.add_trace(go.Surface(
                    x=x_unique,
                    y=y_unique,
                    z=Z,
                    colorscale="RdYlGn",
                    colorbar=dict(title="m/s", len=0.6),
                    opacity=0.92,
                    name="Prędkość",
                    hovertemplate="X: %{x:.0f}m<br>Y: %{y:.0f}m<br>V: %{z:.2f} m/s<extra></extra>",
                ))

                # Turbiny jako markery na surface
                fig3d.add_trace(go.Scatter3d(
                    x=farm.layout_x,
                    y=farm.layout_y,
                    z=np.full(farm.n_turbines, float(np.nanmax(Z)) * 1.02),
                    mode="markers+text",
                    marker=dict(size=5, color="#c8531a", symbol="diamond"),
                    text=[f"T{i}" for i in range(farm.n_turbines)],
                    textposition="top center",
                    textfont=dict(size=8),
                    name="Turbiny",
                ))

                fig3d.update_layout(
                    title=f"Profil wiatru 3D — WD={wd_3d}° WS={ws_3d} m/s | {wake_model.upper()}",
                    scene=dict(
                        xaxis_title="X [m]",
                        yaxis_title="Y [m]",
                        zaxis_title="Prędkość [m/s]",
                        camera=dict(eye=dict(x=1.5, y=-1.2, z=0.7)),
                        zaxis=dict(range=[float(np.nanmin(Z)) * 0.95, float(np.nanmax(Z)) * 1.05]),
                    ),
                    height=700,
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.session_state["fig_3d_wind"] = fig3d

                farm.set_wind_data(wind_rose)

            except ImportError:
                st.error("Plotly nie jest zainstalowany.")
            except Exception as e:
                st.error(f"Błąd: {e}")
                import traceback
                st.code(traceback.format_exc())

        if "fig_3d_wind" in st.session_state:
            st.plotly_chart(st.session_state["fig_3d_wind"], use_container_width=True)

    elif viz_type == "Mapa mocy 3D":
        st.caption("Średnia moc per turbina jako słupki 3D na mapie farmy.")

        if st.button("🌐 Generuj mapę mocy", key="gen_3d_power"):
            try:
                import plotly.graph_objects as go

                farm.set_wind_data(wind_rose)
                farm.run()
                powers_per_turbine = farm.get_turbine_powers_mw()
                # Bezpieczna konwersja — flatten na 1D
                mean_power = np.nanmean(powers_per_turbine, axis=0).flatten()

                x = farm.layout_x.flatten()
                y = farm.layout_y.flatten()
                D = turbine_info["diameter"]

                max_p = float(np.nanmax(mean_power))
                bar_h_scale = float(turbine_info["hub_height"]) * 0.8

                fig3d = go.Figure()

                # Słupki mocy z go.Bar3d nie istnieje — użyj Scatter3d z markerami
                colors_power = []
                heights = []
                for i in range(len(x)):
                    p = float(mean_power[i])
                    cv = p / max_p if max_p > 0 else 0
                    colors_power.append(cv)
                    heights.append(p)

                # Scatter3d — rozmiar = moc
                fig3d.add_trace(go.Scatter3d(
                    x=x, y=y, z=[0.0] * len(x),
                    mode="markers+text",
                    marker=dict(
                        size=[max(6, h / max_p * 25) if max_p > 0 else 8 for h in heights],
                        color=heights,
                        colorscale="RdYlGn",
                        colorbar=dict(title="MW", len=0.6),
                        opacity=0.9,
                    ),
                    text=[f"T{i}: {float(mean_power[i]):.1f} MW" for i in range(len(x))],
                    textposition="top center",
                    textfont=dict(size=8),
                    name="Moc",
                    hovertemplate="T%{text}<br>X: %{x:.0f}m<br>Y: %{y:.0f}m<extra></extra>",
                ))

                # Wieże
                for i in range(len(x)):
                    h_bar = float(mean_power[i]) / max_p * bar_h_scale if max_p > 0 else 0
                    fig3d.add_trace(go.Scatter3d(
                        x=[float(x[i]), float(x[i])],
                        y=[float(y[i]), float(y[i])],
                        z=[0, h_bar],
                        mode="lines",
                        line=dict(
                            color=f"rgb({int(255 * (1 - float(mean_power[i]) / max_p))},{int(200 * float(mean_power[i]) / max_p)},80)" if max_p > 0 else "rgb(128,128,80)",
                            width=8,
                        ),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                fig3d.update_layout(
                    title=f"Średnia moc per turbina — {turbine_info['name']}",
                    scene=dict(
                        xaxis_title="X [m]",
                        yaxis_title="Y [m]",
                        zaxis_title="Moc skalowana [MW]",
                        aspectmode="data",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                    ),
                    height=700,
                    margin=dict(l=0, r=0, t=40, b=0),
                )

                st.session_state["fig_3d_power"] = fig3d

            except ImportError:
                st.error("Plotly nie jest zainstalowany.")
            except Exception as e:
                st.error(f"Błąd: {e}")
                import traceback
                st.code(traceback.format_exc())

        if "fig_3d_power" in st.session_state:
            st.plotly_chart(st.session_state["fig_3d_power"], use_container_width=True)


# =====================================================================
# TAB 10: RAPORT PDF
# =====================================================================
with tab_report:
    st.header("📄 Generowanie raportu")

    st.write(
        "Wygeneruj profesjonalny raport PDF z aktualną konfiguracją farmy, "
        "wykresami i tabelami. Raport zawiera: stronę tytułową, podsumowanie, "
        "layout, dane wiatrowe, analizę AEP i flow field."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Zawartość raportu")
        report_items = [
            "Strona tytułowa z parametrami",
            "Podsumowanie wyników (AEP, CF, wake losses)",
            "Layout farmy (mapa + tabela współrzędnych)",
            "Dane wiatrowe (róża wiatrów, statystyki)",
            "Analiza AEP (sezonowa, miesięczna)",
            "Wizualizacja flow field (wake)",
        ]
        for item in report_items:
            st.write(f"• {item}")

    with col2:
        st.subheader("Parametry")
        st.write(f"**Turbina:** {turbine_info['name']}")
        st.write(f"**Layout:** {layout_type} @ {spacing_D}D")
        st.write(f"**Wake model:** {wake_model.upper()}")
        st.write(f"**Turbiny:** {farm.n_turbines}")
        st.write(f"**AEP:** {aep:.1f} GWh")

    st.divider()

    if st.button("📄 Generuj raport PDF", key="gen_report", type="primary"):
        with st.spinner("Generuję raport PDF..."):
            try:
                farm.set_wind_data(wind_rose)
                farm.run()

                rg = ReportGenerator(farm, loader)
                pdf_bytes = rg.generate()

                st.session_state["report_pdf"] = pdf_bytes
                st.session_state["report_ready"] = True
                st.success(f"Raport wygenerowany — {len(pdf_bytes) / 1024:.0f} KB")
            except Exception as e:
                st.error(f"Błąd generowania raportu: {e}")
                import traceback
                st.code(traceback.format_exc())

    if st.session_state.get("report_ready"):
        st.download_button(
            "⬇️ Pobierz raport PDF",
            data=st.session_state["report_pdf"],
            file_name=f"raport_farma_{turbine_name}_{wake_model}.pdf",
            mime="application/pdf",
            type="primary",
        )


# =====================================================================
# TAB 11: EKSPORT
# =====================================================================
with tab_export:
    st.header("Eksport danych dla grupy")

    st.subheader("Layout CSV — Temat 3 i 4")
    layout_df = pd.DataFrame({
        "turbine_id": range(farm.n_turbines),
        "x_m": farm.layout_x,
        "y_m": farm.layout_y,
        "turbine_type": turbine_name,
    })
    st.dataframe(layout_df, use_container_width=True)
    st.download_button(
        "⬇️ Pobierz layout.csv",
        layout_df.to_csv(index=False),
        file_name="layout.csv", mime="text/csv",
    )

    st.divider()

    st.subheader("AEP CSV — Temat 5")
    if st.button("Generuj AEP export", key="export_aep"):
        with st.spinner("Generuję..."):
            farm.set_wind_data(wind_rose)
            calc = AEPCalculator(farm)
            export_df = calc.export_for_team5(
                "outputs/exports/aep_timeseries.csv", loader=loader,
            )
            st.session_state["df_export_aep"] = export_df

    if "df_export_aep" in st.session_state:
        st.dataframe(st.session_state["df_export_aep"], use_container_width=True)
        st.download_button(
            "⬇️ Pobierz aep_timeseries.csv",
            st.session_state["df_export_aep"].to_csv(index=False),
            file_name="aep_timeseries.csv", mime="text/csv",
        )


# =====================================================================
# FOOTER
# =====================================================================
st.divider()
st.caption(
    f"Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych | "
    f"FLORIS v{floris.__version__} | {turbine_info['name']} | {wake_model.upper()}"
    + (f" | 🌊 Floating" if is_floating else "")
    + (f" | 📡 ERA5 ({era5_lat:.1f}°N, {era5_lon:.1f}°E)" if era5_active else " | 🎲 Mock data")
    + f" | Dashboard v3"
)
