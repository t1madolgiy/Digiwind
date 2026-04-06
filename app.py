"""
Wind Farm Layout Optimization — Streamlit Dashboard v2
=======================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Wyniki zapisywane w session_state — nie znikają po kliknięciu.

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
from src.farm_model import FarmModel, WAKE_MODELS, TURBINE_LIBRARY
from src.optimizer import Optimizer
from src.aep_calculator import AEPCalculator

import floris
from floris.utilities import load_yaml
import floris.layout_visualization as layoutviz

FLORIS_DIR = Path(floris.__file__).parent


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
            f"{TURBINE_LIBRARY[x]['name']} (D={TURBINE_LIBRARY[x]['diameter']}m)"
        ),
    )
    turbine_info = TURBINE_LIBRARY[turbine_name]

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
        options=["grid", "staggered", "circular"],
        format_func=lambda x: {
            "grid": "Siatka regularna",
            "staggered": "Siatka przesunięta",
            "circular": "Kołowy",
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

    st.header("Dane wiatrowe")
    weibull_A = st.slider("Weibull A [m/s]", 6.0, 14.0, 9.5, 0.5)
    weibull_k = st.slider("Weibull k", 1.5, 3.0, 2.1, 0.1)
    n_years = st.selectbox("Lata danych", [1, 2, 3], index=0)

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

loader = generate_wind_data(weibull_A, weibull_k, n_years)

wd_step = 30.0 if "30" in wr_resolution else (5.0 if "5" in wr_resolution else 10.0)
ws_step = 3.0 if "30" in wr_resolution else (1.0 if "5" in wr_resolution else 2.0)
wind_rose = loader.to_wind_rose(wd_step=wd_step, ws_step=ws_step)

farm = FarmModel(wake_model=wake_model, turbine=turbine_name, wind_data=wind_rose)

if layout_type == "grid":
    farm.set_layout_grid(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D)
elif layout_type == "staggered":
    farm.set_layout_staggered(n_rows=n_rows, n_cols=n_cols, spacing_D=spacing_D, offset=stagger_offset)
elif layout_type == "circular":
    farm.set_layout_circular(n_turbines=n_ring_turbines, radius_D=spacing_D, n_rings=n_rings)

farm.set_wind_data(wind_rose)
farm.run()
aep = farm.get_aep_gwh()
rated_total = turbine_info["rated_power"] * farm.n_turbines
cf = aep / (rated_total * 8.76) * 100 if rated_total > 0 else 0


# =====================================================================
# TABS
# =====================================================================
tab_overview, tab_wind, tab_flow, tab_compare, tab_optimize, tab_aep, tab_export = st.tabs([
    "📊 Przegląd", "🌬️ Wiatr", "🌊 Flow field",
    "⚖️ Porównania", "🎯 Optymalizacja", "⚡ AEP", "📁 Eksport",
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
            "Średnica": f"{turbine_info['diameter']} m",
            "Hub height": f"{turbine_info['hub_height']} m",
            "Moc znamionowa": f"{turbine_info['rated_power']} MW",
            "Model wake": wake_model.upper(),
            "Layout": layout_type,
            "Rozstaw": f"{spacing_D}D = {spacing_D * turbine_info['diameter']:.0f} m",
            "Liczba turbin": farm.n_turbines,
        }
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
        turbines_to_compare = st.multiselect(
            "Wybierz turbiny", list(TURBINE_LIBRARY.keys()),
            default=list(TURBINE_LIBRARY.keys()),
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
                farm.set_wind_data(wind_rose)

        show_stored_fig("fig_spacing")
        if "txt_spacing" in st.session_state:
            st.success(st.session_state["txt_spacing"])


# =====================================================================
# TAB 5: OPTYMALIZACJA
# =====================================================================
with tab_optimize:
    st.header("Optymalizacja layoutu")

    opt_method = st.radio("Metoda", ["Scipy (gradient)", "Yaw (wake steering)"],
                          horizontal=True)

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

                # Finalne AEP
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

    else:  # Yaw
        if st.button("🎯 Optymalizuj kąty yaw", key="opt_yaw"):
            with st.spinner("Optymalizacja yaw..."):
                wr_coarse = loader.to_wind_rose(wd_step=30.0, ws_step=3.0)
                farm.set_wind_data(wr_coarse)

                opt = Optimizer(farm)
                opt.set_boundaries_from_layout(margin_D=3.0)

                try:
                    result_yaw = opt.optimize_yaw()
                    st.session_state["yaw_result"] = {
                        "before": result_yaw.initial_aep_gwh,
                        "after": result_yaw.optimized_aep_gwh,
                        "pct": result_yaw.aep_improvement_pct,
                        "time": result_yaw.elapsed_seconds,
                    }
                    fig = opt.plot_yaw_result(result_yaw)
                    st.session_state["fig_yaw"] = fig_to_bytes(fig)
                except Exception as e:
                    st.session_state["yaw_error"] = str(e)

                farm.set_wind_data(wind_rose)

        if "yaw_result" in st.session_state:
            r = st.session_state["yaw_result"]
            col1, col2, col3 = st.columns(3)
            col1.metric("AEP bez yaw", f"{r['before']:.1f} GWh")
            col2.metric("AEP z yaw", f"{r['after']:.1f} GWh", f"+{r['pct']:.2f}%")
            col3.metric("Czas", f"{r['time']:.0f}s")
        show_stored_fig("fig_yaw")
        if "yaw_error" in st.session_state:
            st.error(st.session_state["yaw_error"])


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
# TAB 7: EKSPORT
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
)
