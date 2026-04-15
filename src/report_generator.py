"""
Report Generator for Wind Farm Layout Optimization Project
============================================================
Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych

Generuje raporty PDF z wynikami symulacji, wykresami i tabelami.
Używa reportlab (Platypus) do profesjonalnego formatowania.

Autor: Temat 2
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rejestracja fontu z polskimi znakami
# ---------------------------------------------------------------------------
_FONT_NAME = "Helvetica"  # fallback

def _register_unicode_font():
    """Rejestruje font TTF obsługujący polskie znaki (ą, ć, ę, ł, ń, ó, ś, ź, ż)."""
    global _FONT_NAME
    candidates = [
        # Windows
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        # Linux
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("UniFont", str(path)))
                pdfmetrics.registerFont(TTFont("UniFont-Bold", str(path).replace(".ttf", "bd.ttf")
                    if "arial" in str(path).lower() else str(path)))
                _FONT_NAME = "UniFont"
                logger.info(f"Zarejestrowano font: {path.name}")
                return
            except Exception:
                pass
    logger.warning("Brak fontu TTF z polskimi znakami — polskie znaki mogą nie działać.")

_register_unicode_font()


# ---------------------------------------------------------------------------
# Kolory projektu
# ---------------------------------------------------------------------------
COLOR_PRIMARY = HexColor("#1e5c3a")
COLOR_SECONDARY = HexColor("#534AB7")
COLOR_ACCENT = HexColor("#c8531a")
COLOR_BG_LIGHT = HexColor("#f5f5f0")
COLOR_TEXT = HexColor("#333333")
COLOR_HEADER_BG = HexColor("#1e5c3a")
COLOR_HEADER_FG = colors.white


# ---------------------------------------------------------------------------
# Style dokumentu
# ---------------------------------------------------------------------------
def _get_styles():
    """Zwraca skonfigurowane style paragrafów."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        fontName=_FONT_NAME,
        textColor=COLOR_PRIMARY,
        spaceAfter=6 * mm,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=12,
        fontName=_FONT_NAME,
        textColor=COLOR_TEXT,
        spaceAfter=10 * mm,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=16,
        fontName=_FONT_NAME,
        textColor=COLOR_PRIMARY,
        spaceBefore=8 * mm,
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        "SubSectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        fontName=_FONT_NAME,
        textColor=COLOR_SECONDARY,
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        fontName=_FONT_NAME,
        textColor=COLOR_TEXT,
        spaceAfter=3 * mm,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontSize=8,
        fontName=_FONT_NAME,
        textColor=HexColor("#888888"),
        alignment=TA_CENTER,
    ))

    return styles


# ---------------------------------------------------------------------------
# Helper — matplotlib figure → reportlab Image
# ---------------------------------------------------------------------------
def fig_to_image(fig: plt.Figure, width: float = 170 * mm) -> Image:
    """Konwertuje matplotlib Figure do reportlab Image."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    # Oblicz proporcje
    img = Image(buf, width=width)
    # Zachowaj aspect ratio
    w, h = fig.get_size_inches()
    ratio = h / w
    img.drawHeight = width * ratio
    img.drawWidth = width

    return img


def _make_table(data: list[list], col_widths: list = None) -> Table:
    """Tworzy sformatowaną tabelę."""
    table = Table(data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), COLOR_HEADER_FG),
        ("FONTNAME", (0, 0), (-1, -1), _FONT_NAME),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


# ---------------------------------------------------------------------------
# Główna klasa
# ---------------------------------------------------------------------------
class ReportGenerator:
    """Generator raportów PDF z wynikami symulacji farmy wiatrowej.

    Użycie:
        >>> from src.report_generator import ReportGenerator
        >>> rg = ReportGenerator(farm, loader)
        >>> pdf_bytes = rg.generate()
        >>> # lub
        >>> rg.save("outputs/raport.pdf")
    """

    def __init__(self, farm, loader=None):
        """
        Args:
            farm: FarmModel z przeprowadzoną symulacją.
            loader: WindDataLoader (opcjonalny, do sezonowego rozbicia).
        """
        self.farm = farm
        self.loader = loader
        self._styles = _get_styles()

    def generate(self) -> bytes:
        """Generuje raport PDF i zwraca jako bytes.

        Returns:
            Bytes raportu PDF.
        """
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=15 * mm,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            title="Raport — Farma Wiatrowa",
            author="Temat 2 — DIGIWIND",
        )

        story = []
        self._add_title_page(story)
        story.append(PageBreak())
        self._add_summary_section(story)
        self._add_layout_section(story)
        story.append(PageBreak())
        self._add_wind_section(story)
        story.append(PageBreak())
        self._add_aep_section(story)
        story.append(PageBreak())
        self._add_wake_section(story)
        self._add_wake_comparison_section(story)
        story.append(PageBreak())
        self._add_turbine_power_section(story)
        self._add_spacing_section(story)
        story.append(PageBreak())
        self._add_footer(story)

        doc.build(story)
        return buf.getvalue()

    def save(self, filepath: Union[str, Path] = "outputs/raport_farma.pdf") -> Path:
        """Generuje i zapisuje raport PDF.

        Args:
            filepath: Ścieżka do pliku.

        Returns:
            Ścieżka do zapisanego pliku.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        pdf_bytes = self.generate()
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)

        logger.info(f"Raport zapisany: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Sekcje raportu
    # ------------------------------------------------------------------
    def _add_title_page(self, story: list) -> None:
        """Strona tytułowa."""
        s = self._styles

        story.append(Spacer(1, 30 * mm))
        story.append(Paragraph(
            "Raport — Farma Wiatrowa Offshore",
            s["ReportTitle"],
        ))
        story.append(Paragraph(
            "Temat 2 — Lokalizacja i rozmieszczenie farm wiatrowych",
            s["ReportSubtitle"],
        ))

        story.append(Spacer(1, 5 * mm))
        story.append(HRFlowable(
            width="60%", thickness=2, color=COLOR_PRIMARY,
            spaceAfter=8 * mm, spaceBefore=2 * mm,
        ))

        # Info box
        info_data = [
            ["Parametr", "Wartość"],
            ["Turbina", self.farm.turbine_info["name"]],
            ["Średnica rotora", f"{self.farm.D:.0f} m"],
            ["Hub height", f"{self.farm.hub_height:.0f} m"],
            ["Moc znamionowa", f"{self.farm.turbine_info['rated_power']:.1f} MW"],
            ["Liczba turbin", str(self.farm.n_turbines)],
            ["Moc zainstalowana", f"{self.farm.turbine_info['rated_power'] * self.farm.n_turbines:.0f} MW"],
            ["Model wake", self.farm.wake_model_name.upper()],
            ["Layout", self.farm._layout_type],
            ["Lokalizacja", "Bałtyk Południowy (~54.5°N, 16.5°E)"],
        ]

        if self.farm.is_floating:
            info_data.append(["Typ", "Pływająca (floating)"])
            info_data.append(["Tp / Hs", f"{self.farm._wave_period}s / {self.farm._wave_height}m"])

        table = _make_table(info_data, col_widths=[55 * mm, 85 * mm])
        story.append(table)

        story.append(Spacer(1, 15 * mm))
        story.append(Paragraph(
            f"Data wygenerowania: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            s["SmallText"],
        ))
        story.append(Paragraph(
            "Wygenerowano automatycznie — Wind Farm Optimizer (FLORIS)",
            s["SmallText"],
        ))

    def _add_summary_section(self, story: list) -> None:
        """Podsumowanie wyników."""
        s = self._styles

        story.append(Paragraph("1. Podsumowanie wyników", s["SectionHeader"]))

        # Oblicz metryki
        self.farm.run()
        aep = self.farm.get_aep_gwh()
        rated_total = self.farm.turbine_info["rated_power"] * self.farm.n_turbines
        cf = aep / (rated_total * 8.76) * 100 if rated_total > 0 else 0

        try:
            wake_loss = self.farm.get_wake_losses_percent()
        except Exception:
            wake_loss = 0.0

        summary_data = [
            ["Metryka", "Wartość", "Jednostka"],
            ["AEP (Annual Energy Production)", f"{aep:.1f}", "GWh/rok"],
            ["Capacity Factor", f"{cf:.1f}", "%"],
            ["Wake Losses", f"{wake_loss:.1f}", "%"],
            ["Moc zainstalowana", f"{rated_total:.0f}", "MW"],
            ["Produkcja na turbinę", f"{aep / self.farm.n_turbines:.1f}", "GWh/rok"],
        ]

        table = _make_table(summary_data, col_widths=[70 * mm, 40 * mm, 40 * mm])
        story.append(table)

    def _add_layout_section(self, story: list) -> None:
        """Sekcja z layoutem farmy."""
        s = self._styles

        story.append(Paragraph("2. Layout farmy", s["SectionHeader"]))

        story.append(Paragraph(
            f"Farma składa się z {self.farm.n_turbines} turbin "
            f"{self.farm.turbine_info['name']} rozmieszczonych w układzie "
            f"<b>{self.farm._layout_type}</b>. "
            f"Średnica rotora: {self.farm.D:.0f} m, wysokość piasty: {self.farm.hub_height:.0f} m.",
            s["ReportBody"],
        ))

        # Wykres layoutu
        try:
            fig = self.farm.plot_layout(figsize=(7, 6), show_spacing=True)
            story.append(fig_to_image(fig, width=140 * mm))
        except Exception as e:
            story.append(Paragraph(f"(Wykres layoutu niedostępny: {e})", s["SmallText"]))

        # Tabela współrzędnych (max 25 turbin)
        if self.farm.n_turbines <= 30:
            story.append(Paragraph("Współrzędne turbin", s["SubSectionHeader"]))
            coords = [["ID", "X [m]", "Y [m]"]]
            for i in range(self.farm.n_turbines):
                coords.append([
                    str(i),
                    f"{self.farm.layout_x[i]:.0f}",
                    f"{self.farm.layout_y[i]:.0f}",
                ])
            # Split into columns if many turbines
            table = _make_table(coords, col_widths=[20 * mm, 40 * mm, 40 * mm])
            story.append(table)

    def _add_wind_section(self, story: list) -> None:
        """Sekcja danych wiatrowych."""
        s = self._styles

        if self.loader is None:
            return

        story.append(Paragraph("3. Dane wiatrowe", s["SectionHeader"]))

        stats = self.loader.summary()
        story.append(Paragraph(
            f"Dane: {stats['n_records']} godzinowych rekordów. "
            f"Średnia prędkość wiatru: <b>{stats['wind_speed']['mean']:.1f} m/s</b>, "
            f"Weibull k={stats['wind_speed']['weibull_k_est']:.2f}, "
            f"A={stats['wind_speed']['weibull_A_est']:.1f} m/s. "
            f"Kierunek dominujący: {stats['wind_direction']['dominant']:.0f}°.",
            s["ReportBody"],
        ))

        # Róża wiatrów
        try:
            fig = self.loader.plot_wind_rose(figsize=(6, 6))
            story.append(fig_to_image(fig, width=120 * mm))
        except Exception as e:
            story.append(Paragraph(f"(Roza wiatrow niedostepna: {e})", s["SmallText"]))

        # Seria czasowa
        try:
            fig = self.loader.plot_time_series(figsize=(12, 7))
            story.append(fig_to_image(fig, width=170 * mm))
            story.append(Paragraph(
                "Powyzszy wykres przedstawia serie czasowa predkosci i kierunku wiatru, "
                "histogram z dopasowaniem Weibulla oraz zaleznosc TI od predkosci.",
                s["SmallText"],
            ))
        except Exception as e:
            story.append(Paragraph(f"(Seria czasowa niedostepna: {e})", s["SmallText"]))

    def _add_aep_section(self, story: list) -> None:
        """Sekcja analizy AEP."""
        s = self._styles

        if self.loader is None:
            return

        story.append(Paragraph("4. Analiza produkcji energii", s["SectionHeader"]))

        # AEP breakdown plot
        try:
            from .aep_calculator import AEPCalculator
            calc = AEPCalculator(self.farm)
            fig = calc.plot_aep_breakdown(self.loader, figsize=(12, 8))
            story.append(fig_to_image(fig, width=170 * mm))
        except Exception as e:
            story.append(Paragraph(f"(Wykres AEP niedostępny: {e})", s["SmallText"]))

        # Tabela sezonowa
        try:
            from .aep_calculator import AEPCalculator
            calc = AEPCalculator(self.farm)
            seasonal = calc.compute_seasonal_aep(self.loader)

            story.append(Paragraph("Rozbicie sezonowe", s["SubSectionHeader"]))
            season_data = [["Sezon", "AEP [GWh]", "Śr. wiatr [m/s]", "Kierunek dom."]]
            for _, row in seasonal.iterrows():
                season_data.append([
                    row["season"],
                    f"{row['aep_gwh']:.1f}",
                    f"{row['mean_wind_speed_ms']:.1f}",
                    f"{row['dominant_direction_deg']:.0f}°",
                ])
            table = _make_table(season_data, col_widths=[40 * mm, 35 * mm, 35 * mm, 35 * mm])
            story.append(table)

            # Przywróć pełne dane
            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)

        except Exception as e:
            story.append(Paragraph(f"(Tabela sezonowa niedostępna: {e})", s["SmallText"]))

    def _add_wake_section(self, story: list) -> None:
        """Sekcja z flow field."""
        s = self._styles

        story.append(Paragraph("5. Pole przepływu (wake)", s["SectionHeader"]))

        story.append(Paragraph(
            f"Wizualizacja śladu aerodynamicznego przy wietrze z kierunku dominującego "
            f"(240°) i prędkości 9 m/s. Model wake: <b>{self.farm.wake_model_name.upper()}</b>.",
            s["ReportBody"],
        ))

        try:
            fig = self.farm.plot_flow_field(
                wind_direction=240.0, wind_speed=9.0, ti=0.06, figsize=(12, 5),
            )
            story.append(fig_to_image(fig, width=170 * mm))

            # Drugi kierunek
            story.append(Paragraph(
                "Dla porownania — pole przeplywu przy wietrze z kierunku 180° (poludnie):",
                s["ReportBody"],
            ))
            fig2 = self.farm.plot_flow_field(
                wind_direction=180.0, wind_speed=9.0, ti=0.06, figsize=(12, 5),
            )
            story.append(fig_to_image(fig2, width=170 * mm))

            # Przywróć wind data
            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)
        except Exception as e:
            story.append(Paragraph(f"(Flow field niedostępny: {e})", s["SmallText"]))

    def _add_wake_comparison_section(self, story: list) -> None:
        """Porownanie modeli wake."""
        s = self._styles

        story.append(Paragraph("6. Porownanie modeli wake", s["SectionHeader"]))

        story.append(Paragraph(
            "Porownanie AEP i strat wake dla roznych modeli aerodynamicznych "
            "na tym samym layoucie i danych wiatrowych. Roznice wynikaja z roznych "
            "zalozen fizycznych kazdego modelu.",
            s["ReportBody"],
        ))

        try:
            from .farm_model import WAKE_MODELS, FarmModel
            models = ["jensen", "gch", "turbopark", "cc"]
            wake_data = [["Model", "AEP [GWh]", "Wake losses [%]"]]

            original_wake = self.farm.wake_model_name
            for m in models:
                try:
                    self.farm.switch_wake_model(m)
                    if self.farm._wind_data is not None:
                        self.farm.fmodel.set(wind_data=self.farm._wind_data)
                    self.farm.run()
                    a = self.farm.get_aep_gwh()
                    wl = self.farm.get_wake_losses_percent()
                    wake_data.append([m.upper(), f"{a:.1f}", f"{wl:.1f}"])
                except Exception:
                    wake_data.append([m.upper(), "N/A", "N/A"])

            self.farm.switch_wake_model(original_wake)
            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)

            table = _make_table(wake_data, col_widths=[45 * mm, 45 * mm, 45 * mm])
            story.append(table)

        except Exception as e:
            story.append(Paragraph(f"(Porownanie niedostepne: {e})", s["SmallText"]))

    def _add_turbine_power_section(self, story: list) -> None:
        """Moc per turbina."""
        s = self._styles

        story.append(Paragraph("7. Produkcja per turbina", s["SectionHeader"]))

        story.append(Paragraph(
            "Roczna produkcja energii (AEP) kazdej turbiny w farmie. Turbiny w srodku "
            "farmy maja nizsza produkcje z powodu strat wake od turbin upstream.",
            s["ReportBody"],
        ))

        try:
            self.farm.run()
            # Bezpieczna metoda — AEP per turbina (nie powers!)
            aep_per_turbine = self.farm.fmodel.get_turbine_AEPs() / 1e9  # GWh
            # Upewnij sie ze 1D
            if aep_per_turbine.ndim > 1:
                aep_per_turbine = np.nansum(aep_per_turbine, axis=0)
            aep_per_turbine = aep_per_turbine.flatten()

            n_turb = self.farm.n_turbines
            # Obetnij do rzeczywistej liczby turbin
            aep_per_turbine = aep_per_turbine[:n_turb]
            mean_aep = float(np.nanmean(aep_per_turbine))

            fig, ax = plt.subplots(figsize=(10, 4))
            turbine_ids = np.arange(len(aep_per_turbine))
            colors_bar = ["#1e5c3a" if a >= mean_aep else "#c8531a" for a in aep_per_turbine]
            ax.bar(turbine_ids, aep_per_turbine, color=colors_bar, edgecolor="white", linewidth=0.5)
            ax.axhline(mean_aep, color="#534AB7", linestyle="--",
                       label=f"Srednia: {mean_aep:.2f} GWh")
            ax.set_xlabel("ID turbiny")
            ax.set_ylabel("AEP [GWh]")
            ax.set_title(f"AEP per turbina ({n_turb} turbin)")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")
            story.append(fig_to_image(fig, width=160 * mm))

        except Exception as e:
            story.append(Paragraph(f"(Wykres AEP per turbina niedostepny: {e})", s["SmallText"]))

    def _add_spacing_section(self, story: list) -> None:
        """Analiza rozstawu."""
        s = self._styles

        story.append(Paragraph("8. Wrazliwosc na rozstaw", s["SectionHeader"]))

        story.append(Paragraph(
            "Analiza wplywu rozstawu miedzy turbinami na AEP i straty wake. "
            "Wiekszy rozstaw zmniejsza straty wake, ale wymaga wiekszego obszaru.",
            s["ReportBody"],
        ))

        try:
            # Zapisz oryginalny layout
            orig_x = self.farm.layout_x.copy()
            orig_y = self.farm.layout_y.copy()
            orig_type = self.farm._layout_type

            spacings = np.arange(5.0, 12.5, 1.0)
            aeps = []
            wakes = []

            n_r = int(np.ceil(np.sqrt(self.farm.n_turbines)))
            n_c = int(np.ceil(self.farm.n_turbines / n_r))

            for sp in spacings:
                self.farm.set_layout_grid(n_rows=n_r, n_cols=n_c, spacing_D=sp)
                if self.farm._wind_data is not None:
                    self.farm.fmodel.set(wind_data=self.farm._wind_data)
                self.farm.run()
                aeps.append(self.farm.get_aep_gwh())
                try:
                    wakes.append(self.farm.get_wake_losses_percent())
                except Exception:
                    wakes.append(0)

            # Przywroc oryginalny layout
            self.farm.set_layout_custom(orig_x, orig_y, name=orig_type)
            if self.farm._wind_data is not None:
                self.farm.fmodel.set(wind_data=self.farm._wind_data)

            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(spacings, aeps, "o-", color="#1e5c3a", linewidth=2, label="AEP")
            ax1.set_xlabel("Rozstaw [xD]")
            ax1.set_ylabel("AEP [GWh]", color="#1e5c3a")
            ax2 = ax1.twinx()
            ax2.plot(spacings, wakes, "s--", color="#c8531a", linewidth=2, label="Wake losses")
            ax2.set_ylabel("Wake losses [%]", color="#c8531a")
            fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
            ax1.grid(True, alpha=0.3)
            ax1.set_title("AEP i straty wake vs rozstaw")
            story.append(fig_to_image(fig, width=150 * mm))

            # Tabela
            sp_data = [["Rozstaw [xD]", "AEP [GWh]", "Wake losses [%]"]]
            for i, sp in enumerate(spacings):
                sp_data.append([f"{sp:.1f}", f"{aeps[i]:.1f}", f"{wakes[i]:.1f}"])
            table = _make_table(sp_data, col_widths=[40 * mm, 40 * mm, 40 * mm])
            story.append(table)

        except Exception as e:
            story.append(Paragraph(f"(Analiza rozstawu niedostepna: {e})", s["SmallText"]))

    def _add_footer(self, story: list) -> None:
        """Stopka / disclaimer."""
        s = self._styles

        story.append(HRFlowable(
            width="100%", thickness=1, color=COLOR_PRIMARY,
            spaceAfter=5 * mm, spaceBefore=5 * mm,
        ))
        story.append(Paragraph(
            "Raport wygenerowany automatycznie przez Wind Farm Optimizer. "
            "Dane wiatrowe: model syntetyczny (Weibull/von Mises). "
            "Silnik symulacji: FLORIS. "
            "Wyniki służą celom badawczym i edukacyjnym.",
            s["SmallText"],
        ))
