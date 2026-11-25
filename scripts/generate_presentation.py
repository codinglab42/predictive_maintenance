#!/usr/bin/env python3
"""Generate the ten-slide demo from saved IMS and C++ benchmark results."""

from __future__ import annotations

import json
import csv
import math
import statistics
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


METRICS_PATH = Path("/mnt/shared/predictive-maintenance/processed/ims_test_metrics.json")
CORE_FIT_PATH = Path("/mnt/shared/predictive-maintenance/processed/ml_core_fit_benchmark_20261125.json")
CORE_INFERENCE_PATH = Path("/mnt/shared/predictive-maintenance/processed/ml_core_inference_benchmark_20261125.csv")
SCREENSHOT_PATH = Path("/mnt/shared/predictive-maintenance/Screenshot_2026-11-25_APP.png")
OUTPUT_PATH = Path("/mnt/shared/predictive-maintenance/PdM_AI_Demo.pptx")

WIDTH = 13.333
HEIGHT = 7.5
SLIDE_COUNT = 10
APPLICATION_TITLE = "Progetto applicativo Python: pipeline NASA IMS"
BG = "0E1117"
PANEL = "151A22"
PANEL_ALT = "1B2430"
CYAN = "38BDF8"
GREEN = "10B981"
AMBER = "F59E0B"
WHITE = "F8FAFC"
MUTED = "94A3B8"
GRID = "2B3543"
FONT = "Aptos"

MODEL_ORDER = ("logistic_regression", "neural_network_cpp", "gru_cpp")
MODEL_SHORT_NAMES = {
    "logistic_regression": "Logistica C++",
    "neural_network_cpp": "MLP C++",
    "gru_cpp": "GRU C++",
}
CORE_MODELS = ("Ridge", "Logistic", "MLP", "SimpleRNN", "LSTM", "GRU")


def color(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value)


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"IMS metrics JSON not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid IMS metrics JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"IMS metrics JSON must contain an object: {path}")
    models = document.get("models")
    if not isinstance(models, dict) or set(MODEL_ORDER) - set(models):
        raise ValueError("IMS metrics JSON must contain all three benchmark models")
    for model_id in MODEL_ORDER:
        model = models[model_id]
        if not isinstance(model, dict):
            raise ValueError(f"Invalid metrics for {model_id}")
        for key in ("precision", "recall", "f1_score", "threshold", "training_seconds", "inference_seconds"):
            try:
                value = float(model[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Missing or invalid {key} for {model_id}") from error
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {key} for {model_id}")
            if key in {"precision", "recall", "f1_score", "threshold"} and not 0.0 <= value <= 1.0:
                raise ValueError(f"Out-of-range {key} for {model_id}")
        if model.get("first_alarm_bearing_1_file") is not None:
            try:
                int(model["first_alarm_bearing_1_file"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid first alarm index for {model_id}") from error
    active_model = document.get("active_model")
    if active_model not in MODEL_ORDER:
        raise ValueError("IMS metrics JSON has an invalid active_model")
    try:
        test_samples = int(document["test_samples"])
        full_test_samples = int(document["full_test_samples"])
        time_steps = int(document["gru_network"]["time_steps"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("IMS metrics JSON is missing test or GRU window dimensions") from error
    if test_samples <= 0 or full_test_samples < test_samples or time_steps <= 0:
        raise ValueError("IMS metrics JSON contains invalid test or GRU window dimensions")
    return document


def load_core_benchmarks(fit_path: Path, inference_path: Path) -> list[dict[str, Any]]:
    document = json.loads(fit_path.read_text(encoding="utf-8"))
    summary = document["summary"]
    cases = (
        ("Ridge", "50 feature · λ 0,1", "linear", "regularization", "lambda", 0.1, 1024),
        ("Logistic", "10 feature", "logistic", "sample_scaling", "samples", 10000, 1024),
        ("MLP", "10 → 32 → 16 → 1", "neural", "sample_scaling", "samples", 10000, 1024),
        ("SimpleRNN", "8 step · 8 unità", "sequences", "recurrent", "type", "SimpleRNN", 64),
        ("LSTM", "8 step · 8 unità", "sequences", "recurrent", "type", "LSTM", 64),
        ("GRU", "8 step · 8 unità", "sequences", "recurrent", "type", "GRU", 64),
    )
    with inference_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if set(row["model"] for row in rows) != set(CORE_MODELS):
        raise ValueError("Core inference results must contain the six expected models")
    results = []
    for model, configuration, family, section, key, value, batch_size in cases:
        matches = [row for row in summary[family][section] if row[key] == value]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one fit result for {model}")
        fit_ms = float(matches[0]["time_ms"])
        measurements = [row for row in rows if row["model"] == model]
        if len(measurements) != 5 or {int(row["run"]) for row in measurements} != set(range(1, 6)):
            raise ValueError(f"Expected five inference measurements for {model}")
        if any(int(row["batch_size"]) != batch_size or int(row["calls"]) != 50 for row in measurements):
            raise ValueError(f"Unexpected inference workload for {model}")
        values = [float(row["us_per_item"]) for row in measurements]
        if not math.isfinite(fit_ms) or fit_ms <= 0 or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Invalid core benchmark time for {model}")
        median_us = statistics.median(values)
        results.append({"model": model, "configuration": configuration, "fit_ms": fit_ms,
                        "inference_us": median_us, "items_per_second": 1_000_000 / median_us})
    return results


def add_text(
    slide,
    text: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    size: float = 16,
    text_color: str = WHITE,
    bold: bool = False,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    valign: MSO_ANCHOR = MSO_ANCHOR.TOP,
    margin: float = 0.02,
) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.alignment = align
        paragraph.font.name = FONT
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color(text_color)


def add_rect(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    *,
    line: str | None = None,
    rounded: bool = True,
):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(
        shape_type, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color(fill)
    shape.line.color.rgb = color(line or fill)
    shape.line.width = Pt(1)
    return shape


def add_line(slide, x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1.2) -> None:
    line = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = color(stroke)
    line.line.width = Pt(width)


def add_base(slide, page_number: int, kicker: str) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color(BG)
    add_text(slide, kicker.upper(), 0.68, 0.28, 8.5, 0.24, size=9, text_color=CYAN, bold=True)
    add_line(slide, 0.68, 7.12, 12.65, 7.12, GRID, 0.8)
    add_text(slide, "Maurizio Penna  •  Deda Speed  •  26/11/2026", 0.7, 7.18, 9.5, 0.23, size=8, text_color=MUTED)
    add_text(slide, f"{page_number:02d} / {SLIDE_COUNT:02d}", 11.6, 7.18, 1.0, 0.2, size=8, text_color=MUTED, align=PP_ALIGN.RIGHT)


def add_header(slide, page_number: int, kicker: str, title: str, subtitle: str | None = None) -> None:
    add_base(slide, page_number, kicker)
    add_text(slide, title, 0.68, 0.62, 11.95, 0.65, size=27, bold=True)
    if subtitle:
        add_text(slide, subtitle, 0.72, 1.31, 11.8, 0.42, size=12, text_color=MUTED)


def add_card(slide, x: float, y: float, w: float, h: float, accent: str, title: str, body: str, *, title_size: int = 16, body_size: int = 12) -> None:
    add_rect(slide, x, y, w, h, PANEL, line=GRID)
    add_rect(slide, x, y, 0.08, h, accent, rounded=False)
    add_text(slide, title, x + 0.24, y + 0.2, w - 0.42, 0.4, size=title_size, bold=True, text_color=WHITE)
    add_text(slide, body, x + 0.24, y + 0.7, w - 0.44, h - 0.86, size=body_size, text_color=MUTED)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_seconds(value: float) -> str:
    return f"{value * 1000:.2f} ms" if value < 0.1 else f"{value:.2f} s"


def slide_title(prs: Presentation, metrics: dict[str, Any]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_base(slide, 1, "ml_core / libreria C++")
    add_text(
        slide,
        "ml_core: motore di Machine Learning\nproprietario in C++",
        1.05, 0.78, 11.3, 1.15, size=27, bold=True,
    )
    add_text(slide, "Maurizio Penna  •  26 Novembre 2026", 1.08, 2.03, 10.7, 0.3, size=13, text_color=CYAN, bold=True)
    add_rect(slide, 0.78, 2.5, 5.75, 3.34, PANEL, line=GRID)
    add_rect(slide, 6.78, 2.5, 5.75, 3.34, PANEL_ALT, line=GRID)
    add_text(slide, "ORIGINI  /  LIBRERIA CORE", 1.03, 2.69, 5.25, 0.31, size=12, text_color=CYAN, bold=True)
    add_text(slide, "CAPACITÀ  /  CORE RIUTILIZZABILE", 7.03, 2.69, 5.25, 0.31, size=12, text_color=GREEN, bold=True)
    core_points = [
        ("01  Limiti dei framework standard", "Framework troppo pesanti per ambienti industriali e hardware embedded."),
        ("02  Fondamenti matematici", "Studio della matematica del Machine Learning e delle reti neurali con i corsi di Andrew Ng."),
        ("03  Motore proprietario C++", "Libreria AI sviluppata da zero in C++ con Eigen e binding Python tramite pybind11."),
    ]
    capability_points = [
        ("04  Modelli e ottimizzatori", "Regressione, reti Dense e ricorrenti; SGD, Momentum, Adam, RMSprop e Adagrad."),
        ("05  Interfacce e portabilità", "Core riutilizzabile con binding Python; deployment Edge e risorse sul dispositivo da verificare."),
    ]
    for index, (title, body) in enumerate(core_points):
        y = 3.13 + index * 0.84
        add_text(slide, title, 1.03, y, 5.15, 0.28, size=12, bold=True)
        add_text(slide, body, 1.03, y + 0.3, 5.15, 0.52, size=11, text_color=MUTED)
    for index, (title, body) in enumerate(capability_points):
        y = 3.13 + index * 1.13
        add_text(slide, title, 7.03, y, 5.15, 0.28, size=12, bold=True)
        add_text(slide, body, 7.03, y + 0.3, 5.15, 0.65, size=11, text_color=MUTED)
    cards = [
        ("3", "solver lineari", CYAN),
        ("5", "ottimizzatori", GREEN),
        ("3", "tipi ricorrenti", AMBER),
    ]
    for index, (value, label, accent) in enumerate(cards):
        x = 1.08 + index * 3.25
        add_rect(slide, x, 6.12, 2.88, 0.68, PANEL, line=GRID)
        add_text(slide, value, x + 0.18, 6.23, 1.12, 0.43, size=20, bold=True, text_color=accent)
        add_text(slide, label, x + 1.34, 6.29, 1.35, 0.27, size=10, text_color=MUTED)


def slide_methodology(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 2, "ml_core / metodo e qualità",
        "Libreria C++: metodo e qualità ingegneristica",
        "Motore riutilizzabile: interfacce, implementazione nativa e verifica indipendenti dall'applicativo.",
    )
    steps = [
        ("01", "Specifica", "Interfacce, dimensioni e invarianti.", CYAN),
        ("02", "Implementa", "Core C++/Eigen e binding pybind11.", GREEN),
        ("03", "Reti ricorrenti", "Forward e BPTT per RNN, LSTM e GRU.", AMBER),
        ("04", "Verifica", "Test mirati su shape e risultati.", CYAN),
    ]
    for index, (number, title, body, accent) in enumerate(steps):
        y = 1.89 + index * 0.55
        add_rect(slide, 0.78, y, 6.05, 0.47, PANEL, line=GRID)
        add_text(slide, number, 0.98, y + 0.08, 0.4, 0.26, size=11, bold=True, text_color=accent)
        add_text(slide, title, 1.46, y + 0.08, 1.58, 0.26, size=11, bold=True)
        add_text(slide, body, 3.15, y + 0.08, 3.48, 0.29, size=10, text_color=MUTED)

    add_rect(slide, 7.15, 1.89, 5.35, 2.12, PANEL_ALT, line=GRID)
    add_text(slide, "QUALITY GATES  /  ml_core", 7.42, 2.05, 4.4, 0.27, size=11, text_color=CYAN, bold=True)
    add_text(slide, "Seed 42  •  riproducibilità dei modelli neurali", 7.42, 2.48, 4.8, 0.3, size=11, text_color=GREEN, bold=True)
    add_text(slide, "444 dichiarazioni GoogleTest / 60 target CMake\n63 test pytest raccolti per ml_core", 7.42, 2.92, 4.8, 0.64, size=11, text_color=WHITE)
    add_text(slide, "Dichiarazioni sorgente e raccolta del 25/11/2026, non esiti di esecuzione.", 7.42, 3.57, 4.8, 0.3, size=9, text_color=MUTED)
    add_rect(slide, 0.78, 4.28, 11.72, 2.35, PANEL_ALT, line=GRID)
    add_text(slide, "Evoluzione del workflow AI-driven", 1.02, 4.48, 10.9, 0.3, size=12, text_color=CYAN, bold=True)
    add_text(slide, "Fase 1 – Multi-LLM via Chat (DeepSeek + Gemini)", 1.02, 4.9, 5.3, 0.35, size=11, bold=True)
    add_text(slide, "Sviluppo iniziale e verifica via chat. Limiti: visione parziale del progetto e copia/incolla manuale dei file.", 1.02, 5.32, 5.3, 0.9, size=11, text_color=MUTED)
    add_text(slide, "Fase 2 – Agente IDE + revisore (GitHub Copilot + Gemini)", 6.7, 4.9, 5.45, 0.35, size=11, bold=True)
    add_text(slide, "GitHub Copilot in VS Code opera sul codebase; Gemini supporta revisione architetturale, metodo e QA dei benchmark.", 6.7, 5.32, 5.45, 0.9, size=11, text_color=MUTED)


def slide_core_benchmark(prs: Presentation, results: list[dict[str, Any]]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 3, "ml_core / benchmark C++",
        "Benchmark della libreria core ml_core",
        "Intel Core 5 120U · 1 CPU disponibile · misure sintetiche C++ native.",
    )
    headers = ("Modello", "Configurazione", "Fit mediano", "Inferenza mediana", "Velocità")
    table = slide.shapes.add_table(7, 5, Inches(0.75), Inches(1.95), Inches(11.9), Inches(3.35)).table
    for column, width in zip(table.columns, (1.72, 3.0, 1.8, 2.45, 2.93), strict=True):
        column.width = Inches(width)
    for index, values in enumerate((headers, *[
        ("Logistica" if row["model"] == "Logistic" else row["model"], row["configuration"],
         f'{row["fit_ms"]:.2f}'.replace(".", ",") + " ms",
         f'{row["inference_us"]:.4f}'.replace(".", ",") + " µs / elemento",
         f'{row["items_per_second"]:,.0f}'.replace(",", ".") + " elementi/s")
        for row in results
    ])):
        table.rows[index].height = Inches(0.48)
        for column, value in enumerate(values):
            cell = table.cell(index, column)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(PANEL_ALT if index == 0 else PANEL)
            cell.margin_left = Inches(0.07)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.text = value
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if column < 2 else PP_ALIGN.CENTER
            paragraph.font.name = FONT
            paragraph.font.size = Pt(11)
            paragraph.font.bold = index == 0
            paragraph.font.color.rgb = color(CYAN if index == 0 else WHITE)
    add_text(slide, "Fit: mediana di 3 run; Ridge, Logistica e MLP: 8.000 campioni; MLP: 100 epoche. Ricorrenti: 256 sequenze, 20 epoche.",
             0.86, 5.51, 11.65, 0.42, size=10, text_color=MUTED)
    add_text(slide, "Inferenza: mediana di 5 run, tempo ammortizzato per elemento in batch da 1.024 campioni o 64 finestre; non è latenza a batch singolo.",
             0.86, 5.99, 11.65, 0.42, size=10, text_color=MUTED)
    add_text(slide, "Il benchmark misura le prestazioni core dei modelli implementati e non copre ancora tutte le funzionalità della libreria. Fit e inferenza separati; alta variabilità su macchina condivisa.",
             0.86, 6.43, 11.65, 0.45, size=10, text_color=AMBER)


def slide_application_screenshot(prs: Presentation, screenshot_path: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, 4, "", APPLICATION_TITLE)
    picture = slide.shapes.add_picture(str(screenshot_path), 0, 0)
    image_width, image_height = picture.image.size
    scale = min(Inches(11.9) / image_width, Inches(5.55) / image_height)
    picture.width = round(image_width * scale)
    picture.height = round(image_height * scale)
    picture.left = (prs.slide_width - picture.width) // 2
    picture.top = Inches(1.43) + (Inches(5.55) - picture.height) // 2


def slide_application(prs: Presentation, metrics: dict[str, Any]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    gru = metrics["models"]["gru_cpp"]
    sample_count = int(metrics["test_samples"])
    inference = float(gru["inference_seconds"])
    per_sample_us = inference / sample_count * 1_000_000
    add_header(
        slide, 5, "applicativo / Python + web",
        APPLICATION_TITLE,
        "Il layer applicativo prepara i dati e usa ml_core tramite pybind11; il motore C++ resta riutilizzabile.",
    )
    blocks = [
        ("NASA IMS", "Dati cuscinetti\nfeature e finestre", CYAN),
        ("Auto-Zero", "Baseline per bearing\nscaling delle feature", GREEN),
        ("Inferenza", "Python → pybind11\n→ modelli ml_core C++", AMBER),
        ("Cruscotto web", "Backend Python\nreplay JS / Plotly", CYAN),
    ]
    x_positions = [0.72, 3.83, 6.94, 10.05]
    for index, (title, body, accent) in enumerate(blocks):
        x = x_positions[index]
        add_rect(slide, x, 2.18, 2.55, 1.55, PANEL, line=GRID)
        add_rect(slide, x, 2.18, 2.55, 0.08, accent, rounded=False)
        add_text(slide, title, x + 0.18, 2.48, 2.2, 0.42, size=15, bold=True)
        add_text(slide, body, x + 0.18, 3.0, 2.15, 0.55, size=11, text_color=MUTED)
        if index < len(blocks) - 1:
            add_line(slide, x + 2.62, 2.95, x_positions[index + 1] - 0.12, 2.95, CYAN, 1.6)
    add_card(
        slide, 0.72, 4.25, 5.75, 1.6, GREEN, "Confine tra i livelli",
        "Python gestisce dati IMS, preprocessing e cruscotto. ml_core fornisce i modelli C++ tramite binding, senza dipendere dal dominio applicativo.",
        title_size=15, body_size=11,
    )
    add_rect(slide, 6.8, 4.25, 5.8, 1.6, PANEL_ALT, line=GRID)
    add_text(slide, "GRU / INFERENZA MISURATA", 7.12, 4.52, 4.8, 0.3, size=10, text_color=CYAN, bold=True)
    add_text(slide, f"{inference:.3f} s", 7.12, 4.9, 2.1, 0.55, size=26, text_color=GREEN, bold=True)
    add_text(slide, f"{sample_count:,} windows  |  {per_sample_us:.1f} us/window", 9.35, 5.05, 2.75, 0.34, size=11, text_color=MUTED)
    add_text(slide, "Misura wall-clock da predict_proba sul benchmark; non include preprocessing.", 7.12, 5.58, 4.9, 0.22, size=9, text_color=MUTED)


def slide_iteration(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 6, "scientific iteration",
        "Iterazione scientifica & problemi reali",
        "Tre correzioni metodologiche che hanno trasformato un prototipo in una pipeline verificabile.",
    )
    items = [
        ("01", "Sbilanciamento / FEMTO", "Il forte sbilanciamento dei target ha guidato la scelta del run-to-failure NASA IMS e di una proxy label sull'ultimo 25% della vita.", CYAN),
        ("02", "Variabilità sensori", "Auto-zero per bearing: sottrazione della media delle prime 30 acquisizioni. StandardScaler fit solo su 1st_test + 3rd_test; poi transform sul 2nd_test.", GREEN),
        ("03", "Score e allarmi", "Uso di predict_proba, MLP con L2 e filtro causale: rolling mean 5 + persistenza di 3 step consecutivi sopra soglia.", AMBER),
    ]
    for index, (number, title, body, accent) in enumerate(items):
        x = 0.72 + index * 4.17
        add_rect(slide, x, 2.1, 3.82, 3.65, PANEL, line=GRID)
        add_text(slide, number, x + 0.25, 2.38, 0.8, 0.55, size=27, text_color=accent, bold=True)
        add_text(slide, title, x + 0.25, 3.13, 3.25, 0.68, size=17, bold=True)
        add_text(slide, body, x + 0.25, 3.98, 3.25, 1.45, size=12, text_color=MUTED)
    add_text(slide, "INVARIANTE: finestre GRU costruite separatamente per bearing; nessun campione attraversa il confine tra due cuscinetti.", 0.9, 6.15, 11.5, 0.42, size=12, text_color=CYAN, bold=True)


def slide_benchmark(prs: Presentation, metrics: dict[str, Any]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    models = metrics["models"]
    sample_count = int(metrics["test_samples"])
    full_sample_count = int(metrics["full_test_samples"])
    bearing_count = 4
    per_bearing = sample_count // bearing_count
    first_file = int(metrics["gru_network"]["time_steps"])
    last_file = full_sample_count // bearing_count
    add_header(
        slide, 7, "benchmark / NASA IMS",
        "Logistica vs MLP vs GRU C++",
        f"2nd_test, {sample_count:,} finestre allineate ({per_bearing} per bearing; file {first_file}-{last_file}). "
        "Metriche dopo rolling mean 5 e persistenza 3.",
    )
    headers = ("Modello", "Precision", "Recall", "F1", "Fit", "Inferenza", "Primo allarme B1", "FP sani")
    rows = []
    for model_id in MODEL_ORDER:
        model = models[model_id]
        alarm = model.get("first_alarm_bearing_1_file")
        rows.append(
            (
                MODEL_SHORT_NAMES[model_id],
                format_percent(float(model["precision"])),
                format_percent(float(model["recall"])),
                format_percent(float(model["f1_score"])),
                f'{float(model["training_seconds"]):.2f} s',
                format_seconds(float(model["inference_seconds"])),
                str(alarm) if alarm is not None else "n/a",
                str(int(model["healthy_false_alarm_episodes"])),
            )
        )
    table_shape = slide.shapes.add_table(4, len(headers), Inches(0.65), Inches(2.0), Inches(12.05), Inches(2.35))
    table = table_shape.table
    widths = (2.12, 1.3, 1.25, 1.05, 1.15, 1.38, 1.85, 1.55)
    for column, width in zip(table.columns, widths, strict=True):
        column.width = Inches(width)
    for row_index, values in enumerate((headers, *rows)):
        is_active_model = row_index > 0 and MODEL_ORDER[row_index - 1] == metrics["active_model"]
        table.rows[row_index].height = Inches(0.52 if row_index == 0 else 0.61)
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color(
                PANEL_ALT if row_index == 0 else ("102A2A" if is_active_model else PANEL)
            )
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.text = value
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.LEFT if column_index == 0 else PP_ALIGN.CENTER
            paragraph.font.name = FONT
            paragraph.font.size = Pt(10 if row_index == 0 else 11)
            paragraph.font.bold = row_index == 0 or is_active_model
            paragraph.font.color.rgb = color(CYAN if row_index == 0 else (GREEN if is_active_model else WHITE))

    active_model = metrics["active_model"]
    active = models[active_model]
    healthy_counts = active["false_alarm_episodes_by_bearing"]
    healthy_detail = "  |  ".join(
        f"B{index}: {int(healthy_counts.get(f'2nd_test_bearing_{index}', 0))}"
        for index in (2, 3, 4)
    )
    add_rect(slide, 0.72, 4.8, 11.95, 1.15, PANEL_ALT, line=GREEN)
    add_text(slide, f"MODELLO PREDEFINITO: {MODEL_SHORT_NAMES[active_model]}", 1.0, 5.02, 4.8, 0.32, size=15, text_color=GREEN, bold=True)
    add_text(
        slide,
        f'F1 {format_percent(float(active["f1_score"]))}  |  Recall {format_percent(float(active["recall"]))}  |  '
        f'primo allarme file {active["first_alarm_bearing_1_file"]}  |  falsi allarmi sani: {healthy_detail}',
        1.0, 5.47, 11.2, 0.28, size=11, text_color=WHITE,
    )
    add_text(
        slide,
        "Nota: tuning soglia eseguito sui label del 2nd_test; questo confronto è esplorativo, non una verifica cieca/indipendente. Target proxy Bearing 1: file 739.",
        0.83, 6.27, 11.7, 0.48, size=9, text_color=AMBER,
    )


def slide_web(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 8, "demo web / local replay",
        "Cruscotto web leggero & latenza",
        "Backend Python e asset HTML/JS/Plotly serviti localmente; il replay usa predizioni già calcolate.",
    )
    pipeline = [
        ("Backend", "Python\nvalida CSV/JSON", CYAN),
        ("API replay", "payload locale\nuna sola lettura", GREEN),
        ("Browser", "JS + Plotly\nsenza reload", AMBER),
    ]
    for index, (title, body, accent) in enumerate(pipeline):
        x = 0.72 + index * 1.72
        add_rect(slide, x, 2.18, 1.48, 1.15, PANEL, line=GRID)
        add_text(slide, title, x + 0.12, 2.39, 1.25, 0.32, size=13, text_color=accent, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, body, x + 0.12, 2.82, 1.25, 0.38, size=9, text_color=MUTED, align=PP_ALIGN.CENTER)
        if index < 2:
            add_line(slide, x + 1.5, 2.76, x + 1.68, 2.76, CYAN, 1.5)

    add_rect(slide, 0.72, 3.75, 4.96, 1.55, PANEL_ALT, line=GRID)
    add_text(slide, "CLIENT-SIDE REPLAY", 0.98, 4.0, 4.3, 0.3, size=11, text_color=CYAN, bold=True)
    add_text(slide, "100 ms/tick (~10 acquisizioni/s)\nPlotly aggiornato con throttle ~250 ms\n60 FPS non misurati: target da profilare", 0.98, 4.42, 4.32, 0.75, size=12, text_color=WHITE)

    add_rect(slide, 6.15, 1.92, 6.5, 3.55, "101722", line=CYAN)
    add_rect(slide, 6.4, 2.18, 6.0, 3.04, "151A22", line=GRID)
    add_line(slide, 6.64, 2.57, 12.15, 2.57, GRID, 0.8)
    add_text(slide, "STONECUT  /  LIVE DASHBOARD SCREENSHOT", 6.58, 2.3, 5.7, 0.22, size=9, text_color=CYAN, bold=True)
    add_text(slide, "PLACEHOLDER\nInserire qui lo screenshot del cruscotto dal vivo", 6.82, 3.28, 5.15, 0.9, size=18, text_color=MUTED, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)
    add_text(slide, "Dati storici NASA IMS  |  selettore Logistica / MLP / GRU", 6.6, 5.66, 5.9, 0.3, size=10, text_color=MUTED)


def slide_limits(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 9, "methodology / limitations",
        "Onestà metodologica & limiti noti",
        "I risultati dimostrano una pipeline tecnica, non una prestazione industriale già validata.",
    )
    cautions = [
        ("Soglia esplorativa", "La soglia è stata selezionata usando i label del 2nd_test: non chiamarla test cieco o risultato indipendente.", AMBER),
        ("Label proxy", "L'ultimo 25% della sequenza rappresenta una proxy temporale; non è una RUL calibrata né il timestamp osservato del guasto.", CYAN),
        ("Segnali mancanti", "Per macchine reali servono carico, corrente, temperatura e contesto operativo (es. parametri del taglio pietra).", GREEN),
        ("Generalizzazione", "Servono acquisizioni di stabilimento, split per macchina/bearing e verifica della calibrazione prima di decisioni operative.", AMBER),
    ]
    for index, (title, body, accent) in enumerate(cautions):
        x = 0.75 + (index % 2) * 6.08
        y = 2.03 + (index // 2) * 1.72
        add_card(slide, x, y, 5.65, 1.42, accent, title, body, title_size=15, body_size=11)
    add_rect(slide, 0.75, 5.75, 11.75, 0.7, PANEL_ALT, line=AMBER)
    add_text(slide, "Nessun allarme del prototipo autorizza interventi automatici: decisione e verifica restano umane.", 1.0, 5.95, 11.1, 0.3, size=12, text_color=AMBER, bold=True)


def slide_next_steps(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide, 10, "roadmap / Edge e produzione",
        "Roadmap: dall'Edge alla produzione",
        "Passaggi necessari prima di un impiego industriale affidabile.",
    )
    add_card(slide, 0.75, 2.25, 3.8, 2.3, CYAN, "Validazione indipendente",
             "Raccogliere dati di stabilimento e valutare su macchine e cuscinetti separati.", title_size=15, body_size=12)
    add_card(slide, 4.76, 2.25, 3.8, 2.3, GREEN, "Profilazione Edge",
             "Misurare latenza e risorse sul dispositivo target e verificare il packaging.", title_size=15, body_size=12)
    add_card(slide, 8.77, 2.25, 3.8, 2.3, AMBER, "Contesto operativo",
             "Integrare sensori e verificare calibrazione, soglie e allarmi.", title_size=15, body_size=12)
    add_rect(slide, 0.75, 5.15, 11.82, 0.95, PANEL_ALT, line=CYAN)
    add_text(slide, "Verso la produzione: monitoraggio, manutenzione del modello e verifica umana delle decisioni.",
             1.02, 5.43, 11.2, 0.45, size=13, text_color=WHITE, bold=True)


def build_presentation(metrics: dict[str, Any], core_results: list[dict[str, Any]], output_path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(WIDTH)
    prs.slide_height = Inches(HEIGHT)
    prs.core_properties.title = "ml_core C++ e Manutenzione Predittiva - Deda Speed"
    prs.core_properties.subject = "C++ BPTT, NASA IMS benchmark and local web dashboard"
    prs.core_properties.author = "Maurizio Penna"

    slide_title(prs, metrics)
    slide_methodology(prs)
    slide_core_benchmark(prs, core_results)
    slide_application_screenshot(prs, SCREENSHOT_PATH)
    slide_application(prs, metrics)
    slide_iteration(prs)
    slide_benchmark(prs, metrics)
    slide_web(prs)
    slide_limits(prs)
    slide_next_steps(prs)

    if len(prs.slides) != SLIDE_COUNT:
        raise RuntimeError(f"Expected {SLIDE_COUNT} slides, generated {len(prs.slides)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output_path)


def main() -> None:
    metrics = load_metrics(METRICS_PATH)
    core_results = load_core_benchmarks(CORE_FIT_PATH, CORE_INFERENCE_PATH)
    build_presentation(metrics, core_results, OUTPUT_PATH)
    print(f"Generated {OUTPUT_PATH} ({len(core_results)} core models, {SLIDE_COUNT} slides)")


if __name__ == "__main__":
    main()
