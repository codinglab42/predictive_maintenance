#!/usr/bin/env python3
"""Run an interactive historical FEMTO bearing dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.dashboard_data import (
    BearingSeries,
    load_dashboard_data,
    position_at_progress,
    status_at_position,
)


SHARED_ROOT = Path("/mnt/shared/predictive-maintenance")
DEFAULT_FEATURES = SHARED_ROOT / "processed" / "femto_features.npz"
DEFAULT_PREDICTIONS = (
    SHARED_ROOT
    / "processed"
    / "loco_rms_comparison"
    / "femto_loco_h20_alarm_predictions.csv"
)

STATUS_STYLE = {
    "green": ("VERDE", "#157347", "#e7f5ec", "Nessun segnale sopra la soglia sperimentale"),
    "yellow": ("GIALLO", "#9a6700", "#fff4ce", "Segnale sopra soglia: in attesa di conferma"),
    "red": ("ROSSO", "#b42318", "#ffebe9", "Allarme emesso: ispezione da valutare"),
}


st.set_page_config(
    page_title="Stato cuscinetti | FEMTO",
    page_icon="🔧",
    layout="wide",
)


@st.cache_data(show_spinner="Caricamento delle sequenze e delle predizioni LOCO...")
def load_cached_data(features_path: str, predictions_path: str) -> dict[str, BearingSeries]:
    return load_dashboard_data(Path(features_path), Path(predictions_path))


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
        .eyebrow {color:#687386; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;}
        .notice {background:#fff8e6; border:1px solid #f1d58a; border-radius:12px; padding:14px 18px; color:#614b16; line-height:1.5;}
        div[data-testid="stButton"] > button {height:148px; text-align:left; white-space:pre-wrap; line-height:1.5;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Manutenzione predittiva · FEMTO/PRONOSTIA</div>', unsafe_allow_html=True)
    st.title("Cruscotto stato cuscinetti")
    st.caption("Replay interattivo di sequenze storiche — non è un monitoraggio live.")

    with st.sidebar:
        st.header("Sorgenti dati")
        features_path = st.text_input("Feature FEMTO (.npz)", str(DEFAULT_FEATURES))
        predictions_path = st.text_input("Predizioni LOCO (.csv)", str(DEFAULT_PREDICTIONS))
        st.markdown(
            "**Dataset visualizzato:** NASA FEMTO/PRONOSTIA  \n"
            "**Grandezza misurata:** accelerazione delle vibrazioni su asse "
            "orizzontale e verticale  \n"
            "**Unità:** non specificata nei file utilizzati  \n"
            "**Grafico:** RMS dell'accelerazione orizzontale"
        )
        st.divider()
        progress = st.slider(
            "Posizione nella sequenza storica (%)",
            min_value=0,
            max_value=100,
            value=75,
            help="Allinea i cuscinetti alla stessa percentuale della sequenza; non indica tempo reale.",
        )
        st.caption("Gli intervalli FEMTO non sono convertiti in minuti: la cadenza non è verificata.")

    try:
        series_by_bearing = load_cached_data(features_path, predictions_path)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Impossibile caricare i dati del cruscotto: {error}")
        st.info(
            "Verifica che il file feature FEMTO e il CSV LOCO siano presenti. "
            "Puoi cambiare entrambi i percorsi nella barra laterale."
        )
        st.stop()

    snapshots = {
        bearing_id: position_at_progress(series, progress)
        for bearing_id, series in series_by_bearing.items()
    }
    statuses = {
        bearing_id: status_at_position(series_by_bearing[bearing_id], position)
        for bearing_id, position in snapshots.items()
    }
    bearing_ids = list(series_by_bearing)
    button_styles = []
    for bearing_id, status in statuses.items():
        _, foreground, background, _ = STATUS_STYLE[status]
        button_styles.append(
            f".st-key-open_{bearing_id} button {{"
            f"background-color: {background} !important;"
            f"border-color: {foreground} !important;"
            "color: #202124 !important;"
            "}"
            f".st-key-open_{bearing_id} button:hover {{filter: brightness(0.96);}}"
        )
    st.markdown(f"<style>{''.join(button_styles)}</style>", unsafe_allow_html=True)

    if st.session_state.get("selected_bearing") not in series_by_bearing:
        st.session_state["selected_bearing"] = bearing_ids[0]

    def open_bearing_detail(bearing_id: str) -> None:
        st.session_state["selected_bearing"] = bearing_id
        st.session_state["dashboard_tab"] = "Dettaglio"

    overview_tab, detail_tab = st.tabs(
        ["Quadro d'insieme", "Dettaglio"],
        key="dashboard_tab",
        on_change="rerun",
    )

    with overview_tab:
        st.markdown(
            """
            <div class="notice">
            <strong>Uso dimostrativo.</strong> I colori derivano dalla policy sperimentale
            LOCO h20 e da predizioni fuori-fold su sei cuscinetti da banco. Verde = score
            sotto soglia; giallo = soglia superata ma non confermata; rosso = notifica
            emessa dalla policy. Il rosso richiede verifica del manutentore, non comanda
            una sostituzione. Le prestazioni non sono validate per un impianto reale.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        counts = {
            status: sum(value == status for value in statuses.values())
            for status in STATUS_STYLE
        }
        metric_columns = st.columns(4)
        metric_columns[0].metric("Cuscinetti nel replay", len(series_by_bearing))
        metric_columns[1].metric("Verde", counts["green"])
        metric_columns[2].metric("Giallo", counts["yellow"])
        metric_columns[3].metric("Rosso", counts["red"])
        st.subheader(f"Sequenza storica · posizione {progress}% · seleziona un cuscinetto")

        status_icons = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
        for row_start in range(0, len(bearing_ids), 3):
            columns = st.columns(3)
            for column, bearing_id in zip(
                columns,
                bearing_ids[row_start : row_start + 3],
                strict=False,
            ):
                series = series_by_bearing[bearing_id]
                position = snapshots[bearing_id]
                status = statuses[bearing_id]
                label, _, _, message = STATUS_STYLE[status]
                probability = series.probabilities[position]
                column.button(
                    (
                        f"**{bearing_id}**\n\n"
                        f"{status_icons[status]} **{label}** · Score h20 {probability:.1%}\n\n"
                        f"{message}\n\n"
                        f"Acquisizione {series.sample_indices[position]:,} · Apri dettaglio →"
                    ),
                    key=f"open_{bearing_id}",
                    on_click=open_bearing_detail,
                    args=(bearing_id,),
                    width="stretch",
                )

    with detail_tab:
        selected_bearing = st.session_state["selected_bearing"]
        selected = series_by_bearing[selected_bearing]
        position = snapshots[selected_bearing]
        status = statuses[selected_bearing]
        label, foreground, _, message = STATUS_STYLE[status]

        st.subheader(selected_bearing)
        details = st.columns(4)
        details[0].metric("Stato del cuscinetto", label)
        details[1].metric("Score modello h20", f"{selected.probabilities[position]:.1%}")
        details[2].metric("Soglia policy LOCO", f"{selected.thresholds[position]:.0%}")
        details[3].metric("Acquisizione", f"{selected.sample_indices[position]:,}")
        st.caption(
            f"Condizione operativa {selected.condition_id} · {message}. "
            "Lo score non è una misura diretta dello stato fisico né una probabilità calibrata."
        )

        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(
            go.Scatter(
                x=selected.sample_indices,
                y=selected.horizontal_rms,
                mode="lines",
                name="RMS orizzontale",
                line={"color": "#3b82f6", "width": 1.8},
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=selected.sample_indices,
                y=selected.probabilities,
                mode="lines",
                name="Score modello h20",
                line={"color": "#8b5cf6", "width": 1.7},
            ),
            secondary_y=True,
        )
        figure.add_trace(
            go.Scatter(
                x=selected.sample_indices,
                y=selected.thresholds,
                mode="lines",
                name="Soglia policy",
                line={"color": "#d97706", "width": 1.3, "dash": "dash"},
            ),
            secondary_y=True,
        )
        figure.add_vline(
            x=int(selected.sample_indices[position]),
            line_width=2,
            line_dash="dot",
            line_color=foreground,
        )
        figure.update_layout(
            height=430,
            margin={"l": 8, "r": 8, "t": 28, "b": 8},
            legend={"orientation": "h", "y": 1.12, "x": 0},
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        figure.update_xaxes(title_text="Indice acquisizione", showgrid=True, gridcolor="#edf0f4")
        figure.update_yaxes(
            title_text="RMS orizzontale",
            secondary_y=False,
            showgrid=True,
            gridcolor="#edf0f4",
        )
        figure.update_yaxes(
            title_text="Score / soglia h20",
            secondary_y=True,
            range=[0, 1],
            showgrid=False,
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "La linea verticale indica la posizione selezionata nel replay. "
            "L'allarme rosso resta latched nella sequenza fino a una presa in carico "
            "manuale, non rappresentata da questi dati storici."
        )


if __name__ == "__main__":
    main()
