#!/usr/bin/env python3
"""StoneCut industrial predictive-maintenance dashboard for the IMS replay."""

from __future__ import annotations

from datetime import datetime
from html import escape
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    layout="wide",
    page_title="StoneCut PdM - Neural Monitoring",
)

from backend import (  # noqa: E402
    BEARING_ORDER,
    COMPONENT_NAMES,
    BearingReading,
    DashboardData,
    MachineStatus,
    build_maintenance_reports,
    classify_probability,
    load_dashboard_data,
    readings_at_position,
)


LANGUAGES = ("Italiano", "English", "中文")
TEXT = {
    "Italiano": {
        "language": "Lingua",
        "title": "StoneCut PdM",
        "subtitle": "Monitoraggio predittivo per macchine da taglio della pietra",
        "replay": "Replay storico NASA IMS · 2nd_test",
        "warning": "Soglia selezionata sulle etichette di 2nd_test: analisi esplorativa, non test indipendente. Le probabilità Sigmoide non sono calibrate; nessuna sostituzione automatica.",
        "threshold": "Soglia di allarme sostituzione",
        "threshold_help": "Preselezionata dal tuning esplorativo su 2nd_test; modificabile per la demo.",
        "play": "▶ Play",
        "pause": "⏸ Pausa",
        "file": "Acquisizione",
        "file_help": "La sequenza contiene 984 file ordinati cronologicamente.",
        "machine_status": "Stato macchina",
        "global_health": "Health Index globale",
        "bearings": "Cuscinetti monitorati",
        "engine": "Motore predittivo",
        "operational": "OPERATIVO",
        "prealarm": "PRE-ALLARME",
        "critical": "SOSTITUZIONE CONSIGLIATA",
        "status_short": {
            "operational": "OPERATIVO",
            "prealarm": "PRE-ALLARME",
            "critical": "SOSTITUZIONE",
        },
        "active": "Attivi",
        "critical_count": "critici",
        "engine_badge": "C++ Neural Engine (Active)",
        "bearing_status": "Stato cuscinetti macchina",
        "health_score": "Health Score",
        "rms_current": "RMS attuale",
        "kurtosis_current": "Curtosi",
        "probability": "Probabilità di guasto",
        "chart_title": "Andamento storico della probabilità di guasto",
        "maintenance_threshold": "Soglia Intervento Manutenzione",
        "positive_tail": "Inizio coda etichettata positiva (proxy)",
        "early_alarm": "Primo allarme",
        "lead_proxy": "anticipo rispetto alla coda target",
        "file_label": "File",
        "advanced": "Diagnostica Avanzata & Confronto Modelli (Ingegneria)",
        "comparison": "Rete neurale C++ vs benchmark logistico",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "model": "Modello",
        "technical_rms": "Andamento RMS",
        "technical_kurtosis": "Andamento curtosi",
        "order_title": "Report di manutenzione",
        "order_ready": "Rilevati cuscinetti sopra soglia: genera un ordine per la verifica umana.",
        "order_none": "Nessun cuscinetto sopra soglia nell'acquisizione selezionata.",
        "report_pause": "Metti in pausa il replay per generare il report dell'acquisizione corrente.",
        "download_csv": "Genera Ordine di Manutenzione (CSV)",
        "download_txt": "Genera Ordine di Manutenzione (TXT)",
        "load_error": "Impossibile caricare dati e metriche IMS",
        "sequence_error": "Il replay richiede sequenze allineate per tutti e quattro i cuscinetti.",
        "timestamp": "Timestamp",
        "source": "Modello: MLP C++ via pybind11 · probabilità pre-calcolate sul dataset IMS",
        "label_caveat": "La coda positiva del 25% è un'etichetta proxy del benchmark, non l'istante osservato del guasto.",
        "machine": "Macchina",
        "component_names": {
            "2nd_test_bearing_1": "Mandrino Lama Principale (Ant.)",
            "2nd_test_bearing_2": "Mandrino Lama Principale (Post.)",
            "2nd_test_bearing_3": "Volano Trasmissione",
            "2nd_test_bearing_4": "Rinvio Motoriduttore",
        },
        "model_names": {
            "neural_network_cpp": "Rete neurale C++",
            "logistic_regression": "Regressione logistica",
        },
        "chart_axis_time": "Timestamp acquisizione",
        "chart_axis_probability": "Probabilità di guasto",
        "chart_axis_metric": "Punteggio",
    },
    "English": {
        "language": "Language",
        "title": "StoneCut PdM",
        "subtitle": "Predictive monitoring for stone-cutting machines",
        "replay": "NASA IMS historical replay · 2nd_test",
        "warning": "The threshold was selected using 2nd_test labels: exploratory analysis, not an independent test. Sigmoid scores are uncalibrated; no automatic replacement command.",
        "threshold": "Replacement alarm threshold",
        "threshold_help": "Preselected by exploratory tuning on 2nd_test; adjustable for the demo.",
        "play": "▶ Play",
        "pause": "⏸ Pause",
        "file": "Acquisition",
        "file_help": "The sequence contains 984 files in chronological order.",
        "machine_status": "Machine status",
        "global_health": "Global Health Index",
        "bearings": "Bearings monitored",
        "engine": "Predictive engine",
        "operational": "OPERATIONAL",
        "prealarm": "PRE-ALARM",
        "critical": "REPLACEMENT RECOMMENDED",
        "status_short": {
            "operational": "OPERATIONAL",
            "prealarm": "PRE-ALARM",
            "critical": "REPLACEMENT",
        },
        "active": "Active",
        "critical_count": "critical",
        "engine_badge": "C++ Neural Engine (Active)",
        "bearing_status": "Machine bearing status",
        "health_score": "Health Score",
        "rms_current": "Current RMS",
        "kurtosis_current": "Kurtosis",
        "probability": "Failure probability",
        "chart_title": "Historical failure probability",
        "maintenance_threshold": "Maintenance intervention threshold",
        "positive_tail": "Start of positive-label tail (proxy)",
        "early_alarm": "First alarm",
        "lead_proxy": "lead vs target tail",
        "file_label": "File",
        "advanced": "Advanced Diagnostics & Model Comparison (Engineering)",
        "comparison": "C++ neural network vs logistic benchmark",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-Score",
        "model": "Model",
        "technical_rms": "RMS trend",
        "technical_kurtosis": "Kurtosis trend",
        "order_title": "Maintenance report",
        "order_ready": "Bearings above threshold detected: generate an order for human review.",
        "order_none": "No bearing is above the threshold at the selected acquisition.",
        "report_pause": "Pause the replay to generate a report for the current acquisition.",
        "download_csv": "Generate Maintenance Order (CSV)",
        "download_txt": "Generate Maintenance Order (TXT)",
        "load_error": "Unable to load IMS data and metrics",
        "sequence_error": "Replay requires aligned sequences for all four bearings.",
        "timestamp": "Timestamp",
        "source": "Model: C++ MLP via pybind11 · probabilities precomputed on the IMS dataset",
        "label_caveat": "The final 25% positive tail is a benchmark proxy label, not the observed failure time.",
        "machine": "Machine",
        "component_names": COMPONENT_NAMES,
        "model_names": {
            "neural_network_cpp": "C++ neural network",
            "logistic_regression": "Logistic regression",
        },
        "chart_axis_time": "Acquisition timestamp",
        "chart_axis_probability": "Failure probability",
        "chart_axis_metric": "Score",
    },
    "中文": {
        "language": "语言",
        "title": "StoneCut PdM",
        "subtitle": "石材切割设备预测性维护监控",
        "replay": "NASA IMS 历史回放 · 2nd_test",
        "warning": "阈值使用 2nd_test 标签选择：仅为探索性分析，并非独立测试。Sigmoid 分数未经校准；不会自动发出更换指令。",
        "threshold": "更换报警阈值",
        "threshold_help": "根据 2nd_test 探索性调优预选；可在演示中调整。",
        "play": "▶ 播放",
        "pause": "⏸ 暂停",
        "file": "采集文件",
        "file_help": "序列包含按时间排序的 984 个文件。",
        "machine_status": "设备状态",
        "global_health": "全局健康指数",
        "bearings": "监测轴承",
        "engine": "预测引擎",
        "operational": "运行正常",
        "prealarm": "预警",
        "critical": "建议更换",
        "status_short": {
            "operational": "运行正常",
            "prealarm": "预警",
            "critical": "建议更换",
        },
        "active": "运行中",
        "critical_count": "临界",
        "engine_badge": "C++ Neural Engine (Active)",
        "bearing_status": "设备轴承状态",
        "health_score": "健康评分",
        "rms_current": "当前 RMS",
        "kurtosis_current": "峭度",
        "probability": "故障概率",
        "chart_title": "历史故障概率趋势",
        "maintenance_threshold": "维护干预阈值",
        "positive_tail": "正标签区间起点（代理标签）",
        "early_alarm": "首次报警",
        "lead_proxy": "相对目标区间的提前量",
        "file_label": "文件",
        "advanced": "高级诊断与模型对比（工程）",
        "comparison": "C++ 神经网络与逻辑回归基准",
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1 分数",
        "model": "模型",
        "technical_rms": "RMS 趋势",
        "technical_kurtosis": "峭度趋势",
        "order_title": "维护报告",
        "order_ready": "检测到超过阈值的轴承：生成报告并交由人工复核。",
        "order_none": "当前采集文件没有轴承超过阈值。",
        "report_pause": "请暂停回放后生成当前采集文件的报告。",
        "download_csv": "生成维护工单（CSV）",
        "download_txt": "生成维护工单（TXT）",
        "load_error": "无法加载 IMS 数据和指标",
        "sequence_error": "回放需要四个轴承的时间序列对齐。",
        "timestamp": "时间戳",
        "source": "模型：通过 pybind11 调用 C++ MLP · IMS 数据集预计算概率",
        "label_caveat": "最后 25% 的正标签是基准代理标签，并非实际观测到的故障时刻。",
        "machine": "设备",
        "component_names": {
            "2nd_test_bearing_1": "主刀片主轴（前）",
            "2nd_test_bearing_2": "主刀片主轴（后）",
            "2nd_test_bearing_3": "传动飞轮",
            "2nd_test_bearing_4": "减速电机传动轴",
        },
        "model_names": {
            "neural_network_cpp": "C++ 神经网络",
            "logistic_regression": "逻辑回归",
        },
        "chart_axis_time": "采集时间",
        "chart_axis_probability": "故障概率",
        "chart_axis_metric": "评分",
    },
}

STATUS_STYLE: dict[MachineStatus, tuple[str, str, str]] = {
    "operational": ("#35c98a", "operational", "🟢"),
    "prealarm": ("#f0b547", "prealarm", "🟡"),
    "critical": ("#f06464", "critical", "🔴"),
}
TRACE_COLORS = ("#4da3ff", "#b58cff", "#36c6b5", "#ff9f68")


def _component_names(language: str) -> dict[str, str]:
    return TEXT[language]["component_names"]


def _alarm_label(status: MachineStatus, language: str) -> str:
    return TEXT[language][status]


def _replay_status_callback() -> None:
    st.session_state["is_playing"] = False
    st.session_state["replay_index"] = st.session_state["replay_slider"]


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1680px; padding-top: 1.35rem; padding-bottom: 2rem;}
        .stApp {background: #0E1117;}
        [data-testid="stSidebar"] {background: #151a22; border-right: 1px solid #29313d;}
        [data-testid="stMetricValue"] {font-size:1.15rem !important;}
        [data-testid="stMetricValue"] > div {white-space:normal !important; overflow-wrap:anywhere; line-height:1.12;}
        .eyebrow {color:#8ea0b8; font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase;}
        .pdm-card {min-height: 226px; background:#151a22; border:1px solid #344050; border-left:4px solid #35c98a;
          border-radius:12px; padding:16px 16px 12px; box-sizing:border-box;}
        .pdm-card.prealarm {border-left-color:#f0b547;}
        .pdm-card.critical {border-left-color:#f06464;}
        .pdm-card-head {min-height:66px; display:flex; flex-direction:column; justify-content:space-between; gap:7px;
          color:#e7edf5; font-weight:700; font-size:.9rem; line-height:1.2;}
        .pdm-chip {font-size:.62rem; line-height:1.2; letter-spacing:.04em; color:#35c98a; overflow-wrap:anywhere;}
        .prealarm .pdm-chip {color:#f0b547;}
        .critical .pdm-chip {color:#f06464;}
        .pdm-health {display:flex; align-items:baseline; gap:7px; margin-top:14px; color:#f7f9fc;}
        .pdm-health strong {font-size:1.75rem; line-height:1;}
        .pdm-health span {font-size:.72rem; color:#97a6ba;}
        .pdm-track {height:7px; margin:9px 0 14px; border-radius:8px; overflow:hidden; background:#303845;}
        .pdm-track span {display:block; height:100%; border-radius:8px; background:#35c98a;}
        .prealarm .pdm-track span {background:#f0b547;}
        .critical .pdm-track span {background:#f06464;}
        .pdm-kv {display:flex; justify-content:space-between; color:#9eacc0; font-size:.75rem;
          padding:5px 0; border-top:1px solid #29313d;}
        .pdm-kv strong {color:#e7edf5; font-weight:600;}
        .status-banner {padding:12px 16px; border-radius:10px; background:#1b222d;
          border:1px solid #303b49; margin:4px 0 16px; color:#aab8ca;}
        .status-banner strong {color:#fff;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_text(status: MachineStatus, language: str) -> str:
    _, text_key, _ = STATUS_STYLE[status]
    return _alarm_label(text_key, language)


def _render_bearing_card(
    column: st.delta_generator.DeltaGenerator,
    reading: BearingReading,
    language: str,
) -> None:
    color, _, icon = STATUS_STYLE[reading.status]
    component = escape(_component_names(language)[reading.bearing_id])
    status_text = escape(_status_text(reading.status, language))
    health = min(100.0, max(0.0, reading.health_index))
    column.markdown(
        f"""
        <div class="pdm-card {reading.status}">
          <div class="pdm-card-head">
            <span>{component}</span>
            <span class="pdm-chip">{icon} {status_text}</span>
          </div>
          <div class="pdm-health"><strong>{health:.1f}%</strong>
            <span>{escape(TEXT[language]["health_score"])}</span></div>
          <div class="pdm-track"><span style="width:{health:.2f}%; background:{color};"></span></div>
          <div class="pdm-kv"><span>{escape(TEXT[language]["rms_current"])}</span>
            <strong>{reading.rms:.4f}</strong></div>
          <div class="pdm-kv"><span>{escape(TEXT[language]["probability"])}</span>
            <strong>{reading.probability:.1%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _probability_figure(
    data: DashboardData,
    position: int,
    threshold: float,
    language: str,
) -> go.Figure:
    text = TEXT[language]
    names = _component_names(language)
    figure = go.Figure()
    for bearing_id, color in zip(BEARING_ORDER, TRACE_COLORS, strict=True):
        series = data.series_by_bearing[bearing_id]
        end = position + 1
        figure.add_trace(
            go.Scatter(
                x=series.timestamps[:end],
                y=series.probabilities[:end],
                mode="lines",
                name=names[bearing_id],
                line={"color": color, "width": 2.2},
                hovertemplate="%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            )
        )
        alarm_positions = np.flatnonzero(series.probabilities[:end] > threshold)
        if alarm_positions.size:
            first_alarm = int(alarm_positions[0])
            figure.add_trace(
                go.Scatter(
                    x=[series.timestamps[first_alarm]],
                    y=[series.probabilities[first_alarm]],
                    mode="markers",
                    name=f"{names[bearing_id]} · {text['early_alarm']}",
                    marker={"color": "#ff5d64", "size": 10, "symbol": "diamond"},
                    showlegend=False,
                    hovertemplate=(
                        f"{text['early_alarm']}: %{{x}}<br>%{{y:.1%}}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

    bearing_one = data.series_by_bearing[BEARING_ORDER[0]]
    positive_positions = np.flatnonzero(bearing_one.labels[: position + 1] == 1)
    alarm_positions = np.flatnonzero(
        bearing_one.probabilities[: position + 1] > threshold
    )
    if positive_positions.size:
        positive_start = int(positive_positions[0])
        target_timestamp = bearing_one.timestamps[positive_start]
        figure.add_vline(
            x=target_timestamp,
            line_width=1.5,
            line_dash="dot",
            line_color="#b8c2d1",
            annotation_text=text["positive_tail"],
            annotation_position="top right",
            annotation_font_color="#d2dae5",
        )
        if alarm_positions.size and int(alarm_positions[0]) < positive_start:
            alarm_time = datetime.fromisoformat(
                str(bearing_one.timestamps[int(alarm_positions[0])])
            )
            target_time = datetime.fromisoformat(str(target_timestamp))
            lead_hours = (target_time - alarm_time).total_seconds() / 3600.0
            figure.add_annotation(
                x=bearing_one.timestamps[int(alarm_positions[0])],
                y=float(bearing_one.probabilities[int(alarm_positions[0])]),
                text=f"{text['early_alarm']} · {lead_hours:.1f} h {text['lead_proxy']}",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ff747a",
                font={"color": "#ffdadd", "size": 11},
                bgcolor="#222a35",
            )

    figure.add_hline(
        y=threshold,
        line_width=2,
        line_dash="dash",
        line_color="#ff5d64",
        annotation_text=f"{text['maintenance_threshold']} · {threshold:.1%}",
        annotation_position="top left",
        annotation_font_color="#ff9a9f",
    )
    figure.update_layout(
        height=500,
        margin={"l": 8, "r": 18, "t": 55, "b": 12},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#151a22",
        font={"color": "#dce4ef"},
        legend={"orientation": "h", "y": 1.15, "x": 0},
        hovermode="x unified",
        xaxis={
            "title": text["chart_axis_time"],
            "type": "date",
            "gridcolor": "#28313d",
            "linecolor": "#3a4655",
        },
        yaxis={
            "title": text["chart_axis_probability"],
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": "#28313d",
            "linecolor": "#3a4655",
        },
        title={"text": text["chart_title"], "x": 0.01, "xanchor": "left"},
    )
    return figure


def _comparison_figure(data: DashboardData, language: str) -> go.Figure:
    text = TEXT[language]
    model_keys = ("neural_network_cpp", "logistic_regression")
    figure = go.Figure()
    for metric, color in (
        ("precision", "#4da3ff"),
        ("recall", "#f0b547"),
    ):
        figure.add_trace(
            go.Bar(
                x=[text["model_names"][key] for key in model_keys],
                y=[data.model_metrics[key][metric] for key in model_keys],
                name=text[metric],
                marker_color=color,
                text=[
                    f"{data.model_metrics[key][metric]:.1%}"
                    for key in model_keys
                ],
                textposition="outside",
            )
        )
    figure.update_layout(
        title=text["comparison"],
        barmode="group",
        height=330,
        margin={"l": 8, "r": 8, "t": 55, "b": 12},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#151a22",
        font={"color": "#dce4ef"},
        yaxis={
            "title": text["chart_axis_metric"],
            "range": [0, 1.12],
            "tickformat": ".0%",
            "gridcolor": "#28313d",
        },
        xaxis={"title": text["model"]},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def _trend_figure(
    data: DashboardData,
    position: int,
    feature: str,
    title: str,
    language: str,
) -> go.Figure:
    figure = go.Figure()
    for bearing_id, color in zip(BEARING_ORDER, TRACE_COLORS, strict=True):
        series = data.series_by_bearing[bearing_id]
        values = getattr(series, feature)
        figure.add_trace(
            go.Scatter(
                x=series.timestamps[: position + 1],
                y=values[: position + 1],
                mode="lines",
                name=_component_names(language)[bearing_id],
                line={"color": color, "width": 1.8},
            )
        )
    figure.update_layout(
        title=title,
        height=330,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        paper_bgcolor="#0E1117",
        plot_bgcolor="#151a22",
        font={"color": "#dce4ef"},
        hovermode="x unified",
        xaxis={"title": TEXT[language]["chart_axis_time"], "gridcolor": "#28313d"},
        yaxis={"gridcolor": "#28313d"},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def main() -> None:
    _render_styles()
    with st.sidebar:
        language = st.selectbox(TEXT["Italiano"]["language"], LANGUAGES, index=0)
        text = TEXT[language]

    st.markdown(
        '<div class="eyebrow">INDUSTRIAL CONDITION MONITORING · NASA IMS</div>',
        unsafe_allow_html=True,
    )
    st.title(text["title"])
    st.caption(text["subtitle"])
    st.markdown(f'<div class="status-banner">{escape(text["warning"])}</div>', unsafe_allow_html=True)

    try:
        data = load_dashboard_data()
    except (OSError, ValueError, KeyError) as error:
        st.error(f"{text['load_error']}: {error}")
        st.stop()

    sample_count = data.series_by_bearing[BEARING_ORDER[0]].timestamps.size
    if sample_count == 0 or any(
        series.timestamps.size != sample_count
        for series in data.series_by_bearing.values()
    ):
        st.error(text["sequence_error"])
        st.stop()

    st.session_state.setdefault("is_playing", False)
    st.session_state.setdefault("replay_index", 1)
    if not 1 <= int(st.session_state["replay_index"]) <= sample_count:
        st.session_state["replay_index"] = 1
    st.session_state["replay_slider"] = int(st.session_state["replay_index"])

    with st.sidebar:
        st.header(text["replay"])
        threshold = st.slider(
            text["threshold"],
            min_value=0.50,
            max_value=0.999,
            value=data.selected_threshold,
            step=0.001,
            format="%.3f",
            help=text["threshold_help"],
        )
        if st.button(
            text["pause"] if st.session_state["is_playing"] else text["play"],
            use_container_width=True,
        ):
            if st.session_state["is_playing"]:
                st.session_state["is_playing"] = False
            else:
                if st.session_state["replay_index"] >= sample_count:
                    st.session_state["replay_index"] = 1
                    st.session_state["replay_slider"] = 1
                st.session_state["is_playing"] = True
        sample_index = st.slider(
            text["file"],
            min_value=1,
            max_value=sample_count,
            step=1,
            key="replay_slider",
            help=text["file_help"],
            on_change=_replay_status_callback,
        )
        st.caption(
            f"{text['timestamp']}: "
            f"{data.series_by_bearing[BEARING_ORDER[0]].timestamps[sample_index - 1]}"
        )
        st.caption(text["source"])

    readings = readings_at_position(
        data.series_by_bearing,
        sample_index - 1,
        threshold,
    )
    machine_probability = max(reading.probability for reading in readings)
    machine_status = classify_probability(machine_probability, threshold)
    critical_count = sum(reading.status == "critical" for reading in readings)
    _, _, state_icon = STATUS_STYLE[machine_status]

    st.subheader(text["machine"])
    kpis = st.columns(4)
    kpis[0].metric(
        text["machine_status"],
        f"{state_icon} {text['status_short'][machine_status]}",
        delta=f"{machine_probability:.1%} {text['probability'].lower()}",
    )
    kpis[0].caption(_status_text(machine_status, language))
    kpis[1].metric(
        text["global_health"],
        f"{(1.0 - machine_probability) * 100.0:.1f}%",
    )
    kpis[2].metric(
        text["bearings"],
        f"{len(readings)} / 4 {text['active']}",
        delta=f"{critical_count} {text['critical_count']}",
        delta_color="inverse" if critical_count else "normal",
    )
    kpis[3].metric(text["engine"], "C++ Active")
    kpis[3].caption(text["engine_badge"])

    st.subheader(text["bearing_status"])
    cards = st.columns(4, gap="medium")
    for column, reading in zip(cards, readings, strict=True):
        _render_bearing_card(column, reading, language)

    if critical_count:
        critical_names = ", ".join(
            _component_names(language)[reading.bearing_id]
            for reading in readings
            if reading.status == "critical"
        )
        st.error(f"{_status_text('critical', language)} · {critical_names}")
    elif machine_status == "prealarm":
        st.warning(_status_text("prealarm", language))

    st.plotly_chart(
        _probability_figure(data, sample_index - 1, threshold, language),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(text["label_caveat"])

    with st.expander(text["advanced"], expanded=False):
        st.plotly_chart(
            _comparison_figure(data, language),
            use_container_width=True,
            config={"displaylogo": False},
        )
        trends = st.columns(2)
        trends[0].plotly_chart(
            _trend_figure(
                data,
                sample_index - 1,
                "rms",
                text["technical_rms"],
                language,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )
        trends[1].plotly_chart(
            _trend_figure(
                data,
                sample_index - 1,
                "kurtosis",
                text["technical_kurtosis"],
                language,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

        st.markdown(f"#### {text['order_title']}")
        report_csv, report_txt = build_maintenance_reports(readings, threshold)
        if st.session_state["is_playing"]:
            st.caption(text["report_pause"])
        else:
            if report_csv is None or report_txt is None:
                st.info(text["order_none"])
            else:
                st.warning(text["order_ready"])
            report_columns = st.columns(2)
            report_columns[0].download_button(
                text["download_csv"],
                data=report_csv or "",
                file_name=f"stonecut_maintenance_order_{sample_index:04d}.csv",
                mime="text/csv",
                disabled=report_csv is None,
                use_container_width=True,
            )
            report_columns[1].download_button(
                text["download_txt"],
                data=report_txt or "",
                file_name=f"stonecut_maintenance_order_{sample_index:04d}.txt",
                mime="text/plain",
                disabled=report_txt is None,
                use_container_width=True,
            )
        st.caption(text["label_caveat"])

    if st.session_state["is_playing"]:
        if sample_index >= sample_count:
            st.session_state["is_playing"] = False
        else:
            time.sleep(0.1)
            st.session_state["replay_index"] = sample_index + 1
            if sample_index + 1 >= sample_count:
                st.session_state["is_playing"] = False
            st.rerun()


if __name__ == "__main__":
    main()
