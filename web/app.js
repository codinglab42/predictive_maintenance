"use strict";

const translations = {
  it: {
    language: "Lingua", replay: "Replay storico NASA IMS · 2nd_test", threshold: "Soglia di allarme",
    acquisition: "Acquisizione", source: "Modello C++ · punteggi precalcolati sul dataset IMS",
    subtitle: "Monitoraggio predittivo per macchine da taglio della pietra",
    banner: "Monitoraggio dei mandrini: simulazione su dati storici IMS. Punteggi non calibrati; ogni segnalazione richiede verifica umana.",
    machine: "Macchina", machineStatus: "Stato macchina", globalHealth: "Health Index globale",
    bearings: "Cuscinetti monitorati", engine: "Motore predittivo", engineNote: "Punteggi precalcolati", bearingStatus: "Stato cuscinetti macchina",
    chartTitle: "Andamento storico della probabilità di guasto",
    filterNote: "Media mobile sulle ultime 5 acquisizioni (meno all'inizio); allarme dopo 3 valori consecutivi sopra soglia.",
    advanced: "Diagnostica Avanzata & Confronto Modelli (Ingegneria)",
    methodology: "Soglia selezionata sul 2nd_test: analisi esplorativa, non test indipendente. Punteggi Sigmoide non calibrati. La coda positiva è un'etichetta proxy, non il momento osservato del guasto.",
    metricsNote: "Le metriche confrontano i punteggi originali, non la regola filtrata del cruscotto.",
    report: "Report di manutenzione", downloadCsv: "Scarica report CSV", downloadTxt: "Scarica report TXT",
    operational: "OPERATIVO", prealarm: "PRE-ALLARME", critical: "CONTROLLO CONSIGLIATO",
    play: "▶ Play", pause: "⏸ Pausa", active: "Attivi", criticalCount: "critici",
    health: "Health Score", probability: "Punteggio di guasto", rms: "RMS attuale",
    thresholdLine: "Soglia di intervento", firstAlarm: "Primo allarme confermato",
    positiveTail: "Inizio coda positiva (proxy)",
    comparison: "Rete neurale C++ vs benchmark logistico", logistic: "Regressione logistica",
    neural: "Rete neurale C++", precision: "Precision", recall: "Recall",
    rmsTrend: "Andamento RMS", kurtosisTrend: "Andamento curtosi", time: "Timestamp acquisizione",
    noReport: "Nessun allarme confermato nell'acquisizione selezionata.",
    reportReady: "Allarme confermato: scarica il report per la verifica umana.",
    reportPaused: "Metti in pausa il replay per scaricare il report corrente.",
    loadError: "Impossibile caricare il cruscotto", names: [
      "Mandrino Lama Principale (Ant.)", "Mandrino Lama Principale (Post.)",
      "Volano Trasmissione", "Rinvio Motoriduttore",
    ],
  },
  en: {
    language: "Language", replay: "NASA IMS historical replay · 2nd_test", threshold: "Alarm threshold",
    acquisition: "Acquisition", source: "C++ model · precomputed scores from the IMS dataset",
    subtitle: "Predictive monitoring for stone-cutting machines",
    banner: "Spindle monitoring: simulation using historical IMS data. Scores are uncalibrated; every alert requires human review.",
    machine: "Machine", machineStatus: "Machine status", globalHealth: "Global Health Index",
    bearings: "Bearings monitored", engine: "Predictive engine", engineNote: "Precomputed scores", bearingStatus: "Machine bearing status",
    chartTitle: "Historical failure probability",
    filterNote: "Moving average over the last 5 acquisitions (fewer at the start); alarm after 3 consecutive values above threshold.",
    advanced: "Advanced Diagnostics & Model Comparison (Engineering)",
    methodology: "Threshold selected on 2nd_test: exploratory analysis, not an independent test. Sigmoid scores are uncalibrated. The positive tail is a proxy label, not the observed failure time.",
    metricsNote: "Comparison metrics use the original scores, not the dashboard's filtered rule.",
    report: "Maintenance report", downloadCsv: "Download CSV report", downloadTxt: "Download TXT report",
    operational: "OPERATIONAL", prealarm: "PRE-ALARM", critical: "INSPECTION RECOMMENDED",
    play: "▶ Play", pause: "⏸ Pause", active: "Active", criticalCount: "critical",
    health: "Health Score", probability: "Failure score", rms: "Current RMS",
    thresholdLine: "Maintenance threshold", firstAlarm: "First confirmed alarm",
    positiveTail: "Start of positive tail (proxy)",
    comparison: "C++ neural network vs logistic benchmark", logistic: "Logistic regression",
    neural: "C++ neural network", precision: "Precision", recall: "Recall",
    rmsTrend: "RMS trend", kurtosisTrend: "Kurtosis trend", time: "Acquisition timestamp",
    noReport: "No confirmed alarm at the selected acquisition.",
    reportReady: "Confirmed alarm: download the report for human review.",
    reportPaused: "Pause the replay to download the current report.",
    loadError: "Unable to load the dashboard", names: [
      "Main blade spindle (front)", "Main blade spindle (rear)", "Drive flywheel", "Gearmotor transfer shaft",
    ],
  },
  zh: {
    language: "语言", replay: "NASA IMS 历史回放 · 2nd_test", threshold: "报警阈值",
    acquisition: "采集文件", source: "C++ 模型 · IMS 数据集预计算分数",
    subtitle: "石材切割设备预测性维护监控",
    banner: "主轴监测：基于 IMS 历史数据的模拟。分数未经校准；所有报警均需人工复核。",
    machine: "设备", machineStatus: "设备状态", globalHealth: "全局健康指数",
    bearings: "监测轴承", engine: "预测引擎", engineNote: "预计算分数", bearingStatus: "设备轴承状态",
    chartTitle: "历史故障概率趋势",
    filterNote: "最近 5 次采集的移动平均值（开始时不足 5 次）；连续 3 次超过阈值后报警。",
    advanced: "高级诊断与模型对比（工程）",
    methodology: "阈值使用 2nd_test 标签选择：仅为探索性分析，并非独立测试。Sigmoid 分数未经校准。正标签区间是代理标签，并非实际观测到的故障时刻。",
    metricsNote: "对比指标使用原始分数，不使用仪表盘滤波规则。",
    report: "维护报告", downloadCsv: "下载 CSV 报告", downloadTxt: "下载 TXT 报告",
    operational: "运行正常", prealarm: "预警", critical: "建议检查",
    play: "▶ 播放", pause: "⏸ 暂停", active: "运行中", criticalCount: "临界",
    health: "健康评分", probability: "故障分数", rms: "当前 RMS",
    thresholdLine: "维护阈值", firstAlarm: "首次确认报警",
    positiveTail: "正标签区间起点（代理标签）",
    comparison: "C++ 神经网络与逻辑回归基准", logistic: "逻辑回归",
    neural: "C++ 神经网络", precision: "精确率", recall: "召回率",
    rmsTrend: "RMS 趋势", kurtosisTrend: "峭度趋势", time: "采集时间",
    noReport: "当前采集文件没有已确认的报警。",
    reportReady: "已确认报警：下载报告供人工复核。",
    reportPaused: "请暂停回放后下载当前报告。",
    loadError: "无法加载仪表盘", names: ["主刀片主轴（前）", "主刀片主轴（后）", "传动飞轮", "减速电机传动轴"],
  },
};

const $ = (id) => document.getElementById(id);
const colors = ["#4da3ff", "#b58cff", "#36c6b5", "#ff9f68"];
let data;
let position = 0;
let threshold = 0.999;
let language = "it";
let timer = null;
let chartTimer = null;
let chartRunning = false;
let chartDirty = false;
let lastChart = 0;

function movingAverage(values) {
  return values.map((_, index) => {
    const window = values.slice(Math.max(0, index - 4), index + 1);
    return window.reduce((total, value) => total + value, 0) / window.length;
  });
}

function confirmation(values, limit) {
  let count = 0;
  return values.map((value) => {
    count = value > limit ? count + 1 : 0;
    return count >= 3;
  });
}

function status(series) {
  const score = series.smoothed[position];
  if (series.alarms[position]) return "critical";
  return score >= 0.45 ? "prealarm" : "operational";
}

function percent(value) { return `${(value * 100).toFixed(1)}%`; }
function currentText() { return translations[language]; }

function setLanguage() {
  language = $("language").value;
  document.documentElement.lang = language;
  const t = currentText();
  document.querySelectorAll("[data-text]").forEach((node) => {
    node.textContent = t[node.dataset.text];
  });
  if (data) render();
}

function updateAlarms() {
  for (const series of data.series) {
    series.alarms = confirmation(series.smoothed, threshold);
  }
}

function render() {
  const t = currentText();
  const states = data.series.map(status);
  const maxScore = Math.max(...data.series.map((series) => series.smoothed[position]));
  const critical = states.filter((state) => state === "critical").length;
  const machine = critical ? "critical" : states.includes("prealarm") ? "prealarm" : "operational";
  const icon = { operational: "🟢", prealarm: "🟡", critical: "🔴" };
  $("threshold-value").textContent = threshold.toFixed(3);
  $("position").value = String(position + 1);
  $("position-value").textContent = `${position + 1} / ${data.series[0].timestamps.length}`;
  $("timestamp").textContent = `Timestamp: ${data.series[0].timestamps[position]}`;
  $("play").textContent = timer ? t.pause : t.play;
  $("machine-status").textContent = `${icon[machine]} ${t[machine]}`;
  $("machine-score").textContent = `${percent(maxScore)} · ${t.probability}`;
  $("global-health").textContent = percent(1 - maxScore);
  $("bearing-count").textContent = `${data.series.length} / ${data.series.length} ${t.active}`;
  $("critical-count").textContent = `${critical} ${t.criticalCount}`;
  $("cards").replaceChildren(...data.series.map((series, index) => {
    const state = states[index];
    const health = 1 - series.smoothed[position];
    const card = document.createElement("article");
    card.className = `card ${state}`;
    const heading = document.createElement("h3");
    heading.textContent = t.names[index];
    const chip = document.createElement("div");
    chip.className = "status";
    chip.textContent = `${icon[state]} ${t[state]}`;
    const healthLine = document.createElement("div");
    healthLine.className = "health";
    healthLine.textContent = percent(health);
    const healthLabel = document.createElement("span");
    healthLabel.textContent = ` ${t.health}`;
    healthLine.append(healthLabel);
    const track = document.createElement("div");
    track.className = "track";
    const bar = document.createElement("div");
    bar.style.width = `${Math.max(0, Math.min(100, health * 100))}%`;
    track.append(bar);
    const row = (label, value) => {
      const element = document.createElement("div");
      element.className = "kv";
      const name = document.createElement("span");
      name.textContent = label;
      const text = document.createElement("strong");
      text.textContent = value;
      element.append(name, text);
      return element;
    };
    card.append(heading, chip, healthLine, track,
      row(t.rms, series.rms[position].toFixed(4)),
      row(t.probability, percent(series.smoothed[position])));
    return card;
  }));
  $("alarm").hidden = !critical;
  $("alarm").textContent = critical
    ? `${t.critical} · ${states.map((state, index) => state === "critical" ? t.names[index] : null).filter(Boolean).join(", ")}`
    : "";
  $("report-status").textContent = timer ? t.reportPaused : critical ? t.reportReady : t.noReport;
  $("csv").disabled = $("txt").disabled = Boolean(timer) || !critical;
  requestCharts();
}

function commonLayout() {
  return {
    paper_bgcolor: "#0e1117", plot_bgcolor: "#151a22", font: { color: "#dce4ef" },
    margin: { l: 55, r: 18, t: 55, b: 45 },
    legend: { orientation: "h", y: 1.02, x: 1, xanchor: "right", yanchor: "bottom" },
    hovermode: "x unified",
    xaxis: { type: "date", gridcolor: "#28313d", title: { text: currentText().time } },
    yaxis: { gridcolor: "#28313d" },
  };
}

function drawCharts() {
  const t = currentText();
  const end = position + 1;
  const traces = [];
  data.series.forEach((series, index) => {
    traces.push({
      x: series.timestamps.slice(0, end), y: series.smoothed.slice(0, end),
      type: "scatter", mode: "lines", name: t.names[index], line: { color: colors[index], width: 2 },
      hovertemplate: "%{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
    });
    const first = series.alarms.slice(0, end).indexOf(true);
    if (first >= 0) traces.push({
      x: [series.timestamps[first]], y: [series.smoothed[first]], type: "scatter",
      mode: "markers", name: `${t.names[index]} · ${t.firstAlarm}`, showlegend: false,
      marker: { color: "#ff5d64", size: 10, symbol: "diamond" },
    });
  });
  const layout = commonLayout();
  layout.yaxis = { ...layout.yaxis, range: [0, 1], tickformat: ".0%" };
  layout.shapes = [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: threshold, y1: threshold, line: { color: "#ff5d64", width: 2, dash: "dash" } }];
  layout.annotations = [{ xref: "paper", x: 0, y: threshold, text: `${t.thresholdLine} · ${percent(threshold)}`, showarrow: false, xanchor: "left", yanchor: "top", font: { color: "#ff9a9f" } }];
  const config = { responsive: true, displaylogo: false, scrollZoom: true };
  const draws = [Plotly.react("chart", traces, layout, config)];
  if ($("diagnostics").open) {
    const diagnosticLayout = {
      ...layout,
      shapes: [...layout.shapes],
      annotations: [...layout.annotations],
    };
    const proxy = data.series[0].labels.slice(0, end).indexOf(1);
    if (proxy >= 0) {
      const timestamp = data.series[0].timestamps[proxy];
      diagnosticLayout.shapes.push({
        type: "line", x0: timestamp, x1: timestamp, yref: "paper", y0: 0, y1: 1,
        line: { color: "#b8c2d1", width: 1.5, dash: "dot" },
      });
      diagnosticLayout.annotations.push({
        x: timestamp, yref: "paper", y: 0.9, text: t.positiveTail,
        showarrow: false, xanchor: "right", font: { color: "#d2dae5" },
      });
    }
    draws.push(Plotly.react("diagnostic-chart", traces, diagnosticLayout, config));
    const metricLayout = commonLayout();
    metricLayout.title = { text: t.comparison };
    metricLayout.barmode = "group";
    metricLayout.xaxis = {};
    metricLayout.yaxis = { range: [0, 1.12], tickformat: ".0%", gridcolor: "#28313d" };
    draws.push(Plotly.react("comparison", ["precision", "recall"].map((key, index) => ({
      type: "bar", name: t[key], x: [t.neural, t.logistic],
      y: [data.metrics.neural_network_cpp[key], data.metrics.logistic_regression[key]],
      marker: { color: colors[index] },
    })), metricLayout, config));
    for (const [id, feature, title] of [["rms-chart", "rms", t.rmsTrend], ["kurtosis-chart", "kurtosis", t.kurtosisTrend]]) {
      const trendLayout = commonLayout();
      trendLayout.title = { text: title };
      draws.push(Plotly.react(id, data.series.map((series, index) => ({
        type: "scatter", mode: "lines", name: t.names[index], line: { color: colors[index] },
        x: series.timestamps.slice(0, end), y: series[feature].slice(0, end),
      })), trendLayout, config));
    }
  }
  return Promise.all(draws);
}

function showError(error) {
  $("error").hidden = false;
  $("error").textContent = `${currentText().loadError}: ${error.message}`;
}

function requestCharts() {
  chartDirty = true;
  if (chartRunning || chartTimer !== null) return;
  const delay = Math.max(0, 250 - (performance.now() - lastChart));
  chartTimer = setTimeout(async () => {
    chartTimer = null;
    chartRunning = true;
    chartDirty = false;
    try {
      await drawCharts();
    } catch (error) {
      showError(error);
    } finally {
      lastChart = performance.now();
      chartRunning = false;
      if (chartDirty) requestCharts();
    }
  }, delay);
}

function pause() {
  if (timer !== null) clearInterval(timer);
  timer = null;
}

function download(format) {
  const t = currentText();
  const rows = data.series.flatMap((series, index) => status(series) === "critical" ? [{
    timestamp: series.timestamps[position], bearing_id: series.id, component: t.names[index],
    filtered_failure_score: series.smoothed[position].toFixed(6), health_index_percent: ((1 - series.smoothed[position]) * 100).toFixed(2),
    rms: series.rms[position].toFixed(6), kurtosis: series.kurtosis[position].toFixed(6),
    alarm_threshold: threshold.toFixed(3), recommended_action: "Inspect; replacement requires human assessment",
    context: "Historical IMS simulation; filtered scores are uncalibrated",
  }] : []);
  if (!rows.length || timer) return;
  const columns = Object.keys(rows[0]);
  const csvEscape = (value) => `"${String(value).replaceAll('"', '""')}"`;
  const content = format === "csv"
    ? [columns.join(","), ...rows.map((row) => columns.map((key) => csvEscape(row[key])).join(","))].join("\r\n")
    : ["STONECUT PREDICTIVE MAINTENANCE — INSPECTION REPORT",
      "Historical simulation; scores are uncalibrated. Human review required.",
      ...rows.map((row) => columns.map((key) => `${key}: ${row[key]}`).join("\n"))].join("\n\n");
  const url = URL.createObjectURL(new Blob([content], { type: format === "csv" ? "text/csv;charset=utf-8" : "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `stonecut_inspection_${String(position + 1).padStart(4, "0")}.${format}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("language").addEventListener("change", setLanguage);
$("threshold").addEventListener("input", (event) => {
  if (!data) return;
  threshold = Number(event.target.value);
  updateAlarms();
  render();
});
$("position").addEventListener("input", (event) => {
  pause();
  position = Number(event.target.value) - 1;
  render();
});
$("play").addEventListener("click", () => {
  if (timer) pause();
  else {
    if (position === data.series[0].timestamps.length - 1) position = 0;
    timer = setInterval(() => {
      position += 1;
      if (position >= data.series[0].timestamps.length - 1) {
        position = data.series[0].timestamps.length - 1;
        pause();
      }
      render();
    }, 100);
  }
  render();
});
$("diagnostics").addEventListener("toggle", () => { if (data) requestCharts(); });
$("csv").addEventListener("click", () => download("csv"));
$("txt").addEventListener("click", () => download("txt"));

fetch("/api/replay")
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then((payload) => {
    data = payload;
    threshold = data.threshold;
    $("threshold").value = String(threshold);
    $("position").max = String(data.series[0].timestamps.length);
    for (const series of data.series) series.smoothed = movingAverage(series.probabilities);
    updateAlarms();
    $("position").disabled = false;
    $("play").disabled = false;
    setLanguage();
  })
  .catch(showError);
