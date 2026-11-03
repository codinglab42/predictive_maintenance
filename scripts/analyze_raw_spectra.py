#!/usr/bin/env python3
"""Compare raw vibration spectra across h20 alarm groups for one FEMTO bearing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.femto_data import DEFAULT_DATA_DIR, discover_bearing_files


SAMPLE_RATE_HZ = 25_600.0
PSD_BANDS_HZ = (
    ("0-1kHz", 0.0, 1_000.0),
    ("1-5kHz", 1_000.0, 5_000.0),
    ("5-10kHz", 5_000.0, 10_000.0),
    ("10-12.8kHz", 10_000.0, 12_800.0),
)
COHERENCE_SEGMENT_SAMPLES = 256
BEARING_MODEL_SCENARIO = "NSK 6307DU"
BEARING_SPEED_RPM = 1_800.0
ROLLING_ELEMENT_COUNT = 8
ROLLING_ELEMENT_DIAMETER_MM = 13.49
PITCH_DIAMETER_MM = 57.5
CONTACT_ANGLE_DEGREES = 0.0
LOW_FREQUENCY_PLOT_MAX_HZ = 400.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        help="directory with the h20 timeline, episode CSV, and validation JSON",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bearing", default="Bearing1_1")
    parser.add_argument(
        "--objective",
        choices=("balanced_f1", "sensitivity_first", "precision_at_80pct_recall"),
        default="balanced_f1",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _select_sample_groups(
    timeline_rows: list[dict[str, str]],
    episode_rows: list[dict[str, str]],
    horizon: int,
) -> dict[str, list[int]]:
    if horizon < 1:
        raise ValueError("horizon must be a positive number of acquisitions")

    by_index: dict[int, dict[str, str]] = {}
    for row in timeline_rows:
        sample_index = int(row["sample_index"])
        if sample_index in by_index:
            raise ValueError(f"duplicate timeline sample index: {sample_index}")
        by_index[sample_index] = row
    if not by_index:
        raise ValueError("timeline contains no samples")

    final_sample = max(by_index)
    expected_indices = set(range(min(by_index), final_sample + 1))
    if set(by_index) != expected_indices:
        raise ValueError("timeline sample indices are not contiguous")
    window_start = final_sample - horizon + 1
    if window_start < min(by_index):
        raise ValueError("horizon exceeds the number of timeline samples")

    false_alarm_indices: set[int] = set()
    persistent_pre_window_indices: set[int] = set()
    for episode in episode_rows:
        start = int(episode["start_sample_index"])
        end = int(episode["end_sample_index"])
        if start > end:
            raise ValueError(f"episode has reversed sample bounds: {episode}")
        indices = set(range(start, end + 1))
        if not indices.issubset(by_index):
            raise ValueError(f"episode references samples outside the timeline: {episode}")
        classification = episode["classification"]
        if classification == "false_before_window":
            false_alarm_indices.update(indices)
        elif classification == "early_persistent":
            persistent_pre_window_indices.update(
                index for index in indices if index < window_start
            )

    groups = {
        "far_below_threshold": [
            index
            for index, row in by_index.items()
            if int(row["distance_to_end_acquisitions"]) >= 200
            and int(row["above_threshold"]) == 0
        ],
        "confirmed_false_episodes": sorted(false_alarm_indices),
        "persistent_episode_before_h20": sorted(persistent_pre_window_indices),
        "final_h20_window": [
            index
            for index, row in by_index.items()
            if int(row["distance_to_end_acquisitions"]) < horizon
        ],
    }
    empty_groups = [name for name, indices in groups.items() if not indices]
    if empty_groups:
        raise ValueError(f"no samples selected for group(s): {', '.join(empty_groups)}")
    return groups


def _periodogram(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if signal.ndim != 2 or signal.shape[1] != 2 or signal.shape[0] < 8:
        raise ValueError("raw vibration signal must have two channels and at least 8 rows")
    if not np.isfinite(signal).all():
        raise ValueError("raw vibration signal contains non-finite values")

    centered = signal - signal.mean(axis=0, keepdims=True)
    window = np.hanning(signal.shape[0])[:, None]
    transformed = np.fft.rfft(centered * window, axis=0)
    power = np.square(np.abs(transformed)) / (
        SAMPLE_RATE_HZ * np.sum(np.square(window))
    )
    if signal.shape[0] % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / SAMPLE_RATE_HZ)
    return frequencies, power


def _bearing_characteristic_frequencies() -> dict[str, float]:
    rotation_hz = BEARING_SPEED_RPM / 60.0
    diameter_ratio = ROLLING_ELEMENT_DIAMETER_MM / PITCH_DIAMETER_MM
    contact_cosine = np.cos(np.deg2rad(CONTACT_ANGLE_DEGREES))
    geometry_factor = diameter_ratio * contact_cosine
    return {
        "shaft_1x": rotation_hz,
        "FTF": 0.5 * rotation_hz * (1.0 - geometry_factor),
        "BSF": (
            PITCH_DIAMETER_MM
            / (2.0 * ROLLING_ELEMENT_DIAMETER_MM)
            * rotation_hz
            * (1.0 - geometry_factor**2)
        ),
        "BPFO": (
            ROLLING_ELEMENT_COUNT
            / 2.0
            * rotation_hz
            * (1.0 - geometry_factor)
        ),
        "BPFI": (
            ROLLING_ELEMENT_COUNT
            / 2.0
            * rotation_hz
            * (1.0 + geometry_factor)
        ),
    }


def _frequency_peak_rows(
    bearing_id: str,
    groups: dict[str, list[int]],
    frequencies: np.ndarray,
    psd_by_group: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if "far_below_threshold" not in psd_by_group:
        raise ValueError("far_below_threshold is required as the reference group")
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("frequency axis must contain at least two bins")
    frequency_step = float(frequencies[1] - frequencies[0])
    if frequency_step <= 0.0:
        raise ValueError("frequency axis must be strictly increasing")

    reference = psd_by_group["far_below_threshold"]
    characteristic_frequencies = _bearing_characteristic_frequencies()
    peak_rows: list[dict[str, Any]] = []
    for frequency_name, fundamental in characteristic_frequencies.items():
        for harmonic_order in (1, 2, 3):
            target_frequency = fundamental * harmonic_order
            if target_frequency > LOW_FREQUENCY_PLOT_MAX_HZ:
                continue
            nearest_bin_index = int(np.argmin(np.abs(frequencies - target_frequency)))
            local_start = max(0, nearest_bin_index - 1)
            local_end = min(frequencies.size, nearest_bin_index + 2)

            for group_name, spectrum in psd_by_group.items():
                for channel_index, channel_name in enumerate(("horizontal", "vertical")):
                    local_power = spectrum[local_start:local_end, channel_index]
                    peak_bin_index = local_start + int(np.argmax(local_power))
                    reference_power = float(reference[nearest_bin_index, channel_index])
                    measured_power = float(
                        spectrum[nearest_bin_index, channel_index]
                    )
                    peak_rows.append(
                        {
                            "bearing_id": bearing_id,
                            "model_scenario": BEARING_MODEL_SCENARIO,
                            "group": group_name,
                            "channel": channel_name,
                            "frequency_name": frequency_name,
                            "harmonic_order": harmonic_order,
                            "theoretical_frequency_hz": float(target_frequency),
                            "nearest_bin_frequency_hz": float(
                                frequencies[nearest_bin_index]
                            ),
                            "nearest_bin_psd": measured_power,
                            "peak_within_one_bin_hz": float(frequencies[peak_bin_index]),
                            "peak_offset_hz": float(
                                frequencies[peak_bin_index] - target_frequency
                            ),
                            "reference_nearest_bin_psd": reference_power,
                            "psd_ratio_vs_far_reference": (
                                measured_power / reference_power
                                if reference_power > 0.0
                                else ""
                            ),
                            "sample_count": len(groups[group_name]),
                            "fft_resolution_hz": frequency_step,
                        }
                    )
    return peak_rows


def _welch_components(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    segment_size = COHERENCE_SEGMENT_SAMPLES
    step = segment_size // 2
    if signal.shape[0] < segment_size:
        raise ValueError("raw vibration signal is too short for coherence estimation")
    window = np.hanning(segment_size)
    scale = SAMPLE_RATE_HZ * np.sum(np.square(window))
    auto_horizontal = np.zeros(segment_size // 2 + 1, dtype=np.float64)
    auto_vertical = np.zeros_like(auto_horizontal)
    cross = np.zeros(auto_horizontal.size, dtype=np.complex128)

    segment_count = 0
    for start in range(0, signal.shape[0] - segment_size + 1, step):
        frame = signal[start : start + segment_size]
        frame = frame - frame.mean(axis=0, keepdims=True)
        transformed = np.fft.rfft(frame * window[:, None], axis=0)
        horizontal, vertical = transformed.T
        auto_horizontal += np.square(np.abs(horizontal)) / scale
        auto_vertical += np.square(np.abs(vertical)) / scale
        cross += horizontal * np.conjugate(vertical) / scale
        segment_count += 1

    for values in (auto_horizontal, auto_vertical, cross):
        if segment_size % 2 == 0:
            values[1:-1] *= 2.0
        else:
            values[1:] *= 2.0
    auto_horizontal /= segment_count
    auto_vertical /= segment_count
    cross /= segment_count
    frequencies = np.fft.rfftfreq(segment_size, d=1.0 / SAMPLE_RATE_HZ)
    return frequencies, auto_horizontal, auto_vertical, cross


def _read_raw_signal(path: Path) -> np.ndarray:
    try:
        sample = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read raw vibration CSV {path}: {error}") from error
    if sample.shape[1] == 6:
        sample = sample[:, -2:]
    elif sample.shape[1] != 2:
        raise ValueError(f"expected 2 or 6 columns in {path}, got {sample.shape[1]}")
    return sample


def _analyze_groups(
    groups: dict[str, list[int]],
    files_by_index: dict[int, Path],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    psd_by_group: dict[str, np.ndarray] = {}
    coherence_by_group: dict[str, np.ndarray] = {}
    band_rows: list[dict[str, Any]] = []
    reference_frequencies: np.ndarray | None = None
    coherence_frequencies: np.ndarray | None = None

    for group_name, sample_indices in groups.items():
        group_spectra: list[np.ndarray] = []
        total_horizontal = np.zeros(COHERENCE_SEGMENT_SAMPLES // 2 + 1)
        total_vertical = np.zeros_like(total_horizontal)
        total_cross = np.zeros(total_horizontal.size, dtype=np.complex128)

        for sample_index in sample_indices:
            path = files_by_index.get(sample_index)
            if path is None:
                raise FileNotFoundError(
                    f"no raw acquisition file found for sample {sample_index}"
                )
            signal = _read_raw_signal(path)
            frequencies, spectrum = _periodogram(signal)
            if reference_frequencies is None:
                reference_frequencies = frequencies
            elif not np.array_equal(frequencies, reference_frequencies):
                raise ValueError(f"inconsistent sample length in {path}")
            group_spectra.append(spectrum)

            (
                current_coherence_frequencies,
                horizontal,
                vertical,
                cross,
            ) = _welch_components(signal)
            if coherence_frequencies is None:
                coherence_frequencies = current_coherence_frequencies
            elif not np.array_equal(
                current_coherence_frequencies, coherence_frequencies
            ):
                raise ValueError(f"inconsistent coherence frequencies in {path}")
            total_horizontal += horizontal
            total_vertical += vertical
            total_cross += cross

        stacked_spectra = np.stack(group_spectra)
        psd_by_group[group_name] = np.median(stacked_spectra, axis=0)
        coherence_by_group[group_name] = np.clip(
            np.square(np.abs(total_cross))
            / np.maximum(total_horizontal * total_vertical, np.finfo(float).tiny),
            0.0,
            1.0,
        )

        assert reference_frequencies is not None
        frequency_step = float(reference_frequencies[1] - reference_frequencies[0])
        for channel_index, channel_name in enumerate(("horizontal", "vertical")):
            for band_name, lower, upper in PSD_BANDS_HZ:
                mask = (reference_frequencies >= lower) & (reference_frequencies < upper)
                powers = np.sum(stacked_spectra[:, mask, channel_index], axis=1) * frequency_step
                band_rows.append(
                    {
                        "group": group_name,
                        "channel": channel_name,
                        "frequency_band": band_name,
                        "sample_count": len(sample_indices),
                        "median_band_power": float(np.median(powers)),
                    }
                )

    if reference_frequencies is None or coherence_frequencies is None:
        raise ValueError("no raw signals were analyzed")
    return (
        reference_frequencies,
        psd_by_group,
        coherence_frequencies,
        coherence_by_group,
        band_rows,
    )


def _plot_results(
    output_path: Path,
    bearing_id: str,
    horizon: int,
    groups: dict[str, list[int]],
    frequencies: np.ndarray,
    psd_by_group: dict[str, np.ndarray],
    coherence_frequencies: np.ndarray,
    coherence_by_group: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "matplotlib is required; install project requirements before plotting"
        ) from error

    labels = {
        "far_below_threshold": "Far reference (>=200, below threshold)",
        "confirmed_false_episodes": "Confirmed pre-h20 episodes",
        "persistent_episode_before_h20": "Persistent episode before h20",
        "final_h20_window": f"Final h{horizon} window",
    }
    colors = {
        "far_below_threshold": "tab:blue",
        "confirmed_false_episodes": "tab:orange",
        "persistent_episode_before_h20": "tab:red",
        "final_h20_window": "tab:green",
    }
    figure, axes = plt.subplots(3, 1, figsize=(13, 14), constrained_layout=True)
    for channel_index, channel_name in enumerate(("Horizontal", "Vertical")):
        axis = axes[channel_index]
        for group_name, spectrum in psd_by_group.items():
            axis.loglog(
                frequencies[1:],
                np.maximum(spectrum[1:, channel_index], np.finfo(float).tiny),
                label=f"{labels[group_name]} (n={len(groups[group_name])})",
                color=colors[group_name],
            )
        axis.set_title(f"{channel_name} channel: median raw periodogram")
        axis.set_ylabel("PSD (raw units squared / Hz)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize="small")

    for group_name, coherence in coherence_by_group.items():
        axes[2].plot(
            coherence_frequencies[1:],
            coherence[1:],
            label=f"{labels[group_name]} (n={len(groups[group_name])})",
            color=colors[group_name],
        )
    axes[2].set_title("Pooled horizontal-vertical magnitude-squared coherence")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Coherence")
    axes[2].set_ylim(0, 1.02)
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize="small")
    figure.suptitle(
        f"{bearing_id}: raw vibration spectra, h{horizon} exploratory comparison"
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_characteristic_frequencies(
    output_path: Path,
    bearing_id: str,
    groups: dict[str, list[int]],
    frequencies: np.ndarray,
    psd_by_group: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    labels = {
        "far_below_threshold": "Far reference",
        "confirmed_false_episodes": "Confirmed pre-h20 episodes",
        "persistent_episode_before_h20": "Persistent episode before h20",
        "final_h20_window": "Final h20 window",
    }
    colors = {
        "far_below_threshold": "tab:blue",
        "confirmed_false_episodes": "tab:orange",
        "persistent_episode_before_h20": "tab:red",
        "final_h20_window": "tab:green",
    }
    characteristic_colors = {
        "shaft_1x": "black",
        "FTF": "tab:purple",
        "BSF": "tab:brown",
        "BPFO": "tab:red",
        "BPFI": "tab:cyan",
    }
    characteristic_frequencies = _bearing_characteristic_frequencies()
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    frequency_mask = (frequencies > 0) & (
        frequencies <= LOW_FREQUENCY_PLOT_MAX_HZ
    )
    for channel_index, channel_name in enumerate(("Horizontal", "Vertical")):
        axis = axes[channel_index]
        for group_name, spectrum in psd_by_group.items():
            axis.semilogy(
                frequencies[frequency_mask],
                np.maximum(
                    spectrum[frequency_mask, channel_index], np.finfo(float).tiny
                ),
                label=f"{labels[group_name]} (n={len(groups[group_name])})",
                color=colors[group_name],
            )

        for name, target_frequency in characteristic_frequencies.items():
            marker_color = characteristic_colors[name]
            axis.axvline(
                target_frequency,
                color=marker_color,
                linestyle="-",
                alpha=0.8,
                linewidth=1.2,
                label=f"{name} {target_frequency:.1f} Hz",
            )
            for harmonic in (2, 3):
                harmonic_frequency = target_frequency * harmonic
                if harmonic_frequency <= LOW_FREQUENCY_PLOT_MAX_HZ:
                    axis.axvline(
                        harmonic_frequency,
                        color=marker_color,
                        linestyle=":",
                        alpha=0.3,
                        linewidth=0.9,
                    )

        axis.set_title(f"{channel_name}: low-frequency spectrum and theoretical markers")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Median PSD (raw units squared / Hz)")
        axis.set_xlim(0, LOW_FREQUENCY_PLOT_MAX_HZ)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize="x-small", ncols=2)

    figure.suptitle(
        f"{bearing_id}: {BEARING_MODEL_SCENARIO} scenario at "
        f"{BEARING_SPEED_RPM:.0f} rpm (solid=fundamental, dotted=2x/3x)"
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return _frequency_peak_rows(
        bearing_id, groups, frequencies, psd_by_group
    )


def analyze_raw_spectra(
    data_dir: Path,
    processed_dir: Path,
    output_dir: Path,
    bearing_id: str = "Bearing1_1",
    objective: str = "balanced_f1",
) -> tuple[Path, Path, Path, Path]:
    suffix = f"h20_{objective}"
    timeline_path = processed_dir / f"femto_alarm_timeline_{suffix}.csv"
    episodes_path = processed_dir / f"femto_alarm_episodes_{suffix}.csv"
    validation_path = processed_dir / "femto_nested_h20_validation_results.json"
    for path in (timeline_path, episodes_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(f"required analysis input not found: {path}")

    timeline_rows = [
        row for row in _read_csv(timeline_path) if row["bearing_id"] == bearing_id
    ]
    episode_rows = [
        row for row in _read_csv(episodes_path) if row["bearing_id"] == bearing_id
    ]
    if not timeline_rows:
        raise ValueError(f"no timeline rows found for {bearing_id} in {timeline_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    horizon = int(validation["horizon_acquisitions"])
    groups = _select_sample_groups(timeline_rows, episode_rows, horizon)

    all_files = discover_bearing_files(data_dir)
    if bearing_id not in all_files:
        raise ValueError(f"{bearing_id} not found under {data_dir}")
    files_by_index = {sample_id: path for sample_id, path in all_files[bearing_id]}
    if set(files_by_index) != {int(row["sample_index"]) for row in timeline_rows}:
        raise ValueError(
            f"raw acquisitions and timeline sample IDs do not match for {bearing_id}"
        )

    (
        frequencies,
        psd_by_group,
        coherence_frequencies,
        coherence_by_group,
        band_rows,
    ) = _analyze_groups(groups, files_by_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"femto_{bearing_id}_raw_spectra_{suffix}.png"
    summary_path = output_dir / f"femto_{bearing_id}_raw_spectra_{suffix}_bands.csv"
    frequency_plot_path = (
        output_dir / f"femto_{bearing_id}_bearing_frequency_scenario_{suffix}.png"
    )
    frequency_summary_path = (
        output_dir / f"femto_{bearing_id}_bearing_frequency_scenario_{suffix}.csv"
    )
    _plot_results(
        plot_path,
        bearing_id,
        horizon,
        groups,
        frequencies,
        psd_by_group,
        coherence_frequencies,
        coherence_by_group,
    )
    frequency_rows = _plot_characteristic_frequencies(
        frequency_plot_path,
        bearing_id,
        groups,
        frequencies,
        psd_by_group,
    )

    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "group",
                "channel",
                "frequency_band",
                "sample_count",
                "median_band_power",
            ],
        )
        writer.writeheader()
        writer.writerows(band_rows)
    os.replace(temporary_path, summary_path)

    temporary_path = frequency_summary_path.with_suffix(
        frequency_summary_path.suffix + ".tmp"
    )
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "bearing_id",
                "model_scenario",
                "group",
                "channel",
                "frequency_name",
                "harmonic_order",
                "theoretical_frequency_hz",
                "nearest_bin_frequency_hz",
                "nearest_bin_psd",
                "peak_within_one_bin_hz",
                "peak_offset_hz",
                "reference_nearest_bin_psd",
                "psd_ratio_vs_far_reference",
                "sample_count",
                "fft_resolution_hz",
            ],
        )
        writer.writeheader()
        writer.writerows(frequency_rows)
    os.replace(temporary_path, frequency_summary_path)
    return plot_path, summary_path, frequency_plot_path, frequency_summary_path


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    processed_dir = (
        args.processed_dir.expanduser().resolve()
        if args.processed_dir
        else data_dir.parent.parent / "processed"
    )
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else processed_dir
    )
    plot_path, summary_path, frequency_plot_path, frequency_summary_path = analyze_raw_spectra(
        data_dir,
        processed_dir,
        output_dir,
        args.bearing,
        args.objective,
    )
    print(f"Spettri: {plot_path}")
    print(f"Potenza per banda: {summary_path}")
    print(f"Frequenze caratteristiche (scenario): {frequency_plot_path}")
    print(f"Picchi locali e frequenze teoriche: {frequency_summary_path}")


if __name__ == "__main__":
    main()
