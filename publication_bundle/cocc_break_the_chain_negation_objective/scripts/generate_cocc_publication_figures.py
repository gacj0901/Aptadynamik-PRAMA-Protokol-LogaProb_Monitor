#!/usr/bin/env python
"""Generate publication comparison figures for CoCC PRAMA artifacts.

This script is offline-only: it reads existing raw.json files, reconstructs
sliding token windows, measures PRAMA ProbLog Components over those windows,
and writes the three inter-model comparison PNGs used by the publication
bundle. It never calls APIs, runs models, or modifies raw inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

import matplotlib.pyplot as plt  # noqa: E402

from aptadynamik.prama_problog_components import measure  # noqa: E402
from scripts.analyze_cocc_prama_windows import sliding_window_turns  # noqa: E402

DEFAULT_GPT41_SESSIONS = Path("results/cocc_prama_gpt41_negation_objective_n235/sessions")
DEFAULT_DEEPSEEK_SESSIONS = Path("results/cocc_prama_deepseek_negation_objective_n235/sessions")
DEFAULT_OUTPUT_DIR = Path("publication_bundle/figures")

FIGURE_NAMES = {
    "entropy": "model_comparison_entropy_by_difficulty.png",
    "xi_norm": "model_comparison_xi_by_difficulty.png",
    "viability_margin": "model_comparison_viability_margin_by_difficulty.png",
}


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def first_turn(raw: dict[str, Any]) -> dict[str, Any]:
    turns = raw.get("turns") or []
    return turns[0] if turns and isinstance(turns[0], dict) else {}


def token_entropy(token: dict[str, Any]) -> float | None:
    return finite_float(token.get("entropy"))


def read_trajectory(raw_path: Path, window_size: int, stride: int) -> dict[str, Any] | None:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    turn = first_turn(raw)
    tokens = [tok for tok in (turn.get("tokens") or []) if isinstance(tok, dict)]
    windows = sliding_window_turns(tokens, window_size=window_size, stride=stride)
    if not windows:
        return None
    prama = measure(
        windows,
        calib_window=1,
        min_turns_for_regime=3,
        min_windows_for_regime=12,
        crossing_index_scope="token_window",
    )
    rows = prama.get("turns") or []
    entropy_series: list[float] = []
    xi_series: list[float] = []
    margin_series: list[float] = []
    for index, row in enumerate(rows):
        window_tokens = windows[index].get("tokens") or [] if index < len(windows) else []
        entropies = [value for value in (token_entropy(tok) for tok in window_tokens) if value is not None]
        entropy_series.append(statistics.fmean(entropies) if entropies else math.nan)
        xi_series.append(finite_float(row.get("xi_norm")) if finite_float(row.get("xi_norm")) is not None else math.nan)
        margin_series.append(
            finite_float(row.get("viability_margin"))
            if finite_float(row.get("viability_margin")) is not None
            else math.nan
        )
    if not entropy_series:
        return None
    return {
        "session_id": raw.get("session_id"),
        "difficulty": metadata.get("difficulty") or raw.get("difficulty") or "unknown",
        "entropy": entropy_series,
        "xi_norm": xi_series,
        "viability_margin": margin_series,
    }


def normalized_bins(values: list[float], bins: int) -> list[float]:
    if not values:
        return [math.nan] * bins
    if len(values) == 1:
        return [values[0]] * bins
    out: list[float] = []
    last = len(values) - 1
    for idx in range(bins):
        pos = (idx / (bins - 1)) * last if bins > 1 else 0.0
        lo = int(math.floor(pos))
        hi = min(last, lo + 1)
        frac = pos - lo
        left = values[lo]
        right = values[hi]
        if math.isnan(left) and math.isnan(right):
            out.append(math.nan)
        elif math.isnan(left):
            out.append(right)
        elif math.isnan(right):
            out.append(left)
        else:
            out.append(left * (1.0 - frac) + right * frac)
    return out


def mean_by_bin(series_list: list[list[float]], bins: int) -> list[float]:
    normalized = [normalized_bins(series, bins) for series in series_list]
    means: list[float] = []
    for idx in range(bins):
        values = [series[idx] for series in normalized if not math.isnan(series[idx])]
        means.append(statistics.fmean(values) if values else math.nan)
    return means


def load_grouped(gpt41_sessions: Path, deepseek_sessions: Path, window_size: int, stride: int) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    sources = [("GPT-4.1", gpt41_sessions), ("DeepSeek", deepseek_sessions)]
    for model_name, sessions_dir in sources:
        for raw_path in sorted(sessions_dir.glob("*/raw.json")):
            trajectory = read_trajectory(raw_path, window_size=window_size, stride=stride)
            if trajectory is None:
                continue
            grouped[(model_name, str(trajectory["difficulty"]))].append(trajectory)
    return grouped


def plot_metric(grouped: dict[tuple[str, str], list[dict[str, Any]]], metric: str, output_path: Path, bins: int) -> None:
    difficulties = ["easy", "medium", "hard", "unknown"]
    models = ["GPT-4.1", "DeepSeek"]
    colors = {"GPT-4.1": "#364153", "DeepSeek": "#A23B2A"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    plotted = False
    for ax, difficulty in zip(axes, difficulties[:3]):
        for model in models:
            trajectories = grouped.get((model, difficulty), [])
            if not trajectories:
                continue
            avg = mean_by_bin([trajectory[metric] for trajectory in trajectories], bins=bins)
            ax.plot(range(bins), avg, label=f"{model} (n={len(trajectories)})", color=colors[model], linewidth=2)
            plotted = True
        ax.set_title(difficulty)
        ax.set_xlabel("normalized trajectory progress")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(metric)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=2)
    fig.suptitle(f"CoCC negation_objective: {metric} trajectories by difficulty")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not plotted:
        raise SystemExit(f"no trajectories available for {metric}")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_figures(
    gpt41_sessions: Path,
    deepseek_sessions: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    bins: int,
) -> list[Path]:
    grouped = load_grouped(gpt41_sessions, deepseek_sessions, window_size=window_size, stride=stride)
    outputs: list[Path] = []
    for metric, filename in FIGURE_NAMES.items():
        output_path = output_dir / filename
        plot_metric(grouped, metric, output_path, bins=bins)
        outputs.append(output_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CoCC PRAMA publication comparison figures.")
    parser.add_argument("--gpt41-sessions-dir", type=Path, default=DEFAULT_GPT41_SESSIONS)
    parser.add_argument("--deepseek-sessions-dir", type=Path, default=DEFAULT_DEEPSEEK_SESSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--bins", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = generate_figures(
        gpt41_sessions=args.gpt41_sessions_dir,
        deepseek_sessions=args.deepseek_sessions_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        bins=args.bins,
    )
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
