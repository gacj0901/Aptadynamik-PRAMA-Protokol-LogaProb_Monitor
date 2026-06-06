from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from aptadynamik.observer.session_recorder import SessionRecorder


DETAIL_FIELDS = [
    "session_id",
    "turn_index",
    "window_index",
    "entropy_raw",
    "entropy_norm",
    "gap_norm",
    "rigidity",
    "uncertainty",
    "entropy_std",
    "entropy_range",
    "n_tokens_in_window",
]

SUMMARY_FIELDS = [
    "session_id",
    "model",
    "status",
    "output_folder_name",
    "output_dir",
    "generated_at",
    "created_at",
    "closed_at",
    "duration_seconds",
    "turn_count",
    "total_tokens",
    "total_windows",
    "avg_entropy_raw",
    "avg_entropy_norm",
    "avg_gap_norm",
    "avg_rigidity",
    "avg_uncertainty",
    "max_entropy_std",
    "max_entropy_range",
]

FILE_NAMES = {
    "raw": "raw.json",
    "detail": "detail.csv",
    "summary": "summary.csv",
    "report": "report.md",
    "conversation_json": "conversation.json",
    "conversation_md": "conversation.md",
    "metadata": "metadata.json",
}


def sanitize_model_id(model: str) -> str:
    value = model.replace("/", "-").replace(":", "-").replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]", "", value)
    return value or "unknown-model"


def local_timestamp_for_folder(now: Optional[datetime] = None) -> str:
    current = now or datetime.now().astimezone()
    return current.strftime("%Y%m%d-%H%Mh")


class ReportWriter:
    def __init__(
        self,
        results_dir: str | Path = "results",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.window_size = window_size

    def write(self, recorder: SessionRecorder) -> Dict[str, object]:
        if recorder.status == "active":
            recorder.stop()

        generated_at = datetime.now().astimezone().isoformat()
        folder_name = f"session_{local_timestamp_for_folder()}_{sanitize_model_id(recorder.model)}"
        output_dir = self._unique_output_dir(folder_name)
        output_dir.mkdir(parents=True, exist_ok=False)

        recorder.generated_at = generated_at
        recorder.output_folder_name = output_dir.name
        recorder.output_dir = str(output_dir)

        paths = {key: output_dir / filename for key, filename in FILE_NAMES.items()}

        paths["raw"].write_text(json.dumps(recorder.to_dict(), indent=2), encoding="utf-8")
        self._write_detail(recorder, paths["detail"])
        self._write_summary(recorder, paths["summary"])
        self._write_markdown(recorder, paths["report"])
        self._write_conversation_json(recorder, paths["conversation_json"])
        self._write_conversation_markdown(recorder, paths["conversation_md"])
        paths["metadata"].write_text(json.dumps(self._metadata(recorder), indent=2), encoding="utf-8")

        return {
            "session_id": recorder.session_id,
            "output_folder": output_dir.name,
            "output_dir": str(output_dir),
            "files": {key: str(path) for key, path in paths.items()},
        }

    def _unique_output_dir(self, folder_name: str) -> Path:
        candidate = self.results_dir / folder_name
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            indexed = self.results_dir / f"{folder_name}_{index}"
            if not indexed.exists():
                return indexed
            index += 1

    def _metadata(self, recorder: SessionRecorder) -> Dict[str, object]:
        summary = recorder.live_summary()
        return {
            "session_id": recorder.session_id,
            "output_folder_name": recorder.output_folder_name,
            "output_dir": recorder.output_dir,
            "model": recorder.model,
            "generated_at": recorder.generated_at,
            "started_at": recorder.created_at,
            "stopped_at": recorder.closed_at,
            "duration_seconds": summary["duration_seconds"],
            "total_turns": summary["turn_count"],
            "total_tokens": summary["total_tokens"],
            "total_windows": summary["total_windows"],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_logprobs": self.top_logprobs,
            "window_size": self.window_size,
            "regime_label": summary.get("regime_label"),
            "regime_description": summary.get("regime_description"),
            "trajectory_assessment": summary.get("trajectory_assessment"),
            "recovery_observed": summary.get("recovery_observed"),
            "first_crossing_turn": summary.get("first_crossing_turn"),
            "threshold_crossing_ratio": summary.get("threshold_crossing_ratio"),
            "persistent_crossing_ratio": summary.get("persistent_crossing_ratio"),
            "post_crossing_recovery_turns": summary.get("post_crossing_recovery_turns", []),
        }

    def _write_detail(self, recorder: SessionRecorder, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
            writer.writeheader()
            for turn in recorder.turns:
                for window in turn["windows"]:
                    row = {
                        "session_id": recorder.session_id,
                        "turn_index": turn["turn_index"],
                        **window,
                    }
                    writer.writerow({field: row.get(field) for field in DETAIL_FIELDS})

    def _write_summary(self, recorder: SessionRecorder, path: Path) -> None:
        summary = recorder.live_summary()
        row = {field: summary.get(field) for field in SUMMARY_FIELDS}
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(row)

    def _write_markdown(self, recorder: SessionRecorder, path: Path) -> None:
        summary = recorder.live_summary()
        files = {
            "raw.json": "raw.json",
            "detail.csv": "detail.csv",
            "summary.csv": "summary.csv",
            "report.md": "report.md",
            "conversation.json": "conversation.json",
            "conversation.md": "conversation.md",
            "metadata.json": "metadata.json",
        }
        lines = [
            "# PRAMA Monitor Session Report",
            "",
            "## Session Metadata",
            "",
            f"- Session ID: `{recorder.session_id}`",
            f"- Output folder: `{recorder.output_folder_name}`",
            f"- Model: `{recorder.model}`",
            f"- Generated at: `{recorder.generated_at}`",
            f"- Status: `{recorder.status}`",
            f"- Started: `{recorder.created_at}`",
            f"- Stopped: `{recorder.closed_at or ''}`",
            f"- Duration seconds: `{summary['duration_seconds']}`",
            f"- Turns: `{summary['turn_count']}`",
            f"- Total assistant tokens: `{summary['total_tokens']}`",
            f"- Total windows: `{summary['total_windows']}`",
            "",
            "## Aggregate Geometry",
            "",
            f"- Average entropy norm: `{summary['avg_entropy_norm']}`",
            f"- Average gap norm: `{summary['avg_gap_norm']}`",
            f"- Average rigidity: `{summary['avg_rigidity']}`",
            f"- Average uncertainty: `{summary['avg_uncertainty']}`",
            f"- Max entropy std: `{summary['max_entropy_std']}`",
            f"- Max entropy range: `{summary['max_entropy_range']}`",
            "",
            "## Conversation Summary",
            "",
            "| turn | user chars | assistant tokens | windows | avg entropy norm | avg rigidity | avg uncertainty |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for turn in recorder.turns:
            turn_summary = turn["summary"]
            lines.append(
                f"| {turn['turn_index'] + 1} "
                f"| {len(turn['user_message'])} "
                f"| {turn['token_count']} "
                f"| {len(turn['windows'])} "
                f"| {turn_summary.get('avg_entropy_norm', 0.0)} "
                f"| {turn_summary.get('avg_rigidity', 0.0)} "
                f"| {turn_summary.get('avg_uncertainty', 0.0)} |"
            )
        regime_lines = self._aptadynamic_regime_lines(summary)
        lines.extend([
            "",
            *regime_lines,
            "",
            "## Turns",
            "",
        ])
        for turn in recorder.turns:
            turn_summary = turn["summary"]
            lines.extend(
                [
                    f"### Turn {turn['turn_index'] + 1}",
                    "",
                    "#### User message",
                    "",
                    turn["user_message"],
                    "",
                    "#### Assistant message",
                    "",
                    turn["assistant_message"],
                    "",
                    "#### Metrics",
                    "",
                    f"- tokens: `{turn['token_count']}`",
                    f"- finish_reason: `{turn.get('finish_reason') or ''}`",
                    f"- windows: `{len(turn['windows'])}`",
                    f"- avg_entropy_norm: `{turn_summary.get('avg_entropy_norm', 0.0)}`",
                    f"- avg_gap_norm: `{turn_summary.get('avg_gap_norm', 0.0)}`",
                    f"- avg_rigidity: `{turn_summary.get('avg_rigidity', 0.0)}`",
                    f"- avg_uncertainty: `{turn_summary.get('avg_uncertainty', 0.0)}`",
                    f"- entropy_std: `{turn_summary.get('max_entropy_std', 0.0)}`",
                    f"- entropy_range: `{turn_summary.get('max_entropy_range', 0.0)}`",
                    "",
                ]
            )
        lines.extend([
            "## Generated Files",
            "",
        ])
        for label, filename in files.items():
            lines.append(f"- {label}: `{filename}`")
        lines.extend([
            "",
            "## Methodological Note",
            "",
            "This report currently measures logprob-derived output geometry and conversational trajectory metrics. It does not yet implement predictive-surprise Delta over exogenous user signs.",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")

    def _aptadynamic_regime_lines(self, summary: Dict[str, object]) -> list[str]:
        has_regime = any(
            summary.get(key) is not None
            for key in (
                "regime_label",
                "trajectory_assessment",
                "recovery_observed",
                "first_crossing_turn",
                "threshold_crossing_ratio",
                "persistent_crossing_ratio",
            )
        ) or bool(summary.get("post_crossing_recovery_turns"))
        lines = [
            "## Aptadynamic Regime",
            "",
        ]
        if not has_regime:
            lines.extend(
                [
                    "No aptadynamic regime payload was recorded for this session.",
                    "",
                ]
            )
            return lines
        lines.extend(
            [
                f"- regime_label: `{summary.get('regime_label')}`",
                f"- trajectory_assessment: `{summary.get('trajectory_assessment')}`",
                f"- recovery_observed: `{summary.get('recovery_observed')}`",
                f"- first_crossing_turn: `{summary.get('first_crossing_turn')}`",
                f"- threshold_crossing_ratio: `{summary.get('threshold_crossing_ratio')}`",
                f"- persistent_crossing_ratio: `{summary.get('persistent_crossing_ratio')}`",
                f"- post_crossing_recovery_turns: `{summary.get('post_crossing_recovery_turns', [])}`",
                "",
                "Threshold crossing indicates loss of point-regime viability; terminal collapse requires persistence without recovery.",
                "",
            ]
        )
        return lines

    def _compact_turn_metrics(self, turn):
        summary = turn.get("summary", {})
        return {
            "token_count": turn.get("token_count", 0),
            "finish_reason": turn.get("finish_reason"),
            "avg_entropy_norm": summary.get("avg_entropy_norm", 0.0),
            "avg_gap_norm": summary.get("avg_gap_norm", 0.0),
            "avg_rigidity": summary.get("avg_rigidity", 0.0),
            "avg_uncertainty": summary.get("avg_uncertainty", 0.0),
            "entropy_std": summary.get("max_entropy_std", 0.0),
            "entropy_range": summary.get("max_entropy_range", 0.0),
        }

    def _write_conversation_markdown(self, recorder: SessionRecorder, path: Path) -> None:
        lines = [
            "# PRAMA Conversation Transcript",
            "",
            f"- Session: `{recorder.session_id}`",
            f"- Model: `{recorder.model}`",
            f"- Start time: `{recorder.created_at}`",
            f"- End time: `{recorder.closed_at or ''}`",
            f"- Total turns: `{len(recorder.turns)}`",
            "",
        ]

        for turn in recorder.turns:
            metrics = self._compact_turn_metrics(turn)
            lines.extend(
                [
                    f"## Turn {turn['turn_index'] + 1}",
                    "",
                    f"- token_count: `{metrics['token_count']}`",
                    f"- finish_reason: `{metrics['finish_reason'] or ''}`",
                    f"- avg_entropy_norm: `{metrics['avg_entropy_norm']}`",
                    f"- avg_rigidity: `{metrics['avg_rigidity']}`",
                    f"- avg_uncertainty: `{metrics['avg_uncertainty']}`",
                    f"- entropy_range: `{metrics['entropy_range']}`",
                    "",
                    "### User",
                    "",
                    turn["user_message"],
                    "",
                    "### Assistant",
                    "",
                    turn["assistant_message"],
                    "",
                ]
            )

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_conversation_json(self, recorder: SessionRecorder, path: Path) -> None:
        payload = {
            "session_id": recorder.session_id,
            "model": recorder.model,
            "created_at": recorder.created_at,
            "closed_at": recorder.closed_at,
            "turns": [
                {
                    "turn_index": turn["turn_index"],
                    "timestamp": turn["timestamp"],
                    "user_message": turn["user_message"],
                    "assistant_message": turn["assistant_message"],
                    "finish_reason": turn.get("finish_reason"),
                    "token_count": turn["token_count"],
                    "metrics_summary": self._compact_turn_metrics(turn),
                }
                for turn in recorder.turns
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
