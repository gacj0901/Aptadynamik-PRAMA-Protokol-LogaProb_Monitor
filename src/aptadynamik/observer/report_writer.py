from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

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


class ReportWriter:
    def __init__(self, results_dir: str | Path = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def write(self, recorder: SessionRecorder) -> Dict[str, str]:
        if recorder.status == "active":
            recorder.stop()

        session_id = recorder.session_id
        raw_path = self.results_dir / f"session_{session_id}_raw.json"
        detail_path = self.results_dir / f"session_{session_id}_detail.csv"
        summary_path = self.results_dir / f"session_{session_id}_summary.csv"
        report_path = self.results_dir / f"session_{session_id}_report.md"
        conversation_md_path = self.results_dir / f"session_{session_id}_conversation.md"
        conversation_json_path = self.results_dir / f"session_{session_id}_conversation.json"

        raw_path.write_text(json.dumps(recorder.to_dict(), indent=2), encoding="utf-8")
        self._write_detail(recorder, detail_path)
        self._write_summary(recorder, summary_path)
        self._write_markdown(recorder, report_path)
        self._write_conversation_markdown(recorder, conversation_md_path)
        self._write_conversation_json(recorder, conversation_json_path)

        return {
            "conversation_markdown": str(conversation_md_path),
            "conversation_json": str(conversation_json_path),
            "prama_report": str(report_path),
            "detail_csv": str(detail_path),
            "summary_csv": str(summary_path),
            "raw_json": str(raw_path),
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
            "conversation.md": f"session_{recorder.session_id}_conversation.md",
            "conversation.json": f"session_{recorder.session_id}_conversation.json",
            "raw.json": f"session_{recorder.session_id}_raw.json",
            "detail.csv": f"session_{recorder.session_id}_detail.csv",
            "summary.csv": f"session_{recorder.session_id}_summary.csv",
            "report.md": f"session_{recorder.session_id}_report.md",
        }
        lines = [
            "# PRAMA Monitor Session Report",
            "",
            "## Session Metadata",
            "",
            f"- Session ID: `{recorder.session_id}`",
            f"- Model: `{recorder.model}`",
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
        lines.extend([
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

    def _compact_turn_metrics(self, turn):
        summary = turn.get("summary", {})
        return {
            "token_count": turn.get("token_count", 0),
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
                    "token_count": turn["token_count"],
                    "metrics_summary": self._compact_turn_metrics(turn),
                }
                for turn in recorder.turns
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
