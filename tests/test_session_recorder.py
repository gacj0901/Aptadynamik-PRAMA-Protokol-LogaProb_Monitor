import json
import os
import re
import time
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from aptadynamik.observer.report_writer import ReportWriter, sanitize_model_id
from aptadynamik.observer.session_recorder import SessionRecorder
import scripts.prama_chat_server as chat_server


def sample_windows():
    return [
        {
            "window_index": 0,
            "entropy_raw": 0.5,
            "entropy_norm": 0.25,
            "gap_norm": 0.6,
            "rigidity": 0.45,
            "uncertainty": 0.1,
            "entropy_std": 0.03,
            "entropy_range": 0.08,
            "n_tokens_in_window": 2,
        }
    ]


def recorder_with_turn(model="test/model:fixture v1"):
    recorder = SessionRecorder.create(model=model, session_id="test-session-abc123")
    tokens = [{"token": "hello", "top_logprobs": [-0.1, -0.5]}]
    recorder.append_turn("Hi", "Hello there", tokens, sample_windows(), finish_reason="stop")
    return recorder


def test_session_recorder_stores_turn_and_live_summary():
    recorder = SessionRecorder.create(model="test-model-fixture", session_id="test-session-abc123")
    tokens = [
        {"token": "hello", "top_logprobs": [-0.1, -0.5]},
        {"token": "world", "top_logprobs": [-0.2, -0.8]},
    ]

    recorder.append_turn("Hi", "Hello world", tokens, sample_windows())
    summary = recorder.live_summary()

    assert summary["session_id"] == "test-session-abc123"
    assert summary["turn_count"] == 1
    assert summary["token_count"] == 2
    assert summary["total_tokens"] == 2
    assert summary["window_count"] == 1
    assert summary["total_windows"] == 1
    assert summary["avg_rigidity"] == 0.45


def test_sanitize_model_id_replaces_invalid_filename_characters():
    assert sanitize_model_id("openai/gpt:4o mini!") == "openai-gpt-4o-mini"
    assert sanitize_model_id("gpt_4o-mini.2026") == "gpt_4o-mini.2026"


def test_report_writer_creates_dedicated_output_folder_and_files(tmp_path):
    recorder = recorder_with_turn()
    result = ReportWriter(tmp_path, max_tokens=512, top_logprobs=5, window_size=16).write(recorder)

    output_dir = Path(result["output_dir"])
    output_folder = result["output_folder"]
    files = result["files"]

    assert output_dir.parent == tmp_path
    assert output_dir.name == output_folder
    assert re.match(r"session_\d{8}-\d{4}h_test-model-fixture-v1(?:_\d+)?$", output_folder)
    assert all(Path(path).parent == output_dir for path in files.values())
    assert not any(path.is_file() for path in tmp_path.iterdir())

    expected_files = {
        "raw": "raw.json",
        "detail": "detail.csv",
        "summary": "summary.csv",
        "report": "report.md",
        "conversation_json": "conversation.json",
        "conversation_md": "conversation.md",
        "metadata": "metadata.json",
    }
    assert set(files) == set(expected_files)
    for key, filename in expected_files.items():
        assert Path(files[key]).name == filename
        assert Path(files[key]).exists()


def test_metadata_and_raw_store_output_folder_metadata(tmp_path):
    recorder = recorder_with_turn(model="gpt-4o-mini")
    result = ReportWriter(tmp_path, temperature=0.7, max_tokens=321, top_logprobs=7, window_size=12).write(recorder)
    output_dir = Path(result["output_dir"])

    metadata = json.loads(output_dir.joinpath("metadata.json").read_text(encoding="utf-8"))
    raw = json.loads(output_dir.joinpath("raw.json").read_text(encoding="utf-8"))

    assert metadata["session_id"] == "test-session-abc123"
    assert metadata["output_folder_name"] == output_dir.name
    assert metadata["output_dir"] == str(output_dir)
    assert metadata["model"] == "gpt-4o-mini"
    assert metadata["generated_at"]
    assert metadata["max_tokens"] == 321
    assert metadata["top_logprobs"] == 7
    assert metadata["window_size"] == 12
    assert raw["output_folder_name"] == output_dir.name
    assert raw["output_dir"] == str(output_dir)
    assert raw["generated_at"] == metadata["generated_at"]
    assert raw["turns"][0]["finish_reason"] == "stop"


def test_report_and_conversation_files_use_simple_names(tmp_path):
    recorder = recorder_with_turn(model="test-model-fixture")
    result = ReportWriter(tmp_path).write(recorder)
    output_dir = Path(result["output_dir"])

    conversation_md = output_dir.joinpath("conversation.md").read_text(encoding="utf-8")
    assert "Hi" in conversation_md
    assert "Hello there" in conversation_md
    assert "top_logprobs" not in conversation_md

    conversation_json = json.loads(output_dir.joinpath("conversation.json").read_text(encoding="utf-8"))
    assert conversation_json["session_id"] == "test-session-abc123"
    assert conversation_json["turns"][0]["finish_reason"] == "stop"
    assert conversation_json["turns"][0]["metrics_summary"]["token_count"] == 1
    assert "tokens" not in conversation_json["turns"][0]

    report = output_dir.joinpath("report.md").read_text(encoding="utf-8")
    assert "## Session Metadata" in report
    assert "Output folder" in report
    assert "Generated at" in report
    assert "metadata.json" in report
    assert "This report currently measures logprob-derived output geometry" in report


def test_report_writer_returns_frontend_compatible_file_map(tmp_path):
    result = ReportWriter(tmp_path).write(recorder_with_turn())

    assert set(result) == {"session_id", "output_folder", "output_dir", "files"}
    assert set(result["files"]) == {
        "raw",
        "detail",
        "summary",
        "report",
        "conversation_json",
        "conversation_md",
        "metadata",
    }


def test_closed_at_is_set_after_event_and_differs_from_created_at():
    recorder = SessionRecorder.create(model="test-model-fixture", session_id="test-session-time")
    recorder.append_turn("Hi", "Hello", [{"token": "Hello"}], sample_windows())
    time.sleep(0.01)
    recorder.stop()

    assert recorder.created_at != recorder.closed_at
    assert recorder.live_summary()["duration_seconds"] > 0


def test_total_tokens_equals_sum_of_per_turn_assistant_tokens():
    recorder = SessionRecorder.create(model="test-model-fixture", session_id="test-session-tokens")
    recorder.append_turn("A", "one two", [{"token": "one"}, {"token": "two"}], sample_windows())
    recorder.append_turn("B", "three", [{"token": "three"}], sample_windows())

    expected = sum(turn["token_count"] for turn in recorder.turns)
    assert expected == 3
    assert recorder.total_tokens() == expected
    assert recorder.live_summary()["total_tokens"] == expected


class TestPersistentAptadynamicRegime(unittest.TestCase):
    def test_report_writer_persists_aptadynamic_regime_section(self):
        recorder = recorder_with_turn(model="gpt-4o-mini")
        recorder.update_prama_regime_state(
            {
                "regime_label": "III_STRUCTURAL_PULSATION",
                "regime_description": "threshold crossing followed by recovery",
                "trajectory_assessment": "THRESHOLD_CROSSED_STRUCTURAL_PULSATION",
                "recovery_observed": True,
                "first_crossing_turn": 2,
                "threshold_crossing_ratio": 0.4,
                "persistent_crossing_ratio": 0.3,
                "post_crossing_recovery_turns": [3, 4],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = ReportWriter(tmp, max_tokens=512, top_logprobs=5, window_size=16).write(recorder)
            output_dir = Path(result["output_dir"])
            report = output_dir.joinpath("report.md").read_text(encoding="utf-8")
            raw = json.loads(output_dir.joinpath("raw.json").read_text(encoding="utf-8"))
            metadata = json.loads(output_dir.joinpath("metadata.json").read_text(encoding="utf-8"))

        self.assertIn("## Aptadynamic Regime", report)
        self.assertIn("regime_label: `III_STRUCTURAL_PULSATION`", report)
        self.assertIn("trajectory_assessment: `THRESHOLD_CROSSED_STRUCTURAL_PULSATION`", report)
        self.assertIn("Threshold crossing indicates loss of point-regime viability", report)
        self.assertEqual(raw["regime_label"], "III_STRUCTURAL_PULSATION")
        self.assertEqual(raw["summary"]["trajectory_assessment"], "THRESHOLD_CROSSED_STRUCTURAL_PULSATION")
        self.assertEqual(metadata["first_crossing_turn"], 2)
        self.assertEqual(metadata["post_crossing_recovery_turns"], [3, 4])

    def test_report_writer_handles_missing_aptadynamic_regime(self):
        recorder = recorder_with_turn(model="gpt-4o-mini")
        with tempfile.TemporaryDirectory() as tmp:
            result = ReportWriter(tmp).write(recorder)
            output_dir = Path(result["output_dir"])
            report = output_dir.joinpath("report.md").read_text(encoding="utf-8")
            raw = json.loads(output_dir.joinpath("raw.json").read_text(encoding="utf-8"))

        self.assertIn("# PRAMA Monitor Session Report", report)
        self.assertIn("No aptadynamic regime payload was recorded for this session.", report)
        self.assertIsNone(raw["regime_label"])
        self.assertEqual(raw["post_crossing_recovery_turns"], [])


class TestPramaChatServerPayload(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.pop("OPENAI_API_KEY", None)
        chat_server.SESSIONS.clear()
        self.client = TestClient(chat_server.app)

    def tearDown(self):
        chat_server.SESSIONS.clear()
        if self.previous_key is not None:
            os.environ["OPENAI_API_KEY"] = self.previous_key

    def start_session(self):
        response = self.client.post("/session/start")
        self.assertEqual(response.status_code, 200)
        return response.json()["session_id"]

    def chat_events(self, session_id, message="Hello"):
        response = self.client.post("/chat", json={"session_id": session_id, "user_message": message})
        self.assertEqual(response.status_code, 200)
        return [json.loads(line) for line in response.text.splitlines() if line.strip()]

    def test_chat_payload_includes_prama_v022_regime_and_legacy_fields(self):
        session_id = self.start_session()
        events = self.chat_events(session_id)
        prama_events = [event for event in events if event.get("type") in {"prama", "final_prama"}]
        self.assertTrue(prama_events)
        final = prama_events[-1]

        for field in ["micro", "macro", "viability"]:
            self.assertIn(field, final)
        for field in [
            "micro_raw",
            "micro_health",
            "macro_health",
            "activity_raw",
            "activity_structural",
            "activity_effective",
            "acople_effective",
            "delta_instant",
            "xi_norm",
            "lambda_remaining",
            "theta_dynamic",
            "viability_margin",
            "threshold_crossed",
            "xi_exceeds_theta",
            "boundary_side",
            "viability_status",
            "regime_label",
            "regime_description",
            "recovery_observed",
            "first_crossing_turn",
            "threshold_crossing_ratio",
            "persistent_crossing_ratio",
            "post_crossing_recovery_turns",
            "trajectory_assessment",
        ]:
            self.assertIn(field, final)

        summaries = [event for event in events if event.get("type") == "turn_summary"]
        self.assertTrue(summaries)
        self.assertIn("regime_label", summaries[-1]["session"])
        self.assertIn("trajectory_assessment", summaries[-1]["session"])

    def test_chat_payload_does_not_fail_without_prama_regime(self):
        original_call_dry = chat_server.call_dry

        def empty_call_dry(recorder, user_message):
            return "No token fixture.", [], "stop"

        chat_server.call_dry = empty_call_dry
        try:
            session_id = self.start_session()
            events = self.chat_events(session_id)
        finally:
            chat_server.call_dry = original_call_dry

        final = [event for event in events if event.get("type") == "final_prama"][-1]
        self.assertIn("micro", final)
        self.assertIn("macro", final)
        self.assertIn("viability", final)
        self.assertIn("regime_label", final)
        self.assertIsNone(final["regime_label"])
        self.assertEqual(final["boundary_side"], "UNRESOLVED")
