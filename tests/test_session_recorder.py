import json
import time
from pathlib import Path

from aptadynamik.observer.report_writer import ReportWriter
from aptadynamik.observer.session_recorder import SessionRecorder


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


def test_report_writer_creates_reports_and_transcripts():
    recorder = SessionRecorder.create(model="test-model-fixture", session_id="test-session-abc123")
    tokens = [{"token": "hello", "top_logprobs": [-0.1, -0.5]}]
    recorder.append_turn("Hi", "Hello there", tokens, sample_windows())

    output_dir = Path("results") / "test-session-recorder"
    files = ReportWriter(output_dir).write(recorder)

    assert set(files) == {
        "conversation_markdown",
        "conversation_json",
        "prama_report",
        "detail_csv",
        "summary_csv",
        "raw_json",
    }
    for path in files.values():
        assert Path(path).exists()

    conversation_md = output_dir.joinpath("session_test-session-abc123_conversation.md").read_text(encoding="utf-8")
    assert "Hi" in conversation_md
    assert "Hello there" in conversation_md
    assert "top_logprobs" not in conversation_md

    conversation_json = json.loads(
        output_dir.joinpath("session_test-session-abc123_conversation.json").read_text(encoding="utf-8")
    )
    assert conversation_json["session_id"] == "test-session-abc123"
    assert conversation_json["turns"][0]["metrics_summary"]["token_count"] == 1
    assert "tokens" not in conversation_json["turns"][0]

    report = output_dir.joinpath("session_test-session-abc123_report.md").read_text(encoding="utf-8")
    assert "## Session Metadata" in report
    assert "## Aggregate Geometry" in report
    assert "## Conversation Summary" in report
    assert "## Generated Files" in report
    assert "Duration seconds" in report
    assert "This report currently measures logprob-derived output geometry" in report

    raw = json.loads(output_dir.joinpath("session_test-session-abc123_raw.json").read_text(encoding="utf-8"))
    assert raw["turns"][0]["tokens"][0]["top_logprobs"] == [-0.1, -0.5]


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
