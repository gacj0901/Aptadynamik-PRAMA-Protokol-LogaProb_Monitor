import json
from pathlib import Path

import pytest

from aptadynamik.observer.qualitative_signatures import (
    detect_discontinuity,
    detect_structural_target_effect,
    evaluate_series,
    load_sessions_from_path,
    session_to_observed_series,
    smooth_synthetic_rows,
    synthetic_validation,
)


def test_synthetic_fold_triggers_required_signatures_and_smooth_does_not():
    validation = synthetic_validation()
    fold = {result.signature: result.triggered for result in validation["fold"]}
    smooth = {result.signature: result.triggered for result in validation["smooth"]}

    assert fold["discontinuity"]
    assert fold["hysteresis"]
    assert fold["critical_slowing"]
    assert not smooth["discontinuity"]
    assert not smooth["hysteresis"]
    assert not smooth["critical_slowing"]


def test_viability_proxy_from_raw_session_window():
    session = {
        "session_id": "test-session",
        "model": "test-model",
        "turns": [
            {
                "turn_index": 0,
                "user_message": "ordinary peripheral prompt",
                "windows": [
                    {
                        "window_index": 0,
                        "rigidity": 0.7,
                        "uncertainty": 0.2,
                    }
                ],
            }
        ],
    }

    rows = session_to_observed_series(session)

    assert rows[0]["viability"] == pytest.approx(0.5)
    assert rows[0]["target"] == "peripheral"


def test_discontinuity_detects_large_observed_jump():
    result = detect_discontinuity([0.9, 0.88, 0.2, 0.18])

    assert result.triggered


def test_structural_target_effect_compares_peripheral_and_constitutive_rows():
    rows = [
        {"target": "peripheral", "viability": 0.8},
        {"target": "peripheral", "viability": 0.7},
        {"target": "constitutive", "viability": 0.3},
        {"target": "constitutive", "viability": 0.4},
    ]

    result = detect_structural_target_effect(rows)

    assert result.triggered
    assert result.score > 0


def test_load_sessions_from_path_reads_monitor_raw_json(tmp_path):
    payload = {
        "session_id": "test-session",
        "model": "model-a",
        "turns": [],
    }
    raw_path = tmp_path / "session_test_raw.json"
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    sessions = load_sessions_from_path(tmp_path)

    assert sessions[0]["session_id"] == "test-session"


def test_smooth_series_has_no_phase_transition_signatures():
    results = {result.signature: result.triggered for result in evaluate_series(smooth_synthetic_rows())}

    assert not results["discontinuity"]
    assert not results["hysteresis"]
    assert not results["critical_slowing"]
