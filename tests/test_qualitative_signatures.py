import json
import importlib.util
from pathlib import Path

import pytest

from aptadynamik.observer.qualitative_signatures import (
    sig_discontinuity,
    sig_hysteresis,
    synthetic_smooth_series,
    synthetic_validation,
    viability_from_turn,
)


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prama_phase_signature_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_phase_signature_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)
run_from_raw = RUNNER.run_from_raw


def test_synthetic_fold_triggers_discontinuity():
    validation = synthetic_validation()

    assert validation["fold"]["discontinuity"]["triggered"]


def test_synthetic_smooth_does_not_trigger_discontinuity():
    validation = synthetic_validation()

    assert not validation["smooth"]["discontinuity"]["triggered"]


def test_viability_from_turn_computes_rigidity_minus_uncertainty():
    turn = {
        "turn_index": 2,
        "token_count": 17,
        "summary": {
            "avg_rigidity": 0.72,
            "avg_uncertainty": 0.21,
            "avg_entropy_norm": 0.35,
            "max_entropy_range": 0.44,
            "max_entropy_std": 0.12,
        },
    }

    result = viability_from_turn(turn)

    assert result["turn_index"] == 2
    assert result["viability"] == pytest.approx(0.51)
    assert result["entropy_range"] == pytest.approx(0.44)
    assert result["entropy_std"] == pytest.approx(0.12)


def test_viability_from_turn_fails_clearly_on_missing_summary_field():
    turn = {"turn_index": 0, "token_count": 1, "summary": {"avg_rigidity": 0.5}}

    with pytest.raises(ValueError, match="avg_uncertainty"):
        viability_from_turn(turn)


def test_sig_discontinuity_detects_sharp_drop():
    result = sig_discontinuity([0.91, 0.88, 0.84, 0.22, 0.20])

    assert result["triggered"]
    assert result["strongest_transition_turn"] == 3


def test_sig_hysteresis_returns_positive_area_for_separated_branches():
    result = sig_hysteresis([0.9, 0.82, 0.74, 0.35], [0.50, 0.55, 0.60, 0.65])

    assert result["triggered"]
    assert result["hysteresis_area"] > 0


def test_loading_minimal_raw_json_produces_phase_signatures_csv(tmp_path):
    raw = {
        "session_id": "minimal",
        "model": "test-model",
        "turns": [
            {
                "turn_index": 0,
                "token_count": 10,
                "summary": {
                    "avg_rigidity": 0.8,
                    "avg_uncertainty": 0.1,
                    "avg_entropy_norm": 0.2,
                    "max_entropy_range": 0.1,
                    "max_entropy_std": 0.02,
                },
            },
            {
                "turn_index": 1,
                "token_count": 11,
                "summary": {
                    "avg_rigidity": 0.76,
                    "avg_uncertainty": 0.12,
                    "avg_entropy_norm": 0.25,
                    "max_entropy_range": 0.14,
                    "max_entropy_std": 0.03,
                },
            },
            {
                "turn_index": 2,
                "token_count": 12,
                "summary": {
                    "avg_rigidity": 0.35,
                    "avg_uncertainty": 0.4,
                    "avg_entropy_norm": 0.55,
                    "max_entropy_range": 0.51,
                    "max_entropy_std": 0.18,
                },
            },
        ],
    }
    raw_path = tmp_path / "session_minimal_raw.json"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    run_from_raw(raw_path, tmp_path / "phase_analysis_minimal")

    assert (tmp_path / "phase_analysis_minimal" / "phase_signatures.csv").exists()


def test_report_includes_required_language(tmp_path):
    raw = {
        "session_id": "report-test",
        "model": "test-model",
        "turns": [
            {
                "turn_index": 0,
                "token_count": 8,
                "summary": {
                    "avg_rigidity": 0.7,
                    "avg_uncertainty": 0.2,
                    "avg_entropy_norm": 0.3,
                    "max_entropy_range": 0.2,
                    "max_entropy_std": 0.05,
                },
            }
        ],
    }
    raw_path = tmp_path / "session_report_raw.json"
    output_dir = tmp_path / "phase_report"
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    run_from_raw(raw_path, output_dir)
    report = (output_dir / "phase_report.md").read_text(encoding="utf-8")

    assert "PRAMA Phase Signature Report" in report
    assert "viability = avg_rigidity - avg_uncertainty" in report
    assert "Methodological Note" in report


def test_smooth_series_has_no_discontinuity():
    smooth = synthetic_smooth_series()

    assert not sig_discontinuity(smooth["up"])["triggered"]
