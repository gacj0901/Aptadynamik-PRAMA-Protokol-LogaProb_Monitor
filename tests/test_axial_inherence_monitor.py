import importlib.util
from pathlib import Path

import pytest

from aptadynamik.observer.axial_inherence_adapters import MockAxialInherenceAdapters
from aptadynamik.observer.axial_inherence_monitor import (
    bootstrap_ci,
    compute_iota,
    dominances,
    fatigue_series,
    precedence_lead,
    rotational_mobility,
    run_axial_inherence_session,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "prama_axial_inherence_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_axial_inherence_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_compute_iota_returns_one_with_no_history():
    assert compute_iota([1.0, 0.0], []) == 1.0


def test_compute_iota_decreases_for_similar_history():
    iota = compute_iota([1.0, 0.0], [[0.99, 0.01]])

    assert iota < 0.05


def test_dominances_compute_i_k_r():
    assert dominances(0.4, 0.25, 0.7) == pytest.approx((0.4, 0.75, 0.7))


def test_rotational_mobility_high_for_rotating_sequence():
    seq = ["i", "k", "r"] * 6

    assert rotational_mobility(seq)[-1] > 0.8


def test_rotational_mobility_low_for_stagnant_sequence_after_warmup():
    seq = ["k"] * 18

    assert rotational_mobility(seq)[-1] < 0.2


def test_fatigue_series_rises_when_mobility_collapses():
    rotating = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] * 4
    stagnant = [(0, 1, 0)] * 12
    fatigue = fatigue_series(rotating + stagnant)

    assert fatigue[-1] > fatigue[10]


def test_precedence_lead_positive_when_fatigue_crosses_before_function_drop():
    ikr = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] * 3 + [(0, 1, 0)] * 16
    fn = [0.95] * 18 + [0.2] * 7

    assert precedence_lead(ikr, fn) > 0


def test_bootstrap_ci_returns_mean_lo_hi_n():
    result = bootstrap_ci([1, 2, 3], n=200)

    assert set(result) == {"mean", "lo", "hi", "n"}
    assert result["n"] == 3
    assert result["lo"] <= result["mean"] <= result["hi"]


def test_axial_inherence_report_includes_required_language():
    report = RUNNER.axial_inherence_report_markdown(mode="test", sessions=[])

    assert "PRAMA Axial Inherence Monitor Report" in report
    assert "Axial inherence refers to the operational rotation" in report
    assert "La inherencia axial refiere a la rotación operacional" in report
    assert "Exogenous Judge Constraint" in report
    assert "lead = t(function drops) - t(rotation stagnates)" in report
    assert "Methodological Note" in report


def test_mock_axial_inherence_adapters_selftest_has_higher_echo_fatigue():
    task = {"task_id": "mock", "prompt": "continue"}
    normal = run_axial_inherence_session(MockAxialInherenceAdapters(), task, condition="rotating", max_turns=36)
    echo = run_axial_inherence_session(MockAxialInherenceAdapters(), task, condition="stagnated_echo_decline", max_turns=36)

    assert echo["fatigue"][-1] > normal["fatigue"][-1]
