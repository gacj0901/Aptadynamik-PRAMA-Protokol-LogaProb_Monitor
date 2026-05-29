import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "onset_experiment.py"
SPEC = importlib.util.spec_from_file_location("onset_experiment", SCRIPT_PATH)
onset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(onset)


def test_heavy_tail_xi_is_monotonic_for_constant_positive_delta():
    xi = onset.reconstruct_xi_lm([0.5, 0.5, 0.5, 0.5], beta=0.6)

    assert all(b > a for a, b in zip(xi, xi[1:]))


def test_step_psi_produces_larger_post_delta_than_pre_delta():
    pre_delta = abs(0.45 - onset.psi_for("contradictory_onset", "pre"))
    post_delta = abs(0.45 - onset.psi_for("contradictory_onset", "post"))

    assert post_delta > pre_delta


def test_slope_returns_positive_for_increasing_sequence():
    assert onset.slope([1.0, 2.0, 3.0, 4.0]) > 0


def test_difference_in_differences_positive_when_treatment_post_slope_increases_more():
    control = {
        "post_minus_pre_slope_beta_0_6": 0.1,
    }
    treatment = {
        "post_minus_pre_slope_beta_0_6": 0.4,
    }

    assert onset.difference_in_differences(treatment, control, beta=0.6) > 0


def test_saturation_survival_ratio_is_less_than_one_when_post_tokens_lower():
    assert onset.survival_ratio(saturation_post_tokens=320, control_post_tokens=400) < 1
