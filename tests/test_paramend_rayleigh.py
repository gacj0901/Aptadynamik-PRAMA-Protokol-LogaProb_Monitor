from statistics import median

from aptadynamik.observer.paramend_rayleigh import (
    detect_early_warning,
    paramend_original,
    paramend_rayleigh,
    restitution_passive,
    synthetic_paramend_signals,
)


def test_healthy_quiet_produces_lower_rayleigh_fatigue_than_dying_stuck():
    signals = synthetic_paramend_signals()
    healthy = paramend_rayleigh(signals["healthy_quiet"], mu_star=0.0)
    dying = paramend_rayleigh(signals["dying_stuck"], mu_star=0.0)

    assert median(healthy) < median(dying)


def test_original_operator_incorrectly_assigns_high_fatigue_to_healthy_quiet():
    healthy = synthetic_paramend_signals()["healthy_quiet"]
    original = paramend_original(healthy)

    assert median(original) > 50


def test_restitution_passive_returns_low_g_for_near_unit_root_series():
    g = restitution_passive([0.1, 0.2, 0.3, 0.4, 0.5])

    assert g < 0.05


def test_restitution_passive_returns_high_g_for_strongly_restitutive_series():
    g = restitution_passive([1.0, -0.5, 0.25, -0.125, 0.06])

    assert g > 0.4


def test_transition_real_produces_rising_rayleigh_fatigue_after_onset():
    transition = synthetic_paramend_signals()["transition_real"]
    fatigue = paramend_rayleigh(transition, mu_star=0.0)
    midpoint = len(fatigue) // 2

    assert median(fatigue[midpoint:]) > median(fatigue[:midpoint])


def test_detect_early_warning_finds_g_drop_before_displacement_peak():
    transition = synthetic_paramend_signals()["transition_real"]
    warning = detect_early_warning(transition, g_threshold=0.3)

    assert warning["detected"]
    assert warning["early_warning_turn"] < warning["half_displacement_turn"]
