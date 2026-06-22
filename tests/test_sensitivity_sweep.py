"""Acceptance test for the CoreConfig sensitivity sweep (Task 1).

This file is the fixed specification. Do NOT modify it. Implement the sweep so
that every test here passes.
"""
import math

from aptadynamik.prama_core import CoreConfig


def test_parameter_inventory_partitions_coreconfig():
    from aptadynamik.observer.sensitivity import enumerate_parameters
    inv = enumerate_parameters()
    free, derived, structural = set(inv.free), set(inv.derived), set(inv.structural)

    # Derived quantities are recomputed in __post_init__; never swept directly.
    assert {"kappa_upper", "rev_max"} <= derived
    # Integer / step params are structural and held fixed.
    assert {"dt", "stagnation_window", "pathology_window"} <= structural
    # Representative continuous knobs are free.
    assert {"alpha_phi", "beta", "eta"} <= free
    # Partition is disjoint and the free set is substantive.
    assert free.isdisjoint(derived)
    assert free.isdisjoint(structural)
    assert derived.isdisjoint(structural)
    assert len(free) >= 15


def test_separation_functional_is_finite_and_positive_at_defaults():
    from aptadynamik.observer.sensitivity import separation
    s = separation(CoreConfig())
    assert math.isfinite(s)
    # Default dynamics must separate the viable scenario from the collapsing one.
    assert s > 0.0


def test_sweep_returns_finite_index_per_free_parameter():
    from aptadynamik.observer.sensitivity import enumerate_parameters, run_sweep
    inv = enumerate_parameters()
    result = run_sweep(n_trajectories=8, seed=0)
    # Exactly one sensitivity index per free parameter, all finite.
    assert set(result.indices.keys()) == set(inv.free)
    assert all(math.isfinite(v) for v in result.indices.values())
    # Sign-stability of the separation is reported as a fraction.
    assert 0.0 <= result.sign_stability <= 1.0


def test_sweep_does_not_mutate_coreconfig_defaults():
    from aptadynamik.observer.sensitivity import run_sweep
    before = CoreConfig()
    snapshot = (before.beta, before.eta, before.alpha_phi, before.kappa_base)
    run_sweep(n_trajectories=8, seed=0)
    after = CoreConfig()
    assert (after.beta, after.eta, after.alpha_phi, after.kappa_base) == snapshot
