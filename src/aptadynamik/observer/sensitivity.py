"""Global sensitivity sweep over ``CoreConfig`` (Task 1).

Partitions ``CoreConfig`` fields into FREE (continuous knobs we sweep),
DERIVED (recomputed in ``__post_init__``, never perturbed directly), and
STRUCTURAL (integer/window fields, ``dt``, and the nested ``regime_geometry``
block, held fixed). Screens the FREE parameters with a Morris
elementary-effects analysis against ``separation()``, the clean-vs-stress
viability gap, and reports whether its sign survives the sampled
neighbourhood. See docs/sensitivity_analysis.md for the rendered report.

Modeling defaults flagged for review:
- RANGE_FRACTION = 0.50: +/-50% sweep around each default, per the task brief.
- NATURAL_FLOOR = 0.0: nonnegative rates/thresholds/weights cannot go negative.
- RATIO_FIELDS upper-clamped to 1.0: these are dimensionless ratios/blend
  coefficients, not absolute magnitudes, so +50% must not push them past 1.
- regime_geometry is out of scope for this v1 sweep (listed as structural);
  its lambda_recovery=0.10 / lambda_gain=0.03 override from the scenarios is
  preserved unperturbed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from statistics import fmean

from aptadynamik.prama_core import CoreConfig, CoreState
from aptadynamik.verification import scenarios

DERIVED_FIELDS = ("kappa_upper", "rev_max")
STRUCTURAL_FIELDS = ("dt", "stagnation_window", "pathology_window", "regime_geometry")
RATIO_FIELDS = frozenset({"accel_smooth", "pre_collapse_lambda_ratio"})

RANGE_FRACTION = 0.50
NATURAL_FLOOR = 0.0

TAIL_START = 60
TAIL_END = 80

DEFAULT_LEVELS = 4


@dataclass(frozen=True)
class ParameterInventory:
    free: list[str]
    derived: list[str]
    structural: list[str]


@dataclass(frozen=True)
class SweepResult:
    indices: dict[str, float]
    sign_stability: float
    mu: dict[str, float]
    sigma: dict[str, float]
    default_separation: float
    method: str
    bounds: dict[str, tuple[float, float]]


def enumerate_parameters() -> ParameterInventory:
    defaults = CoreConfig()
    free = [
        f.name
        for f in fields(CoreConfig)
        if f.name not in STRUCTURAL_FIELDS and isinstance(getattr(defaults, f.name), float)
    ]
    return ParameterInventory(free=free, derived=list(DERIVED_FIELDS), structural=list(STRUCTURAL_FIELDS))


def parameter_bounds() -> dict[str, tuple[float, float]]:
    defaults = CoreConfig()
    inv = enumerate_parameters()
    bounds: dict[str, tuple[float, float]] = {}
    for name in inv.free:
        base = float(getattr(defaults, name))
        lower = base * (1.0 - RANGE_FRACTION)
        upper = base * (1.0 + RANGE_FRACTION)
        if name.endswith("_min") or base >= 0.0:
            lower = max(NATURAL_FLOOR, lower)
        if name in RATIO_FIELDS:
            upper = min(1.0, upper)
        if math.isclose(lower, upper):
            upper = lower + 1.0
        bounds[name] = (lower, upper)
    return bounds


def _run_scenario(input_fn, cfg: CoreConfig) -> list[float]:
    # Mirrors verification/scenarios.py::run() exactly (same regime_geometry
    # override), as an internal stepping loop so the cfg-parameterized path
    # never touches the default-cfg scenarios module.
    cfg.regime_geometry.lambda_recovery = 0.10
    cfg.regime_geometry.lambda_gain = 0.03
    state = CoreState(cfg)
    integrity: list[float] = []
    for t in range(scenarios.STEPS):
        dynamic_input, symbolic_input = input_fn(t)
        out = state.step(dynamic_input, symbolic_input, cfg)
        integrity.append(out["dominance"].integrity)
    return integrity


def _tail_mean(values: list[float]) -> float:
    return fmean(values[TAIL_START:TAIL_END])


def separation(cfg: CoreConfig) -> float:
    clean = _run_scenario(scenarios.input_varied, cfg)
    stress = _run_scenario(scenarios.input_monotonic, cfg)
    return _tail_mean(clean) - _tail_mean(stress)


def _build_config(values: dict[str, float]) -> CoreConfig:
    # Rebuild from source fields only; __post_init__ recomputes kappa_upper /
    # rev_max. regime_geometry is untouched (out of scope for v1).
    return CoreConfig(**values)


def _salib_morris(names, bounds, n_trajectories, seed):
    try:
        from SALib.analyze import morris as morris_analyze
        from SALib.sample import morris as morris_sample
    except Exception:
        return None
    problem = {"num_vars": len(names), "names": names, "bounds": [list(bounds[name]) for name in names]}
    samples = morris_sample.sample(problem, N=max(1, n_trajectories), num_levels=DEFAULT_LEVELS, seed=seed)
    default_sign = math.copysign(1.0, separation(CoreConfig()))
    y: list[float] = []
    sign_preserved = 0
    for sample in samples:
        values = {name: float(value) for name, value in zip(names, sample)}
        value = separation(_build_config(values))
        y.append(value)
        if math.copysign(1.0, value) == default_sign:
            sign_preserved += 1
    analysis = morris_analyze.analyze(problem, samples, y, print_to_console=False)
    mu_star = {name: float(analysis["mu_star"][i]) for i, name in enumerate(names)}
    mu = {name: float(analysis["mu"][i]) for i, name in enumerate(names)}
    sigma = {name: float(analysis["sigma"][i]) for i, name in enumerate(names)}
    return mu_star, mu, sigma, sign_preserved / max(len(y), 1), "SALib Morris"


def _fallback_morris(names, bounds, n_trajectories, seed):
    rng = random.Random(seed)
    effects: dict[str, list[float]] = {name: [] for name in names}
    default_sign = math.copysign(1.0, separation(CoreConfig()))
    sign_preserved = 0
    total = 0
    for _ in range(max(1, n_trajectories)):
        base_values = {name: rng.uniform(*bounds[name]) for name in names}
        base_sep = separation(_build_config(base_values))
        if math.copysign(1.0, base_sep) == default_sign:
            sign_preserved += 1
        total += 1
        for name in names:
            lo, hi = bounds[name]
            delta = (hi - lo) / 3.0
            if delta <= 0:
                continue
            shifted = dict(base_values)
            candidate = shifted[name] + delta
            if candidate > hi:
                candidate = shifted[name] - delta
            shifted[name] = candidate
            step = shifted[name] - base_values[name]
            if step == 0:
                continue
            shifted_sep = separation(_build_config(shifted))
            effects[name].append((shifted_sep - base_sep) / step)
    mu_star: dict[str, float] = {}
    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for name, values in effects.items():
        mu_star[name] = fmean(abs(v) for v in values) if values else 0.0
        mu[name] = fmean(values) if values else 0.0
        m = mu[name]
        sigma[name] = math.sqrt(fmean((v - m) ** 2 for v in values)) if len(values) > 1 else 0.0
    return mu_star, mu, sigma, sign_preserved / max(total, 1), "fallback elementary effects"


def run_sweep(n_trajectories: int = 8, seed: int = 0) -> SweepResult:
    inv = enumerate_parameters()
    names = inv.free
    bounds = parameter_bounds()
    result = _salib_morris(names, bounds, n_trajectories, seed)
    if result is None:
        result = _fallback_morris(names, bounds, n_trajectories, seed)
    mu_star, mu, sigma, sign_stability, method = result
    return SweepResult(
        indices=mu_star,
        sign_stability=sign_stability,
        mu=mu,
        sigma=sigma,
        default_separation=separation(CoreConfig()),
        method=method,
        bounds=bounds,
    )
