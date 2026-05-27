"""
AptadynamiK — PRAMA Protokol
Faithful Python port for verification.
Each function maps 1:1 to AptadinamaiK_-_PRAMA_Protokol.rs
"""
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List

EPS = 1e-12

# ── RÉGIMEN ──

class Regime(Enum):
    Stable = auto()
    Stress = auto()
    PreCollapse = auto()
    Collapse = auto()

class QuantizedRegime(Enum):
    Stable = auto()
    Tension = auto()
    Transition = auto()
    Critical = auto()
    Collapse = auto()

    def as_core_regime(self) -> Regime:
        return {
            QuantizedRegime.Stable: Regime.Stable,
            QuantizedRegime.Tension: Regime.Stress,
            QuantizedRegime.Transition: Regime.PreCollapse,
            QuantizedRegime.Critical: Regime.PreCollapse,
            QuantizedRegime.Collapse: Regime.Collapse,
        }[self]

@dataclass
class RegimeBoundaries:
    mu_stable: float
    mu_tension: float
    mu_transition: float
    mu_critical: float

# ── UTILIDADES NUMÉRICAS ──

def sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))

def squash_positive(x: float) -> float:
    x = max(0.0, x) if math.isfinite(x) else 0.0
    return x / (1.0 + x + EPS)

def effective_theta(theta_0: float, lambda_sensitivity: float) -> float:
    return theta_0 * (1.0 - max(0.0, min(1.0, lambda_sensitivity)))

def regime_indicator(delta: float, theta_eff: float, k: float) -> float:
    x = max(-60.0, min(60.0, k * (delta - theta_eff)))
    return 1.0 / (1.0 + math.exp(-x))

def classify_quantized_regime(r_t: float, b: RegimeBoundaries) -> QuantizedRegime:
    if r_t < b.mu_stable:
        return QuantizedRegime.Stable
    elif r_t < b.mu_tension:
        return QuantizedRegime.Tension
    elif r_t < b.mu_transition:
        return QuantizedRegime.Transition
    elif r_t < b.mu_critical:
        return QuantizedRegime.Critical
    else:
        return QuantizedRegime.Collapse

# ── REGIME GEOMETRY ──

@dataclass
class RegimeGeometryConfig:
    theta_0: float = 0.5
    k_regime: float = 2.0
    lambda_gain: float = 0.08
    lambda_recovery: float = 0.03
    lambda_jump_size: float = 0.15
    rho_geometry_gain: float = 0.04
    rho_geometry_recovery: float = 0.01
    mu_stable_0: float = 0.25
    mu_tension_0: float = 0.50
    mu_transition_0: float = 0.75
    mu_critical_0: float = 0.90

    def boundaries(self, rho_geometry: float) -> RegimeBoundaries:
        scale = max(0.0, min(1.0, 1.0 - max(0.0, min(1.0, rho_geometry))))
        return RegimeBoundaries(
            mu_stable=self.mu_stable_0 * scale,
            mu_tension=self.mu_tension_0 * scale,
            mu_transition=self.mu_transition_0 * scale,
            mu_critical=self.mu_critical_0 * scale,
        )

@dataclass
class RegimeGeometryState:
    lambda_sensitivity: float = 0.0
    lambda_sensitivity_prev: float = 0.0
    rho_geometry: float = 0.0
    rho_geometry_prev: float = 0.0
    collapse_count: int = 0
    theta_eff: float = 0.5
    r_t: float = 0.0
    boundaries: RegimeBoundaries = field(default_factory=lambda: RegimeBoundaries(0.25, 0.50, 0.75, 0.90))
    quantized_regime: QuantizedRegime = QuantizedRegime.Stable
    core_regime: Regime = Regime.Stable
    time_in_regime: int = 0
    last_quantized_regime: Optional[QuantizedRegime] = None

    def step(self, delta, lambda_driver, geometry_driver, collapse_shock, cfg, dt):
        self.lambda_sensitivity_prev = self.lambda_sensitivity
        self.rho_geometry_prev = self.rho_geometry

        ld = squash_positive(lambda_driver)
        gd = squash_positive(geometry_driver)

        self.lambda_sensitivity += dt * (
            cfg.lambda_gain * ld - cfg.lambda_recovery * self.lambda_sensitivity
        )
        if collapse_shock:
            self.lambda_sensitivity += cfg.lambda_jump_size
            self.collapse_count += 1

        self.lambda_sensitivity = max(0.0, min(1.0, self.lambda_sensitivity))

        self.rho_geometry += dt * (
            cfg.rho_geometry_gain * gd - cfg.rho_geometry_recovery * self.rho_geometry
        )
        self.rho_geometry = max(0.0, min(1.0, self.rho_geometry))

        self.theta_eff = effective_theta(cfg.theta_0, self.lambda_sensitivity)
        self.r_t = regime_indicator(delta, self.theta_eff, cfg.k_regime)
        self.boundaries = cfg.boundaries(self.rho_geometry)
        self.quantized_regime = classify_quantized_regime(self.r_t, self.boundaries)
        self.core_regime = self.quantized_regime.as_core_regime()

        if self.last_quantized_regime == self.quantized_regime:
            self.time_in_regime += 1
        else:
            self.time_in_regime = 1
            self.last_quantized_regime = self.quantized_regime

# ── TRÍGONO CONSTITUTIVO ──

class ConstitutionMode(Enum):
    Instauration = auto()
    Conservation = auto()
    Resolution = auto()

class ConstitutionHealth(Enum):
    Viable = auto()
    Stagnant = auto()
    Pathological = auto()
    Annihilated = auto()

@dataclass
class ConstitutionDominance:
    mode: ConstitutionMode = ConstitutionMode.Conservation
    mag_iota: float = 1/3
    mag_kappa: float = 1/3
    mag_rho: float = 1/3
    integrity: float = 1.0
    steps_in_mode: int = 0

@dataclass
class ConstitutionStatus:
    health: ConstitutionHealth = ConstitutionHealth.Viable
    integrity: float = 1.0
    rotation_count: int = 0
    dominant_mode: ConstitutionMode = ConstitutionMode.Conservation
    stagnation_steps: int = 0

def count_rotations(history: List[ConstitutionMode]) -> int:
    if len(history) < 2:
        return 0
    return sum(1 for i in range(len(history)-1) if history[i] != history[i+1])

# ── CONFIGURACIÓN CORE ──

@dataclass
class CoreConfig:
    alpha_a: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    dt: float = 1.0
    alpha_phi: float = 0.1
    alpha_psi: float = 0.1
    b_cubic: float = 0.05
    kappa_base: float = 0.02
    kappa_min: float = 0.005
    lambda_eq: float = 1.0
    lambda_min: float = 0.1
    k_sigmoid: float = 2.0
    theta: float = 0.5
    k_rev: float = 0.01
    dynamic_weight: float = 1.0
    symbolic_weight: float = 1.0
    polarized_threshold: float = 0.35
    pre_collapse_lambda_ratio: float = 0.5
    accel_cap: float = 50.0
    accel_smooth: float = 0.7
    eta: float = 0.04
    mu_r: float = 0.06
    alpha_memory: float = 0.15
    psi_input_weight: float = 0.5
    stagnation_window: int = 10
    pathology_window: int = 25
    annihilation_threshold: float = 0.15
    regime_geometry: RegimeGeometryConfig = field(default_factory=RegimeGeometryConfig)

    def __post_init__(self):
        self._kappa_upper = self.alpha_a * self.beta * self.dt * 0.99
        self._rev_max = (self._kappa_upper - self.kappa_min) * 0.5

    @property
    def kappa_upper(self): return self._kappa_upper
    @property
    def rev_max(self): return self._rev_max

# ── ESTADO CORE ──

class CoreState:
    def __init__(self, cfg: CoreConfig):
        self.phi = 0.0
        self.psi = 0.0
        self.affectio = 0.0
        self.lambda_ = cfg.lambda_eq
        self.rho = 0.5
        self.xi = 0.0
        self.delta_prev = 0.0
        self.delta_prev2 = 0.0
        self.accel_prev = 0.0

        # Constitution tracker
        self.const_history: List[ConstitutionMode] = []
        self.const_max_history = cfg.pathology_window * 2
        self.const_steps_in_mode = 0
        self.const_current_mode = ConstitutionMode.Conservation
        self.mag_iota = 1/3
        self.mag_kappa = 1/3
        self.mag_rho = 1/3
        self.alpha_mag = 0.15

        self.regime_geometry = RegimeGeometryState()

    def step(self, dynamic_input: float, symbolic_input: float, cfg: CoreConfig) -> dict:
        dt = cfg.dt
        phi_pre = self.phi
        xi_pre = self.xi
        lambda_pre = self.lambda_
        affectio_pre = self.affectio

        # Input
        total_input = cfg.dynamic_weight * dynamic_input + cfg.symbolic_weight * symbolic_input

        # Coherence Φ
        self.phi += dt * (total_input - cfg.alpha_phi * self.phi - cfg.b_cubic * self.phi**3)

        # Pressure Ψ
        psi_drive = cfg.psi_input_weight * symbolic_input
        self.psi += dt * (psi_drive - cfg.alpha_psi * self.psi)

        # Delta
        delta = abs(self.phi - self.psi)
        velocity = (delta - self.delta_prev) / dt
        accel_raw = (delta - 2*self.delta_prev + self.delta_prev2) / (dt*dt)
        accel_filtered = cfg.accel_smooth * self.accel_prev + (1 - cfg.accel_smooth) * accel_raw
        accel = max(-cfg.accel_cap, min(cfg.accel_cap, accel_filtered))
        self.accel_prev = accel
        self.delta_prev2 = self.delta_prev
        self.delta_prev = delta

        # Regime geometry
        geometry_driver = min(1.0, max(0.0, abs(accel)/cfg.accel_cap)) + abs(self.rho - 0.5)
        collapse_shock = self.lambda_ <= cfg.lambda_min

        self.regime_geometry.step(
            delta, self.xi + delta, geometry_driver,
            collapse_shock, cfg.regime_geometry, dt
        )

        activation = self.regime_geometry.r_t
        theta_eff = self.regime_geometry.theta_eff
        force = max(0.0, delta - theta_eff)

        # Reversibility
        kinetic = velocity * velocity
        direction = 1.0 if velocity > 0 else (-1.0 if velocity < 0 else 0.0)
        reversibility = max(-cfg.rev_max, min(cfg.rev_max,
            cfg.k_rev * force * activation * kinetic * direction))
        kappa_eff = max(cfg.kappa_min, min(cfg.kappa_upper, cfg.kappa_base - reversibility))

        # Affectio
        self.affectio += dt * (force - cfg.alpha_a * self.affectio)
        self.affectio = max(0.0, self.affectio)

        # Memory Ξ
        self.xi += dt * (cfg.alpha_memory * (delta - self.xi))

        # Structural surplus
        s = self.phi - self.psi
        degradation = cfg.eta * abs(s)
        recovery = cfg.mu_r * max(0.0, s)

        # Permissivity λ
        self.lambda_ += dt * (
            -degradation + recovery - kappa_eff * self.affectio
            - cfg.beta * (self.lambda_ - cfg.lambda_eq)
        )

        # Polarization ρ
        rho_target = sigmoid(cfg.k_sigmoid * total_input)
        self.rho += dt * cfg.gamma * (rho_target - self.rho)
        self.rho = max(0.0, min(1.0, self.rho))

        regime = self.regime_geometry.core_regime

        lambda_norm = max(0.0, min(1.0,
            (self.lambda_ - cfg.lambda_min) / max(EPS, cfg.lambda_eq - cfg.lambda_min)))

        anomaly_index = max(0.0, min(1.0,
            (1.0 - lambda_norm) * activation * (1.0 + abs(self.rho - 0.5))))

        # Constitution update
        dominance = self._update_constitution(phi_pre, xi_pre, lambda_pre, affectio_pre, delta, cfg)
        status = self._evaluate_constitution_status(cfg)

        return {
            'delta': delta,
            'theta_eff': theta_eff,
            'lambda': self.lambda_,
            'lambda_sensitivity': self.regime_geometry.lambda_sensitivity,
            'affectio': self.affectio,
            'kappa_eff': kappa_eff,
            'rho': self.rho,
            'xi': self.xi,
            'regime': regime,
            'quantized_regime': self.regime_geometry.quantized_regime,
            'anomaly_index': anomaly_index,
            'dominance': dominance,
            'status': status,
            'phi': self.phi,
            'psi': self.psi,
        }

    def _update_constitution(self, phi_pre, xi_pre, lambda_pre, affectio_pre, delta, cfg):
        phi_growth = max(0.0, self.phi - phi_pre)
        delta_growth = max(0.0, delta - self.delta_prev2)
        lambda_health = max(0.0, min(1.0,
            (self.lambda_ - cfg.lambda_min) / max(EPS, cfg.lambda_eq - cfg.lambda_min)))
        iota_raw = (phi_growth + delta_growth) * lambda_health

        lambda_stability = 1.0 / (1.0 + abs(self.lambda_ - lambda_pre) * 10.0)
        delta_bounded = 1.0 / (1.0 + delta)
        affectio_managed = 1.0 / (1.0 + self.affectio)
        kappa_raw = lambda_stability * delta_bounded * affectio_managed

        xi_decrease = max(0.0, xi_pre - self.xi)
        affectio_decrease = max(0.0, affectio_pre - self.affectio)
        delta_contraction = max(0.0, self.delta_prev2 - delta)
        rho_raw = xi_decrease + affectio_decrease + delta_contraction

        total = iota_raw + kappa_raw + rho_raw + EPS
        iota_norm = iota_raw / total
        kappa_norm = kappa_raw / total
        rho_norm = rho_raw / total

        a = self.alpha_mag
        self.mag_iota += a * (iota_norm - self.mag_iota)
        self.mag_kappa += a * (kappa_norm - self.mag_kappa)
        self.mag_rho += a * (rho_norm - self.mag_rho)

        if self.mag_iota > self.mag_kappa and self.mag_iota > self.mag_rho:
            mode = ConstitutionMode.Instauration
        elif self.mag_rho > self.mag_iota and self.mag_rho > self.mag_kappa:
            mode = ConstitutionMode.Resolution
        else:
            mode = ConstitutionMode.Conservation

        if mode != self.const_current_mode:
            self.const_steps_in_mode = 0
            self.const_current_mode = mode
        else:
            self.const_steps_in_mode += 1

        self.const_history.append(mode)
        if len(self.const_history) > self.const_max_history:
            self.const_history.pop(0)

        geo_mean = (self.mag_iota * self.mag_kappa * self.mag_rho) ** (1/3)
        integrity = max(0.0, min(1.0, geo_mean / (1/3)))

        return ConstitutionDominance(mode, self.mag_iota, self.mag_kappa, self.mag_rho, integrity, self.const_steps_in_mode)

    def _evaluate_constitution_status(self, cfg):
        geo_mean = (self.mag_iota * self.mag_kappa * self.mag_rho) ** (1/3)
        integrity = max(0.0, min(1.0, geo_mean / (1/3)))
        rotations = count_rotations(self.const_history)

        if integrity < cfg.annihilation_threshold:
            health = ConstitutionHealth.Annihilated
        elif self.const_steps_in_mode >= cfg.pathology_window or integrity < cfg.annihilation_threshold * 2:
            health = ConstitutionHealth.Pathological
        elif self.const_steps_in_mode >= cfg.stagnation_window:
            health = ConstitutionHealth.Stagnant
        elif rotations >= 2 or len(self.const_history) < cfg.stagnation_window:
            health = ConstitutionHealth.Viable
        else:
            health = ConstitutionHealth.Stagnant

        return ConstitutionStatus(health, integrity, rotations, self.const_current_mode, self.const_steps_in_mode)
