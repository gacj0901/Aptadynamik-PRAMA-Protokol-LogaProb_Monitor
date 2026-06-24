# AptadynamiK / PRAMA Protokol

**Structural analysis of LLM generation dynamics via logprob geometry and aptadynamic monitoring.**

Copyright © 2026 G.A.C.J. — AGPL-3.0

---

## Architecture

The repository is organized in three independent layers. Each has its own
validation status and own domain of application. They are not collapsible
into one another — that separation is itself an experimental finding.

### Layer 1 — Logprob Geometry

Extracts rigidity, uncertainty, margin, and entropy-volatility metrics from
token-level logprob distributions. Discriminates prompt families by
generation structure, not by content truth.

- **Domain:** single-response analysis.
- **Validation status:** stable across replications.
- **Discriminative power:** confirmed.

### Layer 2 — PRAMA Dynamics

Aptadynamic engine that integrates tension accumulation (Ξ), permissivity
decay (λ), threshold contraction (Θ), and constitutional rotation (ι-κ-ρ)
over time series.

- **Domain:** long time series (50+ steps): multi-turn conversation,
  production monitoring, training dynamics.
- **Validation status:** verified offline (4/4 synthetic scenarios). Not
  validated on single-response logprob trajectories — 16 windows are
  insufficient for accumulator dynamics.
- **Discriminative power on single responses:** not confirmed.

### Layer 3 — Verification

Tests whether Layer 2 adds discriminative power over Layer 1 for a given
domain. For single-response logprob analysis, the answer is currently no.
For longer time series, the question remains open.

---

## Empirical findings

### Replication summary (8 runs, GPT-4o-mini, N=3-5 per family per run)

| Metric (aggregate) | Canonical | Fictional | Contradictory | Saturation |
|--------------------|-----------|-----------|---------------|------------|
| Rigidity | 0.875 | 0.643 | 0.727 | 0.316 |
| Uncertainty | 0.005 | 0.058 | 0.039 | 0.241 |
| Entropy std | 0.232 | 0.414 | 0.374 | 0.484 |
| Entropy range | 0.645 | 1.174 | 1.026 | 1.345 |
| Margin | 0.610 | 0.753 | 0.715 | 0.877 |
| Xi / window | 0.133 | 0.121 | 0.136 | 0.124 |

### Test stability across 8 replications

| Test | Stability | Status |
|------|-----------|--------|
| Rigidity gradient: canonical > contradictory > fictional > saturation | 8/8 | strong |
| Margin: contradictory < fictional | 7/8 | strong |
| Entropy std: saturation > canonical | 6/8 | moderate |
| Entropy range: saturation > canonical | 6/8 | moderate |
| Xi/window: structural > semantic | 5/8 | weak |
| Xi/window: saturation > canonical | 1/8 | not stable |

### What this means

**Logprob geometry metrics discriminate stress families.** The rigidity
gradient is perfectly monotonic across all 8 replications. The distinction
between semantic stress (fiction) and structural stress (contradiction,
saturation) is measurable in the volatility of the logprob distribution
itself — not in any derived dynamical quantity.

**PRAMA xi/window does not add reliable discrimination over raw geometry
for single-response analysis.** Range of xi/window across families is
0.012 against a base of 0.130 — signal-to-noise insufficient. This is not
an engine failure: the engine works correctly when given structurally
distinct inputs and long enough trajectories (verified 4/4 offline). It is
a domain-mismatch finding: 16 windows of similar geometric inputs do not
amplify into distinct trajectories.

**The honest separation is the result.** Layer 1 works for what Layer 1
does. Layer 2 needs its own domain. Both are kept because both are real.

---

## Repository structure

```
src/aptadynamik/                      # Source package
benchmarks/prama_components_v0.2/     # Component benchmarks
docs/                                 # Method, mapping, philosophy, failure modes
examples/                             # Runnable demos
frontend/                             # Visualization layer for multi-turn experiments
protocols/                            # Experimental protocols (prompt families, configs)
results/                              # Outputs from replication runs
scripts/                              # Trajectory analysis, replication tooling
tests/                                # Unit tests + offline verification (4/4)
.github/workflows/                    # CI
```

---

## Install

```bash
python -m pip install -e .
```

For pipelines that hit external APIs:

```bash
python -m pip install -r requirements.txt
```

---

## Offline verification

Confirms the engine discriminates structurally distinct synthetic inputs.

```bash
prama-verify
```

Writes `results/results.json`. Should report `4/4 tests passed`.

This verifies Layer 2 in isolation — the engine itself, not its mapping
from logprobs.

---

## Pipelines

### Gemini (free tier, requires API key)

```bash
export GEMINI_API_KEY="your-key"
prama-gemini
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
python -m aptadynamik.pipelines.v2
```

Both write timestamped JSON and CSV outputs to `results/`.

### Replication analysis

After running the v2 pipeline multiple times:

```bash
python scripts/analyze_v2_replications.py
```

Produces aggregate means, per-run test results, and stability scores
across runs.

---

## Open questions

Layer 2 is not invalidated. It is unvalidated for single-response analysis
and untested in its natural domain. The three candidate domains:

- **Multi-turn conversation monitoring:** each turn = 1 step. 40-60 turns
  provides sufficient history. Dynamic channel = prompt complexity;
  symbolic channel = response quality metrics from Layer 1.
- **Production endpoint monitoring:** each request = 1 step over hours
  and days. Dynamic = load; symbolic = quality aggregate.
- **Training dynamics:** each epoch or batch = 1 step. Dynamic = gradient
  magnitude; symbolic = validation metrics.

Each of these has structurally independent input channels and time series
long enough for accumulator dynamics, threshold contraction, and
constitutional rotation to operate as designed.

---

## Documentation

- `docs/quickstart.md` — installation and first run
- `docs/method.md` — technical method description
- `docs/logprob_mapping.md` — how token-level signals enter the engine
- `docs/failure_modes.md` — what doesn't work and why
- `docs/philosophy.md` — aptadynamic framework

---

## What this repository is not

This repo contains the measurement and monitoring tooling. The full
aptadynamic philosophical corpus, the formal mathematical proofs of the
viability framework, and the Rust reference implementation are documented
separately and are not redistributed here.

---

## License

Released under the GNU Affero General Public License v3.0 (AGPL-3.0).

Commercial licensing and research collaborations may be available
separately. Contact the author.

---

## Citation

```
@software{gacj2026aptadynamik,
  author = {G.A.C.J.},
  title  = {{AptadynamiK / PRAMA Protokol: Structural viability monitoring
             for LLM generation trajectories via logprob-derived dynamics}},
  year   = {2026},
  url    = {https://github.com/gacj0901/Aptadynamik-PRAMA-Protokol-ProbLog_Monitor}
}
```
