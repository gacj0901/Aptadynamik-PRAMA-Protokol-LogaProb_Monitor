# Aptadynamic Prama Protokol logaprob Monitor 

**PRAMA Protokol applied to LLM generation via logarithmic probabilities.**

Copyright © 2026 G.A.C.J. — AGPL-3.0

---

## What this is

This repository is **one application** of PRAMA Protokol, the general
aptadynamic monitoring engine. PRAMA Protokol is domain-agnostic: it
operates on any time series where two structurally independent input
channels can be defined and where tension accumulation, permissivity
decay, threshold contraction, and constitutional rotation are meaningful.

`logaprob` applies that engine to LLM generation trajectories using
token-level logarithmic probability signals as input. Other applications
of PRAMA Protokol — physiological signals, organizational dynamics,
training dynamics, production monitoring — would be separate repositories
sharing the same engine.

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

### Layer 2 — PRAMA Dynamics (engine)

The general aptadynamic engine. Integrates tension accumulation (Ξ),
permissivity decay (λ), threshold contraction (Θ), and constitutional
rotation (ι-κ-ρ) over time series. Independent of input domain.

- **Domain:** long time series (50+ steps) with structurally independent
  input channels.
- **Validation status:** verified offline (4/4 synthetic scenarios).
- **Applied to single-response logprob trajectories:** not validated —
  16 windows are insufficient for accumulator dynamics.

### Layer 3 — Verification

Tests whether the PRAMA engine adds discriminative power over Layer 1
geometry for a given domain. For single-response logprob analysis, the
answer is currently no. For multi-turn conversation, production
monitoring, and training dynamics, the question remains open.

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
for single-response analysis in this domain.** Range of xi/window across
families is 0.012 against a base of 0.130 — signal-to-noise insufficient.
This is not an engine failure: PRAMA Protokol works correctly when given
structurally distinct inputs and long enough trajectories (verified 4/4
offline). It is a domain-mismatch finding for this particular application:
16 windows of similar geometric inputs do not amplify into distinct
trajectories.

**The honest separation is the result.** Layer 1 works for what Layer 1
does in this domain. The PRAMA engine needs longer trajectories with
genuinely independent input channels — conditions that single-response
logprob analysis does not satisfy. Both layers are kept because both are
real, and the separation guides where future work should look.

---

## Repository structure

```
src/aptadynamik/                      # Source package (engine + geometry)
benchmarks/prama_components_v0.2/     # Engine component benchmarks
docs/                                 # Method, mapping, philosophy, failure modes
examples/                             # Runnable demos
frontend/                             # Visualization for multi-turn experiments
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

Confirms the PRAMA engine discriminates structurally distinct synthetic
inputs. This verifies the engine in isolation, independent of any
specific application domain.

```bash
prama-verify
```

Writes `results/results.json`. Should report `4/4 tests passed`.

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

## PRAMA Protokol applied elsewhere

This repository applies PRAMA Protokol to logprob signals from LLM
generation. The same engine is intended to operate on other domains where
the conditions are met — long time series, two structurally independent
input channels, sustained dynamics where accumulation matters.

Candidate applications, each of which would be a separate repository:

- **Multi-turn conversation monitoring:** each turn = 1 step. 40-60 turns
  provides sufficient history. Dynamic channel = prompt complexity;
  symbolic channel = response quality metrics from Layer 1.
- **Production endpoint monitoring:** each request = 1 step over hours
  and days. Dynamic = load; symbolic = quality aggregate.
- **Training dynamics:** each epoch or batch = 1 step. Dynamic = gradient
  magnitude; symbolic = validation metrics.
- **Physiological time series, organizational dynamics, financial
  systems:** wherever the structural conditions of the engine are met.

---

## Documentation

- `docs/quickstart.md` — installation and first run
- `docs/method.md` — technical method description
- `docs/logprob_mapping.md` — how token-level signals enter the engine
- `docs/failure_modes.md` — what doesn't work and why
- `docs/philosophy.md` — aptadynamic framework

---

## What this repository is not

This repo contains the measurement tooling and the logprob application of
PRAMA Protokol. It does not redistribute:

- The full aptadynamic philosophical corpus
- The formal mathematical proofs of the viability framework
- The Rust reference implementation of the engine

Those are documented separately.

---

## License

Released under the GNU Affero General Public License v3.0 (AGPL-3.0).

Commercial licensing and research collaborations may be available
separately. Contact the author.

---

## Citation

```
@software{gacj2026logaprob,
  author = {G.A.C.J.},
  title  = {{logaprob: PRAMA Protokol applied to LLM generation
             via logarithmic probabilities}},
  year   = {2026},
  url    = {https://github.com/gacj0901/logaprob}
}
```
