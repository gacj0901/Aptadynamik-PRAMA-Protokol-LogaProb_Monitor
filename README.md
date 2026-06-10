# AptadynamiK - PRAMA Protokol: ProbLog Monitor
// Copyright © 2026 G.A.C.J.  Released under AGPL -3.0

PRAMA Protokol monitors the structural viability of LLM generation trajectories from token-level uncertainty signals. It maps local generation signals such as logprob gaps and entropy into the PRAMA core state, then records trajectory variables including integrity, xi, lambda, regime, and anomaly index.

<!-- ============================================================
  DRAFT: "Findings so far" section for README.md
  Suggested placement: immediately after the opening paragraph,
  before "Install".
  ============================================================ -->

## Findings so far

PRAMA Monitor is developed against explicit hypotheses, and negative results are published alongside positive ones. The numbered reports in `docs/empirical_result_001.md` … `docs/empirical_result_007.md` document the full arc. Summary:

**1. The instrument does not detect hallucination — it detects structural regime.**
First contact with live logprobs inverted the initial prediction: semantically "clean" prompts accumulated *more* tension Ξ than semantically stressful ones, because high-confidence generation is structurally rigid (low entropy, large top-1 gaps, fewer alternatives). PRAMA measures the probabilistic geometry of a generation trajectory — concentration, rigidity, loss of alternatives, accumulated tension, loss of permissivity — not semantic truth. Structural (aptadynamic) stress and semantic stress are different variables. ([Result 001](docs/empirical_result_001.md))

**2. The naive operational hypothesis failed at scale — and the failure was diagnostic.**
With a larger prompt-family sample, the hypothesis "contradiction/saturation ⇒ higher Ξ" scored 0/4 ([Result 003](docs/empirical_result_003.md)). Adding intra-window entropy variance recovered 3/4 ([Result 004](docs/empirical_result_004.md)). The deeper diagnosis: the discriminating signals were being computed *before* the PRAMA dynamics — they were properties of raw logprob geometry, not of the dynamic core — and both the coherence channel Φ and the pressure channel Ψ were being derived from the same output stream, making the coupling circular ([Result 005](docs/empirical_result_005.md)).

**3. An independent prompt-pressure channel Ψ restores genuine dynamics.**
Extracting Ψ directly from the input prompt *before* generation — independent of the output-derived Φ — makes Δ = |Φ − Ψ| a real coupling between environmental demand and generative response, and restores dynamic discrimination in the v2 pipeline ([Result 006](docs/empirical_result_006.md)).

**4. Preregistered onset experiment: the form of pressure determines the dynamic regime.**
Across 50 neutral topics and 150 long trajectories, with three memory kernels (β = 0.3 / 0.6 / 0.9):

- **Contradictory pressure** produces sustained post-onset growth of long-memory tension Ξ — positive difference-in-differences in 50/50 topics at every β (mean DiD: +0.333 / +0.198 / +0.119).
- **Saturation pressure** does not accumulate tension; it consumes margin and truncates the trajectory — survival below control in 50/50 topics, with 0/50 completing the post-onset stage.

Ψ does not induce a homogeneous Ξ response: the *type* of pressure determines the regime. This dissociation is invisible to event counting and is the core claim of trajectory-level structural monitoring. ([Result 007](docs/empirical_result_007.md))

**Status.** These results come from gpt-4o-mini, DeepSeek and Gemini pipelines at modest scale. They establish internal consistency and a reproducible dissociation between pressure types; they do **not** constitute calibrated risk scores. Known operational limits are listed in [`docs/failure_modes.md`](docs/failure_modes.md). Short trajectories are protected from terminal-regime false positives by the `CALIBRATING / INSUFFICIENT_HISTORY` guard, validated in [`docs/regime_benchmark.md`](docs/regime_benchmark.md).

<!-- ============================================================
  OPTIONAL: suggested replacement for the README opening paragraph
  ============================================================ -->

> PRAMA Protokol monitors the **structural viability of LLM generation trajectories** from token-level uncertainty signals (logprob gaps, entropy, intra-window variance). It maps local generation geometry into a dynamic core with memory — tension Ξ, permissivity λ, a contracting viability threshold Θ(λ), and regime classification — so that risk is assessed at the level of the **trajectory**, not of isolated events. Developed as the measurement layer of the Aptadynamics framework ([formal corpus](https://doi.org/10.5281/zenodo.20369325)); see also [ORDSPOC](https://github.com/gacj0901/prama-protokol-ordspoc), a companion proof-of-concept applying the same core to autonomous-orchestration risk.

<!-- ============================================================
  Suggested GitHub repo topics (Settings → Topics):
  llm-monitoring · interpretability · logprobs · ai-safety ·
  uncertainty-quantification · dynamical-systems · viability-theory ·
  llm-evaluation · trajectory-analysis
  ============================================================ -->

This repository is organized as a Python package under `src/aptadynamik`.

## Install

```bash
python -m pip install -e .
```

For the Gemini demo:

```bash
python -m pip install -r requirements.txt
```

## Offline Verification

```bash
prama-verify
```

or:

```bash
python examples/run_offline_verify.py
```

The offline run writes `results/results.json` and should report `4/4 tests passed`.

## Gemini Pipeline

Set an API key and run:

```bash
export GEMINI_API_KEY="your-key"
prama-gemini
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="your-key"
prama-gemini
```

The Gemini pipeline writes timestamped JSON and CSV outputs to `results/`.

## Documentation

- `docs/quickstart.md`
- `docs/method.md`
- `docs/logprob_mapping.md`
- `docs/failure_modes.md`
- `docs/philosophy.md`

- License

This project is released under the GNU Affero General Public License v3.0 (AGPL-3.0).

Commercial licensing and research collaborations may be available separately.

