# PRAMA Break-The-Chain / CoCC Publication Evidence Bundle

This bundle packages existing evidence for a PRAMA analysis of the Break-The-Chain code-generation perturbation setting associated with Chain-of-Code Collapse (CoCC). It is generated from already-existing artifacts summarized in `metrics/gpt41_cocc_prama_window_metrics.csv`, `metrics/gpt41_cocc_prama_window_summary.json`, and `metrics/gpt41_cocc_prama_window_report.md`; it does not contain any new model generations.

## Terminology

- Break-The-Chain is treated here as the external benchmark/protocol family.
- Chain-of-Code Collapse (CoCC) names the collapse phenomenon and associated perturbation framework.
- `negation_objective` is the single perturbation family analyzed in this bundle.
- PRAMA is the structural monitor applied to existing model outputs; it is not the benchmark and does not provide the ground truth task outcome.

## Purpose

The purpose is to organize reproducible evidence for a technical paper draft: aggregate metrics, descriptive statistics, extreme cases, representative case studies, figures, and a manifest of source files.

## PRAMA Description

PRAMA is used here as a structural viability monitor over logprob-derived generation trajectories. The analyzed layer is PRAMA ProbLog Components: token-level entropy, gap, top-1 logprob and derived window dynamics are used to describe trajectory regimes. This is not a semantic truth detector and not a measurement of material cost.

## Break-The-Chain / Chain-of-Code Collapse

The available run analyzes the `negation_objective` perturbation family from the normalized CoCC/Break-The-Chain assets only. The benchmark supplies prompts and perturbation metadata; PRAMA supplies a logprob-derived structural reading of already-generated trajectories.

## Experimental Configuration

- Dataset validated upstream: 1410 records, six perturbations, 235 cases per perturbation.
- Run analyzed here: `negation_objective`.
- Expected `negation_objective` sessions: 235.
- Successfully generated sessions available in the artifact set: 230.
- Missing sessions: 5, due to API quota exhaustion interrupting generation rather than PRAMA-based filtering.
- Generation provider/model recorded in metrics: `openai` / `gpt-4.1`.
- Resolved model: `gpt-4.1-2025-04-14`.
- Window size: `64`.
- Stride: `16`.
- Sessions analyzed in window metrics: `230`.


## Inter-Model Comparison Finding

The strongest currently available observation is comparative rather than single-model. A companion artifact set analyzes the same `negation_objective` perturbation at comparable scale for DeepSeek:

| model | analyzed_sessions | mean_entropy | entropy_ratio_vs_deepseek | III_STRUCTURAL_PULSATION | IV_ENTROPIC_COLLAPSE | length_truncations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-4.1 | 230 | 0.244159 | 5.27x | 230 | 0 | 16 |
| DeepSeek | 232 | 0.046344 | 1.00x | 205 | 5 | 21 |

This indicates two separated logprob-geometric signatures under the same perturbation family: GPT-4.1 shows approximately 5x higher mean entropy than DeepSeek while producing zero `IV_ENTROPIC_COLLAPSE` assignments in the analyzed GPT-4.1 artifact set; DeepSeek shows lower entropy and 5 `IV_ENTROPIC_COLLAPSE` assignments in its companion artifact set. This comparison is descriptive and model-specific. It does not establish causality, functional correctness, or perturbation-induced collapse by itself. Of the truncated sessions (16 in GPT-4.1, 21 in DeepSeek), none received an IV classification under the CALIBRATING / persistence guards; this rules out external truncation as the source of the five IV assignments in DeepSeek. GPT-4.1 contains sessions with sustained threshold crossing (persistent_crossing_ratio up to 0.87), none of which crossed the terminal-regime threshold under the inherited calibration; whether this reflects structural recovery in GPT-4.1, model-specific calibration mismatch, or both, is a question this corpus alone cannot decide. Companion DeepSeek aggregate artifacts are included as `metrics/deepseek_cocc_prama_window_metrics.csv`, `metrics/deepseek_cocc_prama_window_summary.json`, and `metrics/deepseek_cocc_prama_window_report.md`. Bundling parity is recommended for the inter-model claim to be independently reproducible.

Recommended Figure 1 for the paper draft: `figures/model_comparison_entropy_by_difficulty.png`. Companion trajectory panels are `figures/model_comparison_xi_by_difficulty.png` and `figures/model_comparison_viability_margin_by_difficulty.png`.

## Bundle Inventory

- `figures/model_comparison_entropy_by_difficulty.png`
- `figures/model_comparison_xi_by_difficulty.png`
- `figures/model_comparison_viability_margin_by_difficulty.png`
- `metrics/gpt41_cocc_prama_window_metrics.csv`
- `metrics/gpt41_cocc_prama_window_summary.json`
- `metrics/gpt41_cocc_prama_window_report.md`
- `metrics/deepseek_cocc_prama_window_metrics.csv`
- `metrics/deepseek_cocc_prama_window_summary.json`
- `metrics/deepseek_cocc_prama_window_report.md`
- `scripts/analyze_cocc_prama_windows.py`
- `scripts/aggregate_cocc_existing_raw.py`
- `scripts/build_cocc_prama_dataset.py`
- `scripts/inspect_cocc_dataset.py`
- `scripts/run_break_the_chain_prama_eval.py`
- `scripts/generate_cocc_publication_figures.py`

## Limitations

- Only one perturbation family is included in this analyzed run.
- No control arm (neutral perturbation or no-perturbation baseline) was generated; the observed regime distribution describes the model under this perturbation but does not establish that the perturbation caused this distribution.
- 16 sessions ended with `finish_reason=length`, which indicates external truncation.
- All analyzed sessions were observed as `III_STRUCTURAL_PULSATION` under the PRAMA window analysis; this is a structural observation derived from logprob trajectories, not a functional correctness verdict or a causal explanation.
- Because all analyzed sessions mapped to a single regime, this bundle supports regime characterization for the analyzed corpus, not regime-separation performance.
- Missing top-level artifacts are recorded in the manifest rather than reconstructed.

## Reproducibility

Use `reproducibility_manifest.json` to identify every source artifact used. Raw session files are referenced, not copied into the bundle.
- Analysis and figure-generation script commit SHA recorded for reproducibility: `00857467087ed3da308b4780f2328c84920d723c`.
