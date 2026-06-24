# Empirical Result 008: Early-Warning Benchmark Protocol

## 1. Purpose

Empirical Result 008 evaluates whether PRAMA provides predictive structural warning before trajectories later fail an external automatic verifier.

PRAMA is not being evaluated as a semantic hallucination detector. PRAMA is being evaluated as a structural early-warning signal for generative trajectories.

## 2. Ground Truth

Ground truth must come from automatic verification, not from PRAMA and not from the author. The verifier may be task-specific, but it must operate outside the PRAMA scoring channel.

Semantic judge outputs must not be used as the primary ground truth unless they are formalized as an automatic verifier with documented criteria and separated from PRAMA features.

## 3. Null Hypothesis

PRAMA adds no predictive signal beyond raw entropy, logprob, perplexity, top-1 gap, and entropy-variance baselines.

## 4. Success Criterion

The benchmark succeeds only if at least one out-of-sample criterion holds:

- `AUROC(PRAMA) > AUROC(best baseline)`, or
- `median lead_tokens(PRAMA) > median lead_tokens(best baseline)` at matched FPR.

The default matched false-positive rate is `0.10`.

## 5. Admissible Null Result

A null result is admissible and must be reported. If PRAMA does not outperform the strongest baseline, the report must state that the tested PRAMA score did not provide additional predictive signal under this protocol.

## 6. Required Labels

`labels.csv` must include:

- `session_id`
- `label`
- `event_token`
- `event_turn`
- `event_type`

Optional fields:

- `prompt_id`
- `expected_answer`
- `observed_answer`
- `verifier_name`
- `backend`
- `split`

If only final outcome is known, set `event_token` to the final token and document that lead time means alarm before final answer. This is a final-outcome proxy, not a localized failure lead.

## 7. Scores

Primary PRAMA score:

- `boundary_pressure`

Secondary PRAMA scores:

- `anomaly_index`
- `xi_norm`
- `negative_instant_viability_margin`
- `phase_discontinuity_score`
- `critical_slowing_score`

Baseline scores:

- `mean_entropy`
- `rolling_entropy`
- `mean_logprob`
- `rolling_mean_logprob`
- `perplexity`
- `top1_gap`
- `entropy_variance`

## 8. Causal Evaluation Rule

Scores must be evaluated causally. No threshold, alarm, or comparison may use future tokens beyond the evaluated token/turn.

If split information is available, thresholds must be selected on calibration/train examples only and evaluated separately on the test split.

## 9. Reporting

Reports must include:

- primary PRAMA score
- baseline scores
- matched-FPR thresholds
- AUROC values
- confusion matrices
- precision and recall
- median lead tokens when available
- result status: `positive`, `negative`, or `inconclusive`

## 10. Methodological Boundary

This benchmark does not measure semantic truth directly, author judgment, model intention, or material cost. It evaluates whether structural PRAMA scores provide early warning beyond simpler logprob-derived baselines.

## 11. Current Status

ER008 remains a pending external benchmark protocol. No valid Break-The-Chain / Chain-of-Code Collapse result is claimed in this repository.

Clean code-generation benchmark runs must not be reported as Break-The-Chain evidence. A future ER008 result requires a benchmark whose ground truth matches the protocol target and whose labels are external to PRAMA.
