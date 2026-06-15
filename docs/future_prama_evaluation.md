# Future PRAMA Evaluation

## Purpose

Empirical Result 008 showed that final-outcome proxy aggregation is not enough to demonstrate PRAMA advantage over simple logprob baselines. Future evaluations should target the dynamic claims of PRAMA more directly.

PRAMA should be evaluated as a trajectory instrument: regime formation, threshold approach, perturbation, recovery, and relapse.

## 1. Localized Event Tokens

Future datasets should identify where failure emerges, not only whether a final answer passes or fails.

Useful event markers include:

- first syntactic impossibility;
- first failed invariant;
- first contradiction with a required constraint;
- first incorrect branch in a derivation;
- first unit-test-relevant code defect;
- first recovery after an incorrect local step.

When exact localization is impossible, the report must distinguish:

- exact event token;
- approximate event span;
- final-outcome proxy.

Only exact or approximate event localization can support strong early-warning claims.

## 2. Multi-Turn Series

PRAMA should be tested on multi-turn trajectories rather than isolated final responses.

Recommended designs:

- neutral prefix followed by perturbation;
- progressive constraint tightening;
- recovery prompts after a failure;
- alternating low/high pressure conditions;
- repeated attempts on the same task with controlled reformulations.

The unit of analysis should be the trajectory, not only the final answer.

## 3. Independent Structural Pressure

Future protocols should include an input-side pressure channel independent of output logprobs.

Examples:

- externally defined constraint load;
- contradiction load;
- prompt pressure Ψ;
- perturbation type;
- closure strictness;
- number of simultaneous restrictions.

This helps distinguish pressure-driven regime change from uncertainty already visible in output entropy.

## 4. Recovery After Perturbation

PRAMA should be evaluated on recovery dynamics:

- Does the trajectory cross a threshold?
- Does it recover?
- How many turns or tokens does recovery require?
- Does recovery occur before external failure?
- Does relapse occur after temporary recovery?

This better matches aptadynamic regime classification than binary pass/fail.

## 5. Compare by Regime, Not Only Pass/Fail

Future reports should stratify examples by regime:

- `CALIBRATING`
- `II_ORGANIZED_STABILITY`
- `III_STRUCTURAL_PULSATION`
- `IV_ENTROPIC_COLLAPSE`

The relevant question is not only whether a final output failed, but whether certain regimes predict later failure, recovery, or instability better than entropy baselines.

## 6. Top1-Gap Inversion

The observed behavior of top-1 gap should be treated as a separate empirical finding.

Possible interpretations:

- confident wrong paths;
- premature narrowing;
- reduced local uncertainty before failure;
- code-generation determinacy without functional correctness.

This should not be folded into PRAMA without analysis. It should be reported as a baseline phenomenon and compared independently.

## 7. Stratification

Future evaluations should report metrics stratified by:

- difficulty;
- platform;
- truncation status;
- token length;
- pass/fail class;
- unsupported verifier count;
- backend/model.

Truncation must be tracked separately because it can dominate both entropy baselines and failure labels.

## 8. Required Controls

Any future positive PRAMA claim must compare against:

- `mean_entropy`
- `entropy_variance`
- `perplexity`
- `mean_logprob`
- `rolling_mean_logprob`
- `top1_gap`

The strongest baseline should be treated as the comparison target.

## 9. Publication Standard

A stronger PRAMA result should include:

- at least one external benchmark;
- no fabricated labels;
- train/test or calibration/test split when thresholds are tuned;
- localized or span-level event tokens;
- matched-FPR comparison;
- lead-token or lead-turn analysis;
- null-result reporting.

If PRAMA does not beat baselines, the result should be reported as negative, not reframed.

