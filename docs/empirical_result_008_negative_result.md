# Empirical Result 008 — External LiveCodeBench Negative Result

## Executive Summary

Empirical Result 008 evaluated whether PRAMA provides external early-warning signal beyond simple logprob-derived baselines on LiveCodeBench `code_generation_lite`.

The preliminary external result is negative under the current evaluation form. PRAMA produced measurable structural signal, especially through `xi_norm`, but did not outperform the strongest uncertainty baselines when the task was framed as post-hoc classification of final functional pass/fail.

This does not invalidate PRAMA as a dynamic framework. It rejects a narrower application: using aggregated PRAMA scores from individual code-generation responses as a detector of final functional failure when compared against entropy-derived baselines.

## Null Hypothesis

The null hypothesis for ER008 is:

PRAMA adds no predictive signal beyond raw entropy, logprob, perplexity, top-1 gap, and entropy-variance baselines.

Under the LiveCodeBench runs summarized here, the null is not rejected.

## Protocol

The benchmark used externally supplied LiveCodeBench items rather than author-designed synthetic tasks. The ground truth channel was external to PRAMA:

- benchmark: LiveCodeBench `code_generation_lite`
- verifier: `public_test_cases`
- label `0`: solution passed all public tests
- label `1`: solution failed public tests, failed compilation, timed out, raised runtime error, produced unsupported output, or could not be verified
- primary PRAMA score: `boundary_pressure`
- secondary PRAMA scores: `xi_norm`, `negative_instant_viability_margin`, phase diagnostics where available
- baselines: entropy/logprob/perplexity/top1-gap/entropy-variance controls

The evaluation still used final-response outcome tokens. Therefore, it should be read as final-outcome proxy aggregation, not localized early-warning anticipation.

## DeepSeek Results

The DeepSeek / LiveCodeBench run produced a negative result against baselines.

PRAMA showed signal, especially in `xi_norm`, but simple entropy-derived baselines dominated. The run also had a strong truncation limitation: many failures reached the generation token limit. Because truncation can itself drive uncertainty and entropy metrics, this run should be treated only as weak convergent evidence.

Interpretation:

- useful as a pipeline sanity check;
- not sufficient as a strong empirical claim;
- not suitable for isolating PRAMA-specific structural warning from length/truncation effects.

## OpenAI GPT-4.1 Results

The OpenAI GPT-4.1 / LiveCodeBench run produced a cleaner external evaluation:

- validation: `True`
- valid examples: `19`
- unsupported examples: `1`
- truncated examples: `1`
- truncated failures: `1`
- final outcome proxy count: `19`
- result status: `negative`
- best PRAMA score: `boundary_pressure`, AUROC approximately `0.637`
- best baseline: `entropy_variance`, AUROC approximately `0.810`

The strongest baseline outperformed the strongest PRAMA score. Under the ER008 success criterion, this is a null result.

## Limitations

The central limitation is that ER008 currently evaluates association with final pass/fail, not a localized early-warning event.

Specific limitations:

- `event_token` is the represented final token, so lead time is a final-outcome proxy.
- There is no localized failure token inside the generated program.
- The evaluation aggregates individual responses rather than tracking long trajectories.
- Code-generation failures may be driven by task difficulty, length, syntax, incomplete solutions, or truncation.
- Entropy and entropy variance are strong controls and may capture broad uncertainty better than current PRAMA scores in this setting.
- DeepSeek results are particularly limited by truncation.

## Interpretation

The result means:

PRAMA, as currently evaluated in ER008, does not beat simple uncertainty baselines for post-hoc classification of LiveCodeBench public-test failure.

The result does not mean:

- PRAMA has no structural signal.
- PRAMA is invalid as a dynamic framework.
- PRAMA cannot detect regime changes.
- PRAMA cannot provide early warning in longer trajectories.
- PRAMA should be compared without entropy/logprob controls.

The result clarifies the boundary of the current evidence.

## What This Result Means

This is a useful negative result. It shows that final-response aggregate PRAMA scores are not enough to claim external predictive advantage over entropy-based baselines.

It also establishes that future PRAMA claims must clear a stronger control standard:

- `entropy_variance`
- `mean_entropy`
- `perplexity`
- top-1 gap variants

Any future positive result must show added value beyond those baselines.

## What This Result Does Not Mean

This result does not address PRAMA's stronger intended use cases:

- long multi-turn trajectories;
- regime transitions;
- perturbation and recovery;
- independent structural pressure channels;
- localized event-token prediction;
- trajectory-level changes before external failure.

It also does not evaluate PRAMA as a semantic judge or factuality detector.

## Implications for PRAMA

The next evaluation should move away from final pass/fail aggregation and toward trajectory dynamics.

Priority shifts:

- use localized `event_token` when available;
- evaluate recovery and relapse rather than only final correctness;
- introduce independent structural pressure rather than deriving all signal from output geometry;
- stratify by difficulty and truncation;
- preserve entropy baselines as mandatory controls;
- report null results explicitly.

## Future Work

Future ER008-style evaluations should test whether PRAMA predicts failure before final output, not merely whether PRAMA is associated with failed final answers.

The next protocol should include:

- exact or approximate failure-token localization;
- multi-turn perturbation sequences;
- longer trajectories;
- pass/fail plus recovery labels;
- difficulty-stratified AUROC;
- truncation-controlled evaluation;
- explicit analysis of top1-gap inversion as a separate finding;
- comparison by aptadynamic regime rather than final pass/fail only.

