# Early-Warning Evaluation

## Internal Regime Validation vs External Predictive Validation

Internal regime validation checks whether PRAMA's own structural measurements behave consistently under controlled or deterministic trajectories. External predictive validation asks a harder question: whether PRAMA provides warning before a trajectory later fails an external automatic verifier.

Empirical Result 008 belongs to the second category. It does not treat PRAMA as a semantic hallucination detector. It treats PRAMA as a structural early-warning signal.

## Why Baselines Are Mandatory

Logprob-derived baselines such as entropy, mean logprob, perplexity, top-1 gap, and entropy variance can themselves predict later failure. A PRAMA score is only informative as an early-warning layer if it adds signal beyond those simpler baselines.

## Why Matched FPR Is Required

A warning method can look better simply by alarming more often. Matched false-positive rate compares methods at the same false-positive budget, making lead time and recall more meaningful.

## Why Lead Time in Tokens Is the Headline Metric

AUROC measures ranking quality, but early warning also requires time advantage. `lead_tokens` measures how many output tokens occur between the first alarm and the externally verified failure event. A larger median lead at matched FPR indicates more practical warning value.

## Final-Token Labels Are Only a Proxy

If the only available ground truth is final success or failure, `event_token` may be set to the final token. In that case, lead time means alarm before the final answer, not before a localized verified failure point. Reports must mark this as a final-outcome proxy.

## Null Results Are Admissible

The null hypothesis is that PRAMA adds no predictive signal beyond raw entropy/logprob/perplexity baselines. If the PRAMA score does not outperform the strongest baseline out of sample, the result is negative or inconclusive and must be reported as such.

## Limitations

- The benchmark depends on the quality of external automatic verification.
- Thresholds must be selected on calibration/train examples when split information exists.
- Semantic judge output is not primary ground truth unless formalized as an external automatic verifier.
- This evaluation does not measure truth directly, intention, material cost, GPU state, energy use, temperature, memory pressure, or latency.
