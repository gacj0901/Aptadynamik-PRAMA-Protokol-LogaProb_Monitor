# CoCC PRAMA Window Analysis Report

- total raw_json analizados: `232`
- window_size: `64`
- stride: `16`

Cases with `finish_reason=length` are truncated by an external token limit and must be interpreted separately.

### finish_reason

| value | count |
|---|---:|
| length | 21 |
| stop | 211 |

### regime

| value | count |
|---|---:|
| CALIBRATING | 22 |
| III_STRUCTURAL_PULSATION | 205 |
| IV_ENTROPIC_COLLAPSE | 5 |

### regime by finish_reason

| group | regime | count |
|---|---|---:|
| length | III_STRUCTURAL_PULSATION | 21 |
| stop | CALIBRATING | 22 |
| stop | III_STRUCTURAL_PULSATION | 184 |
| stop | IV_ENTROPIC_COLLAPSE | 5 |

### regime by difficulty

| group | regime | count |
|---|---|---:|
| easy | CALIBRATING | 17 |
| easy | III_STRUCTURAL_PULSATION | 59 |
| easy | IV_ENTROPIC_COLLAPSE | 2 |
| hard | CALIBRATING | 1 |
| hard | III_STRUCTURAL_PULSATION | 41 |
| medium | CALIBRATING | 4 |
| medium | III_STRUCTURAL_PULSATION | 105 |
| medium | IV_ENTROPIC_COLLAPSE | 3 |

### PRAMA ratios and token metrics

| metric | mean | median | p90 | p10 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| token_count | 1191.293103 | 725.000000 | 4026.600000 | 230.700000 | 96.000000 | 4096.000000 |
| assistant_chars | 4122.530172 | 2584.500000 | 12125.500000 | 850.700000 | 334.000000 | 16409.000000 |
| mean_entropy | 0.046344 | 0.043497 | 0.070954 | 0.026268 | 0.010588 | 0.108862 |
| entropy_variance | 0.019064 | 0.018370 | 0.028832 | 0.011539 | 0.004398 | 0.036401 |
| mean_gap | 2.968369 | 1.970100 | 5.485388 | 1.333481 | 0.618551 | 32.424147 |
| mean_top1_logprob | -0.059881 | -0.055435 | -0.031743 | -0.096222 | -0.195193 | -0.015467 |
| perplexity | 1.062158 | 1.057000 | 1.101004 | 1.032252 | 1.015587 | 1.215545 |
| threshold_crossing_ratio | 0.715972 | 0.769231 | 0.936371 | 0.442119 | 0.000000 | 0.996047 |
| persistent_crossing_ratio | 0.451947 | 0.472136 | 0.799116 | 0.000000 | 0.000000 | 1.000000 |

### Top 10 by token_count

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-2755_negation_objective_4-bac6b66e | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2779_negation_objective_8-056865ea | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2791_negation_objective_12-74bb749b | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-2810_negation_objective_17-16b4302c | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2825_negation_objective_24-bac752f6 | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-2837_negation_objective_31-27c5606e | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2849_negation_objective_37-63b8def2 | 4096.000000 | length | hard | III_STRUCTURAL_PULSATION |
| cocc-2850_negation_objective_38-b5115d5d | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2872_negation_objective_48-b4f7f07f | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2881_negation_objective_54-9b5cb922 | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |

### Top 10 by mean_entropy

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-2979_negation_objective_76-d11fd879 | 0.108862 | stop | medium | CALIBRATING |
| cocc-3228_negation_objective_141-5abf3db8 | 0.104411 | stop | medium | CALIBRATING |
| cocc-3244_negation_objective_153-3b637d0a | 0.097587 | stop | medium | IV_ENTROPIC_COLLAPSE |
| cocc-3080_negation_objective_94-deefc800 | 0.092692 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3331_negation_objective_179-4f419b79 | 0.092383 | stop | easy | III_STRUCTURAL_PULSATION |
| cocc-3081_negation_objective_95-7357a187 | 0.091801 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3394_negation_objective_214-df3c31ba | 0.088623 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3414_negation_objective_227-8bd94c37 | 0.087998 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-3299_negation_objective_170-7e213b7c | 0.084194 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3403_negation_objective_231-32117d90 | 0.081147 | stop | medium | III_STRUCTURAL_PULSATION |

### Top 10 by persistent_crossing_ratio

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-3209_negation_objective_128-47408aa2 | 1.000000 | stop | medium | IV_ENTROPIC_COLLAPSE |
| cocc-3242_negation_objective_151-35b93ce0 | 1.000000 | stop | easy | IV_ENTROPIC_COLLAPSE |
| cocc-3244_negation_objective_153-3b637d0a | 1.000000 | stop | medium | IV_ENTROPIC_COLLAPSE |
| cocc-3363_negation_objective_191-31b6d43f | 1.000000 | stop | medium | IV_ENTROPIC_COLLAPSE |
| cocc-2952_negation_objective_71-623bf045 | 0.987013 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-3080_negation_objective_94-deefc800 | 0.966667 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3298_negation_objective_169-dd0e2bef | 0.950000 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-2876_negation_objective_50-0c2523ee | 0.920000 | stop | easy | III_STRUCTURAL_PULSATION |
| cocc-2754_negation_objective_3-57204e26 | 0.897959 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3328_negation_objective_190-cb7ea1ed | 0.894737 | stop | medium | III_STRUCTURAL_PULSATION |
