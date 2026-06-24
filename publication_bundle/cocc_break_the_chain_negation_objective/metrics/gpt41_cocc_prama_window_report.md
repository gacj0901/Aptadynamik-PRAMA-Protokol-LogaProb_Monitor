# CoCC PRAMA Window Analysis Report

- total raw_json analizados: `230`
- window_size: `64`
- stride: `16`

Cases with `finish_reason=length` are truncated by an external token limit and must be interpreted separately.

### finish_reason

| value | count |
|---|---:|
| length | 16 |
| stop | 214 |

### regime

| value | count |
|---|---:|
| III_STRUCTURAL_PULSATION | 230 |

### regime by finish_reason

| group | regime | count |
|---|---|---:|
| length | III_STRUCTURAL_PULSATION | 16 |
| stop | III_STRUCTURAL_PULSATION | 214 |

### regime by difficulty

| group | regime | count |
|---|---|---:|
| easy | III_STRUCTURAL_PULSATION | 78 |
| hard | III_STRUCTURAL_PULSATION | 41 |
| medium | III_STRUCTURAL_PULSATION | 111 |

### PRAMA ratios and token metrics

| metric | mean | median | p90 | p10 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| token_count | 1421.200000 | 1056.500000 | 3097.200000 | 505.800000 | 256.000000 | 4096.000000 |
| assistant_chars | 4871.839130 | 3661.500000 | 10420.000000 | 1790.600000 | 915.000000 | 17978.000000 |
| mean_entropy | 0.244159 | 0.240087 | 0.326909 | 0.157023 | 0.065749 | 0.458058 |
| entropy_variance | 0.092980 | 0.094388 | 0.109305 | 0.072295 | 0.038048 | 0.119199 |
| mean_gap | 7.958840 | 7.911258 | 10.407999 | 5.522022 | 3.555591 | 12.628072 |
| mean_top1_logprob | -0.246142 | -0.239413 | -0.153475 | -0.339755 | -0.496092 | -0.067380 |
| perplexity | 1.282655 | 1.270504 | 1.404604 | 1.165879 | 1.069702 | 1.642291 |
| threshold_crossing_ratio | 0.684660 | 0.696311 | 0.887244 | 0.476910 | 0.126482 | 0.969697 |
| persistent_crossing_ratio | 0.356113 | 0.333333 | 0.619014 | 0.132180 | 0.000000 | 0.868750 |

### Top 10 by token_count

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-2777_negation_objective_7-e6f77fe0 | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-2785_negation_objective_10-dd396371 | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-2811_negation_objective_18-a781136d | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-2919_negation_objective_68-40645a42 | 4096.000000 | length | hard | III_STRUCTURAL_PULSATION |
| cocc-2955_negation_objective_74-de1f6186 | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-3024_negation_objective_83-57a1c44c | 4096.000000 | length | hard | III_STRUCTURAL_PULSATION |
| cocc-3044_negation_objective_89-4c15476b | 4096.000000 | length | easy | III_STRUCTURAL_PULSATION |
| cocc-3080_negation_objective_94-a6bcba0d | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-3094_negation_objective_98-5cce902a | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |
| cocc-3141_negation_objective_102-5400ee8b | 4096.000000 | length | medium | III_STRUCTURAL_PULSATION |

### Top 10 by mean_entropy

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-3081_negation_objective_95-0a01ea79 | 0.458058 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3224_negation_objective_137-40f68c22 | 0.443411 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-2979_negation_objective_76-65cb9bc5 | 0.418786 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3416_negation_objective_226-723e30db | 0.409189 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3209_negation_objective_128-2064b227 | 0.403323 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-2872_negation_objective_48-fa61c8c0 | 0.394704 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3298_negation_objective_169-86fb64cc | 0.382045 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-3351_negation_objective_182-b4242f32 | 0.378523 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-2810_negation_objective_17-59e76668 | 0.372833 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3249_negation_objective_156-62e18fdd | 0.371187 | stop | medium | III_STRUCTURAL_PULSATION |

### Top 10 by persistent_crossing_ratio

| session_id | value | finish_reason | difficulty | regime |
|---|---:|---|---|---|
| cocc-3055_negation_objective_93-5f7b7c0a | 0.868750 | stop | easy | III_STRUCTURAL_PULSATION |
| cocc-3416_negation_objective_226-723e30db | 0.860465 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3298_negation_objective_169-86fb64cc | 0.801527 | stop | hard | III_STRUCTURAL_PULSATION |
| cocc-3324_negation_objective_177-e7d4bfad | 0.781609 | stop | easy | III_STRUCTURAL_PULSATION |
| cocc-3387_negation_objective_198-57454c3c | 0.769231 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3239_negation_objective_149-e2e090aa | 0.754386 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-2893_negation_objective_63-8f8a966d | 0.750000 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3394_negation_objective_214-9ed7fa5b | 0.750000 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3244_negation_objective_153-364bc5a5 | 0.739130 | stop | medium | III_STRUCTURAL_PULSATION |
| cocc-3320_negation_objective_176-5e20a449 | 0.723684 | stop | easy | III_STRUCTURAL_PULSATION |
