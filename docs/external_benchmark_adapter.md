# External Benchmark Adapter for Empirical Result 008

This adapter provides an external-data route for Empirical Result 008. It is an execution utility, not an empirical result document.

## Purpose

The adapter is intended to evaluate PRAMA against externally supplied benchmark items and externally defined labels or verifiers. It should not be used to fabricate tasks, labels, class balance, or ground truth.

ER008 remains pending until a benchmark matching the target protocol is run and validated. Clean code-generation runs are not Break-The-Chain / Chain-of-Code Collapse evidence.

## Ground Truth Requirements

Ground truth must come from an external benchmark or verifier. PRAMA must not create the label it is evaluated against.

Acceptable sources include:

- externally supplied pass/fail labels;
- externally supplied answer keys with a documented automatic verifier;
- externally supplied public tests when the benchmark is explicitly a programming benchmark;
- future task-specific verifiers that are documented and separated from PRAMA features.

Unsupported or unverifiable examples must be excluded from `labels.csv` and reported in `session_index`.

## Prompt Handling

For textual answer-key benchmarks, the adapter preserves the external problem text and adds only a minimal answer-extraction wrapper:

```text
Answer the following benchmark item. End with one final line exactly:
FINAL: <answer>

<external benchmark prompt>
```

For programming benchmarks, any wrapper must preserve the external problem and avoid adding substantive hints or transformations.

## Outputs

A run writes under the selected `--output-dir`:

- `dataset_schema_report.md`
- `labels.csv`
- `session_index.json`
- `session_index.md`
- `sessions/<session_id>/raw.json`
- `validation/` if validation runs
- `evaluation/` if evaluation runs

Generated outputs are not intended to be committed.

## Methodological Boundary

This adapter is a plumbing layer. It does not establish evidence by itself.

A result is only interpretable after:

- input validation passes;
- labels are externally justified;
- unsupported rows are excluded rather than coerced;
- truncation and final-outcome proxy limitations are reported;
- PRAMA scores are compared against entropy/logprob/perplexity baselines.

## Current ER008 Status

No valid Break-The-Chain / Chain-of-Code Collapse result is claimed here.

The adapter may be used for future external benchmark runs, but clean code-generation benchmark output must not be described as Break-The-Chain evidence.

## Example Commands

Single dataset file:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name external_benchmark `
  --dataset-file C:\path\to\dataset.jsonl `
  --provider deepseek `
  --model deepseek-chat `
  --n 20 `
  --output-dir results\external_benchmark_prama
```

Benchmark directory:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name external_benchmark `
  --benchmark-dir C:\path\to\benchmark_dir `
  --provider openai `
  --model gpt-4.1 `
  --n 20 `
  --output-dir results\external_benchmark_prama
```

Dry schema inspection:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name external_benchmark `
  --dataset-file C:\path\to\data.jsonl `
  --provider deepseek `
  --dry-run `
  --output-dir results\external_schema_check
```
