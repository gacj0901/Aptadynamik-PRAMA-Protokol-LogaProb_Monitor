# External Benchmark Adapter for Empirical Result 008

This adapter provides an external-data route for Empirical Result 008. It is designed to evaluate PRAMA as a structural early-warning signal against externally supplied benchmark items, not as a semantic hallucination detector.

## Purpose

The adapter prevents the evaluation from depending on tasks invented by the author of the experiment. It consumes local benchmark files from Break-The-Chain, ShortcutQA, or an equivalent external source, then generates PRAMA-compatible `raw.json` files and a `labels.csv` suitable for the early-warning validator and evaluator.

The adapter does not download datasets automatically. The user must provide either `--dataset-file` or `--benchmark-dir`.

## Ground Truth

Ground truth must come from the dataset or an external verifier path:

- If the dataset contains pass/fail labels, those labels are used as external ground truth.
- If the dataset contains a textual answer key, the first implementation uses a documented normalized exact-match verifier against the model's final answer line.
- For LiveCodeBench `code_generation_lite`, the adapter executes `public_test_cases` as a minimal functional verifier.
- Other datasets with tests or code remain unsupported unless they have a documented adapter or external label.

PRAMA does not generate the labels and does not decide correctness.

## Prompt Handling

Each prompt preserves the external problem text. The adapter only adds a minimal wrapper:

```text
Answer the following benchmark item. End with one final line exactly:
FINAL: <answer>

<external benchmark prompt>
```

This wrapper exists to make verification mechanically extractable. It should not change the substantive benchmark item.

## Outputs

By default the adapter writes under `results/external_break_the_chain_prama/`:

- `dataset_schema_report.md`
- `labels.csv`
- `session_index.json`
- `session_index.md`
- `sessions/<session_id>/raw.json`
- `validation/` if validation runs
- `evaluation/` if evaluation runs

Generated results are not intended to be committed.

## Methodological Notes

This adapter avoids synthetic tasks, fabricated labels, artificial class balancing, and author-designed difficulty as ground truth. It is an evaluation route for external benchmarks.

`event_token` is set to the final response token in this first version. That makes lead time a final-outcome proxy, which is weaker than a verifier that identifies the exact failure token. Results with `n < 100` should be treated as sanity checks, not strong empirical claims.

The existing `scripts/run_early_warning_sanity_sessions.py` remains an internal wiring sanity runner. It is useful for checking that PRAMA scores, validation, and evaluation are connected, but it is not an external benchmark.

## Observed ER008 Status

The LiveCodeBench adapter is now the main external functional benchmark route for ER008.

Observed implementation status:

- OpenAI GPT-4.1 with logprobs produced a valid LiveCodeBench run.
- GPT-5.5 did not support logprobs in this route, so it is not usable for PRAMA ProbLog evaluation here.
- DeepSeek produced usable logprob payloads, but the preliminary run was strongly limited by max-token truncation.

The current negative ER008 result should therefore be read as a preliminary final-outcome proxy result, not as a localized early-warning result.

## Example Commands

Single dataset file:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name break_the_chain `
  --dataset-file C:\path\to\break_the_chain.jsonl `
  --provider deepseek `
  --model deepseek-chat `
  --n 20 `
  --output-dir results\external_break_the_chain_prama
```

Benchmark directory:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name shortcutqa `
  --benchmark-dir C:\path\to\ShortcutQA `
  --provider openai `
  --model gpt-4o-mini `
  --n 20 `
  --output-dir results\external_shortcutqa_prama
```

Dry schema inspection:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name break_the_chain `
  --dataset-file C:\path\to\data.jsonl `
  --provider deepseek `
  --dry-run `
  --output-dir results\external_schema_check
```

## LiveCodeBench code_generation_lite

When `--benchmark-name livecodebench` is used, or when the dataset schema contains `question_content`, `public_test_cases`, and `question_id`, the adapter switches to a functional programming verifier route.

The LiveCodeBench mapping is:

- `question_content` becomes the external problem text.
- `question_id` becomes `prompt_id` and `benchmark_item_id`.
- `difficulty` is used as `perturbation_type` when present.
- `platform`, `question_title`, `starter_code`, and source fields are persisted as metadata.
- `public_test_cases` are used as the v1 verifier.

The prompt wrapper is intentionally minimal:

```text
You are solving a programming benchmark item.
Write a complete Python 3 program that reads from standard input and writes to standard output.
Return only code. Do not use markdown.

Problem:
{question_content}

Starter code, if useful:
{starter_code}
```

The adapter extracts Python code from fenced or plain responses, writes it to a temporary `solution.py`, and runs each public stdin test with `subprocess.run(..., shell=False)` and a timeout. Output is compared after normalizing line endings and trailing whitespace.

`private_test_cases` are not used in v1. Items with missing or unparsable `public_test_cases`, non-`stdin` test types, or insufficient PRAMA token windows are included in `session_index` but excluded from `labels.csv` and evaluation.

This is a lightweight local execution guard, not a strong security sandbox. Run only benchmark code-generation outputs that you are comfortable executing locally.

LiveCodeBench evaluates PRAMA against external functional failure. PRAMA is still not treated as a semantic judge. In this first adapter version, `event_token` is the final output token, so lead time remains a proxy for alarm before final answer rather than exact failure-token anticipation.

Example:

```powershell
python scripts\run_external_benchmark_prama_eval.py `
  --benchmark-name livecodebench `
  --dataset-file C:\path\to\test6.jsonl `
  --provider deepseek `
  --model deepseek-chat `
  --n 20 `
  --output-dir results\external_livecodebench_prama
```
