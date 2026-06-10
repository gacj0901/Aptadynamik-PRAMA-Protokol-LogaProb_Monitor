# AptadynamiK - PRAMA Protokol: ProbLog Monitor
// Copyright © 2026 G.A.C.J.  Released under AGPL -3.0

PRAMA Protokol monitors the structural viability of LLM generation trajectories from token-level uncertainty signals. It maps local generation signals such as logprob gaps and entropy into the PRAMA core state, then records trajectory variables including integrity, xi, lambda, regime, and anomaly index.

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

