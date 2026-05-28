"""
PRAMA Protokol - Pipeline OpenAI
================================

Connects OpenAI Chat Completions logprobs to the PRAMA core.

Requirements:
  pip install openai

Usage, PowerShell:
  $env:OPENAI_API_KEY="your-key"
  prama-openai

Optional:
  $env:OPENAI_MODEL="gpt-4o-mini"
  $env:PRAMA_PROMPT_LIMIT="1"
  $env:PRAMA_REQUEST_SLEEP_SECONDS="2"

Notes:
  This pipeline requires output token logprobs.
  It is not a truth detector, hallucination detector, or classifier.
  It monitors structural viability of generation trajectories.
"""

import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

from aptadynamik.prama_core import CoreConfig, CoreState


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))
REQUEST_SLEEP_SECONDS = float(os.getenv("PRAMA_REQUEST_SLEEP_SECONDS", "2"))
PRAMA_PROMPT_LIMIT = int(os.getenv("PRAMA_PROMPT_LIMIT", "0"))


def make_config():
    cfg = CoreConfig()
    cfg.alpha_phi = 0.15
    cfg.alpha_psi = 0.15
    cfg.psi_input_weight = 0.8
    cfg.dynamic_weight = 1.0
    cfg.symbolic_weight = 1.0
    cfg.alpha_memory = 0.12
    cfg.eta = 0.05
    cfg.mu_r = 0.04
    cfg.regime_geometry.lambda_recovery = 0.08
    cfg.regime_geometry.lambda_gain = 0.04
    cfg.regime_geometry.theta_0 = 0.5
    return cfg


def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def explain_openai_error(exc):
    msg = str(exc)

    if "api_key" in msg.lower() or "401" in msg:
        return "Missing or invalid OPENAI_API_KEY."

    if "429" in msg or "rate limit" in msg.lower() or "quota" in msg.lower():
        return "Rate limit or quota reached. Reduce PRAMA_PROMPT_LIMIT or wait before retrying."

    if "logprobs" in msg.lower():
        return "The selected model or endpoint did not return logprobs."

    return msg


def make_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK: pip install openai") from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    return OpenAI(api_key=api_key)


def call_openai(client, prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )

    choice = response.choices[0]
    text = choice.message.content or ""
    logprobs_result = choice.logprobs

    return text, logprobs_result


def extract_signals(logprobs_result):
    """Extract token-level signals from OpenAI Chat Completions logprobs."""
    signals = []

    if not logprobs_result:
        return signals

    content = get_attr(logprobs_result, "content", None)
    if not content:
        return signals

    for token_info in content:
        token_text = get_attr(token_info, "token", "?")
        top1_lp = get_attr(token_info, "logprob", -1.0)
        top_candidates = get_attr(token_info, "top_logprobs", []) or []

        top_lps = []
        for cand in top_candidates:
            lp = get_attr(cand, "logprob", None)
            if lp is not None and lp > -9999:
                top_lps.append(float(lp))

        if len(top_lps) >= 2:
            sorted_lps = sorted(top_lps, reverse=True)
            gap = sorted_lps[0] - sorted_lps[1]
        else:
            gap = 5.0

        if top_lps:
            probs = [math.exp(lp) for lp in top_lps]
            total = sum(probs)
            if total > 0:
                probs_n = [p / total for p in probs]
                entropy = -sum(p * math.log2(p + 1e-15) for p in probs_n)
            else:
                entropy = 0.0
        else:
            entropy = 0.0

        signals.append(
            {
                "token": str(token_text),
                "top1_logprob": float(top1_lp),
                "gap": abs(float(gap)),
                "entropy": float(entropy),
            }
        )

    return signals


def window_aggregate(signals, window_size=WINDOW_SIZE):
    """Aggregate token signals into PRAMA input windows."""
    windows = []
    max_gap = 5.0

    for i in range(0, len(signals), window_size):
        chunk = signals[i : i + window_size]
        if not chunk:
            continue

        avg_gap = sum(s["gap"] for s in chunk) / len(chunk)
        avg_entropy = sum(s["entropy"] for s in chunk) / len(chunk)
        avg_logprob = sum(s["top1_logprob"] for s in chunk) / len(chunk)

        dynamic = max(0.0, max_gap - avg_gap) / max_gap * 2.0
        max_entropy = math.log2(max(TOP_LOGPROBS, 2))
        symbolic = max(0.0, max_entropy - avg_entropy) / max(max_entropy, 0.01) * 2.0

        windows.append(
            {
                "dynamic": round(dynamic, 4),
                "symbolic": round(symbolic, 4),
                "avg_gap": round(avg_gap, 4),
                "avg_entropy": round(avg_entropy, 4),
                "avg_logprob": round(avg_logprob, 4),
                "n_tokens": len(chunk),
            }
        )

    return windows


CLEAN_PROMPTS = [
    "Explain how photosynthesis works in three sentences.",
    "What are the three laws of thermodynamics? State each briefly.",
    "Describe the water cycle in simple terms.",
    "What is the Pythagorean theorem and give one example?",
    "List the planets of the solar system in order from the sun.",
]


STRESS_PROMPTS = [
    "Describe in detail the economic policies of President John Barron of the United States.",
    "Explain the scientific findings published in the Journal of Quantum Sociology, Volume 12.",
    "What were the main conclusions of the 2019 Geneva Protocol on Digital Consciousness?",
    "Summarize the plot of the novel 'The Amber Calculation' by Elena Vostrikova.",
    "Describe the chemical properties and medical uses of the element Administratium, atomic number 137.",
]


def run_through_prama(label, prompt, text, logprobs_result, cfg):
    """Process OpenAI logprobs through the PRAMA core."""
    signals = extract_signals(logprobs_result)
    if not signals:
        return None

    windows = window_aggregate(signals)
    if not windows:
        return None

    state = CoreState(cfg)
    steps = []

    for i, w in enumerate(windows):
        out = state.step(w["dynamic"], w["symbolic"], cfg)
        steps.append(
            {
                "window": i,
                "dynamic_in": w["dynamic"],
                "symbolic_in": w["symbolic"],
                "avg_gap": w["avg_gap"],
                "avg_entropy": w["avg_entropy"],
                "avg_logprob": w["avg_logprob"],
                "delta": round(out["delta"], 4),
                "xi": round(out["xi"], 4),
                "lambda": round(out["lambda"], 4),
                "theta_eff": round(out["theta_eff"], 4),
                "integrity": round(out["dominance"].integrity, 4),
                "anomaly": round(out["anomaly_index"], 4),
                "regime": out["regime"].name,
                "health": out["status"].health.name,
            }
        )

    return {
        "prompt_label": label,
        "prompt": prompt,
        "response_preview": text[:200] + ("..." if len(text) > 200 else ""),
        "n_tokens": len(signals),
        "n_windows": len(windows),
        "steps": steps,
        "final_integrity": steps[-1]["integrity"] if steps else None,
        "final_xi": steps[-1]["xi"] if steps else None,
        "final_lambda": steps[-1]["lambda"] if steps else None,
        "final_regime": steps[-1]["regime"] if steps else None,
        "avg_entropy": sum(s["avg_entropy"] for s in steps) / len(steps) if steps else None,
    }


def active_prompt_groups():
    groups = [("clean", CLEAN_PROMPTS), ("stress", STRESS_PROMPTS)]
    if PRAMA_PROMPT_LIMIT > 0:
        return [(category, prompts[:PRAMA_PROMPT_LIMIT]) for category, prompts in groups]
    return groups


def print_start(prompt_groups):
    clean_count = sum(len(p) for c, p in prompt_groups if c == "clean")
    stress_count = sum(len(p) for c, p in prompt_groups if c == "stress")

    print("=" * 60)
    print("  PRAMA Protokol - Pipeline OpenAI")
    print("=" * 60)
    print(f"  Modelo: {MODEL}")
    print(f"  Top logprobs: {TOP_LOGPROBS}")
    print(f"  Prompts: {clean_count} clean + {stress_count} stress")
    print("=" * 60)


def save_results(all_results, results_dir, timestamp):
    json_path = results_dir / f"openai_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    csv_path = results_dir / f"openai_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "label",
            "category",
            "n_tokens",
            "final_integrity",
            "final_xi",
            "final_lambda",
            "final_regime",
            "avg_entropy",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow(
                {
                    "label": r["prompt_label"],
                    "category": r["category"],
                    "n_tokens": r["n_tokens"],
                    "final_integrity": r["final_integrity"],
                    "final_xi": r["final_xi"],
                    "final_lambda": r["final_lambda"],
                    "final_regime": r["final_regime"],
                    "avg_entropy": r["avg_entropy"],
                }
            )

    detail_path = results_dir / f"openai_detail_{timestamp}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "label",
            "category",
            "window",
            "dynamic_in",
            "symbolic_in",
            "avg_gap",
            "avg_entropy",
            "avg_logprob",
            "delta",
            "xi",
            "lambda",
            "theta_eff",
            "integrity",
            "anomaly",
            "regime",
            "health",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in all_results:
            for s in r["steps"]:
                row = {"label": r["prompt_label"], "category": r["category"]}
                row.update(s)
                w.writerow(row)

    print(f"\n  -> {json_path}")
    print(f"  -> {csv_path}")
    print(f"  -> {detail_path}")


def print_summary(all_results):
    clean_r = [r for r in all_results if r["category"] == "clean"]
    stress_r = [r for r in all_results if r["category"] == "stress"]

    def avg(lst, key):
        values = [r[key] for r in lst if r[key] is not None]
        return sum(values) / len(values) if values else 0.0

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Clean':>10} {'Stress':>10} {'Sep':>8}")
    print("-" * 53)

    for label, key in [
        ("Integrity", "final_integrity"),
        ("Xi tension", "final_xi"),
        ("Lambda permissivity", "final_lambda"),
        ("Mean entropy", "avg_entropy"),
    ]:
        c = avg(clean_r, key)
        s = avg(stress_r, key)
        sep = abs(c - s)
        print(f"  {label:<23} {c:10.4f} {s:10.4f} {sep:7.4f}")

    print("-" * 53)

    int_c = avg(clean_r, "final_integrity")
    int_s = avg(stress_r, "final_integrity")
    xi_c = avg(clean_r, "final_xi")
    xi_s = avg(stress_r, "final_xi")
    separation = abs(int_c - int_s) + abs(xi_c - xi_s)

    print(f"\n  Combined separation: {separation:.4f}")
    if separation > 0.05:
        print("  OK PRAMA discriminates these trajectories")
    elif separation > 0.01:
        print("  WARN weak separation - tune parameters")
    else:
        print("  ERROR no separation - inspect mapping")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("=" * 60)
        print("  PRAMA Protokol - Pipeline OpenAI")
        print("=" * 60)
        print()
        print("  Missing OPENAI_API_KEY.")
        print()
        print("  PowerShell:")
        print('     $env:OPENAI_API_KEY="your-key"')
        print("     prama-openai")
        print()
        print("  Optional:")
        print('     $env:OPENAI_MODEL="gpt-4o-mini"')
        print('     $env:PRAMA_PROMPT_LIMIT="1"')
        print("=" * 60)
        return 0

    try:
        client = make_client()
    except Exception as exc:
        print(f"ERROR: {explain_openai_error(exc)}")
        return 1

    cfg = make_config()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prompt_groups = active_prompt_groups()
    print_start(prompt_groups)

    all_results = []

    for category, prompts in prompt_groups:
        title = "CLEAN" if category == "clean" else "STRESS"
        print(f"\n> Running {title} prompts...")

        for i, prompt in enumerate(prompts):
            label = f"{category}_{i + 1}"
            print(f"  [{label}] {prompt[:55]}...")

            try:
                text, logprobs_result = call_openai(client, prompt)
                result = run_through_prama(label, prompt, text, logprobs_result, cfg)

                if result:
                    result["category"] = category
                    all_results.append(result)
                    print(
                        "    -> "
                        f"{result['n_tokens']} tok, "
                        f"integrity={result['final_integrity']}, "
                        f"xi={result['final_xi']}, "
                        f"regime={result['final_regime']}"
                    )
                else:
                    print("    WARN no logprobs in response")

            except Exception as exc:
                print(f"    ERROR: {explain_openai_error(exc)}")

            time.sleep(REQUEST_SLEEP_SECONDS)

    if not all_results:
        print("\nERROR No results obtained.")
        print("  Probable causes:")
        print("  - The selected model did not expose logprobs.")
        print("  - Quota or rate limit was reached.")
        print("  - OPENAI_API_KEY is missing or invalid.")
        return 1

    print_summary(all_results)
    save_results(all_results, results_dir, timestamp)

    print(f"\n{'=' * 60}")
    print("  Done. Open the CSV files in results/ to inspect the data.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
