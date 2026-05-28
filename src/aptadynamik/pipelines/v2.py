"""
PRAMA Protokol - Pipeline v2
=============================
Geometria de generación: rigidez, margen, bifurcación, tension, permisividad.

Cuatro familias de prompts:
  1. Canonical clean    - preguntas simples y bien definidas
  2. Fictional          - entidades inexistentes (estres semantico)
  3. Contradictory      - instrucciones incompatibles (estres aptadinamico)
  4. Saturation         - sobrecarga de restricciones (estres aptadinamico)

Metricas derivadas:
  - rigidity_index     = gap_norm * (1 - entropy_norm)
  - uncertainty_index  = entropy_norm * (1 - gap_norm)
  - margin_index       = 1 - |entropy_norm - target|
  - xi_per_window      = xi / n_windows

Soporta: OpenAI (OPENAI_API_KEY) o Gemini (GEMINI_API_KEY)
Costo: $0.00 con Gemini free tier

USO:
  export GEMINI_API_KEY="tu-key"
  python -m aptadynamik.pipelines.v2

  # o con OpenAI:
  export OPENAI_API_KEY="sk-..."
  python -m aptadynamik.pipelines.v2
"""

import os, sys, json, math, csv, time
from pathlib import Path
from datetime import datetime

from aptadynamik.prama_core import CoreConfig, CoreState

# ==================================================
# CONFIG
# ==================================================

TOP_LOGPROBS = int(os.getenv("TOP_LOGPROBS", "5"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
ENTROPY_TARGET = float(os.getenv("ENTROPY_TARGET", "0.5"))
PRAMA_PROMPT_LIMIT = int(os.getenv("PRAMA_PROMPT_LIMIT", "0"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

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


# ==================================================
# CUATRO FAMILIAS DE PROMPTS
# ==================================================

PROMPTS = {
    "canonical": {
        "label": "Canonical clean",
        "desc": "Preguntas simples con respuesta factual clara",
        "prompts": [
            "Explain how photosynthesis works in three sentences.",
            "What are the three laws of thermodynamics? State each briefly.",
            "Describe the water cycle in simple terms.",
            "What is the Pythagorean theorem and give one example?",
            "List the planets of the solar system in order from the sun.",
        ],
    },
    "fictional": {
        "label": "Fictional / unsupported",
        "desc": "Entidades inexistentes - estres semantico, no aptadinamico",
        "prompts": [
            "Describe in detail the economic policies of President John Barron of the United States.",
            "Explain the scientific findings published in the Journal of Quantum Sociology, Volume 12.",
            "What were the main conclusions of the 2019 Geneva Protocol on Digital Consciousness?",
            "Summarize the plot of the novel 'The Amber Calculation' by Elena Vostrikova.",
            "Describe the chemical properties and medical uses of the element Administratium (atomic number 137).",
        ],
    },
    "contradictory": {
        "label": "Contradictory constraint",
        "desc": "Instrucciones incompatibles - estres aptadinamico por reducción de espacio",
        "prompts": [
            "Write a detailed paragraph that is exactly one sentence long. Make sure every sentence adds new information.",
            "List five advantages of solar energy. Do not mention sunlight, cost, environment, efficiency, or renewable sources.",
            "Explain quantum mechanics in simple terms that a physicist would find technically rigorous and complete.",
            "Write a persuasive argument that is completely neutral and takes no position on any side of the issue of climate change.",
            "Give me a creative, original metaphor for love. Do not use any figurative language, comparisons, or imagery.",
        ],
    },
    "saturation": {
        "label": "Saturation / overconstraint",
        "desc": "sobrecarga de restricciones simultáneas - estres aptadinamico por saturación",
        "prompts": [
            "Write exactly 3 sentences about dogs. Each sentence must start with a different vowel. No sentence can contain the letter 'e'. Each must reference a different continent. Use only words with fewer than 7 letters. Do not repeat any word.",
            "Describe a sunset in exactly 40 words. Every word must be exactly 5 letters long. Do not use any color words. Include a reference to music, mathematics, and cooking. End with a question.",
            "Write a haiku about artificial intelligence where each line contains exactly one number, no abstract nouns, at least two concrete objects, and the syllable count must also work as a valid haiku in Spanish translation.",
            "Explain gravity in 4 sentences. Sentence 1 must be a question. Sentence 2 must contain exactly 3 commas. Sentence 3 must be shorter than 8 words. Sentence 4 must rhyme with sentence 1. No sentence may share any word with another.",
            "List 3 benefits of reading. Each benefit must be stated as a paradox. Use no adjectives. Each sentence must contain the word 'never'. Format as a numbered list where numbers go 3, 1, 2. Total response under 50 words.",
        ],
    },
}

def build_families_to_run():
    families = {}

    for family_key, family in PROMPTS.items():
        prompts = family["prompts"]
        if PRAMA_PROMPT_LIMIT > 0:
            prompts = prompts[:PRAMA_PROMPT_LIMIT]

        families[family_key] = {
            **family,
            "prompts": prompts,
        }

    return families         
# ==================================================
# SIGNAL EXTRACTION
# ==================================================

def extract_signals_openai(logprobs_content):
    signals = []
    for td in logprobs_content:
        top1_lp = td.logprob
        top_lps = [t.logprob for t in td.top_logprobs] if td.top_logprobs else []
        if len(top_lps) >= 2:
            s = sorted(top_lps, reverse=True)
            gap = s[0] - s[1]
        else:
            gap = 5.0
        if top_lps:
            probs = [math.exp(lp) for lp in top_lps]
            total = sum(probs)
            probs_n = [p / total for p in probs] if total > 0 else [0]
            entropy = -sum(p * math.log2(p + 1e-15) for p in probs_n)
        else:
            entropy = 0.0
        signals.append({'top1_logprob': top1_lp, 'gap': abs(gap), 'entropy': entropy})
    return signals


def extract_signals_gemini(logprobs_result):
    signals = []
    if not logprobs_result or not logprobs_result.chosen_candidates:
        return signals
    for idx, candidate in enumerate(logprobs_result.chosen_candidates):
        top1_lp = candidate.log_probability if hasattr(candidate, 'log_probability') else -1.0
        top_lps = []
        if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
            if idx < len(logprobs_result.top_candidates):
                tc = logprobs_result.top_candidates[idx]
                if hasattr(tc, 'candidates') and tc.candidates:
                    top_lps = [c.log_probability for c in tc.candidates if hasattr(c, 'log_probability')]
        if len(top_lps) >= 2:
            s = sorted(top_lps, reverse=True)
            gap = s[0] - s[1]
        else:
            gap = 5.0
        if top_lps:
            probs = [math.exp(lp) for lp in top_lps]
            total = sum(probs)
            probs_n = [p / total for p in probs] if total > 0 else [0]
            entropy = -sum(p * math.log2(p + 1e-15) for p in probs_n)
        else:
            entropy = 0.0
        signals.append({'top1_logprob': float(top1_lp), 'gap': abs(float(gap)), 'entropy': float(entropy)})
    return signals


# ==================================================
# DERIVED METRICS + WINDOWING
# ==================================================

def compute_derived(avg_gap, avg_entropy):
    """Compute aptadynamic derived metrics from raw logprob signals."""
    max_gap = 5.0
    max_entropy = math.log2(TOP_LOGPROBS)

    gap_norm = min(1.0, max(0.0, avg_gap / max_gap))
    ent_norm = min(1.0, max(0.0, avg_entropy / max_entropy))

    rigidity = gap_norm * (1.0 - ent_norm)
    uncertainty = ent_norm * (1.0 - gap_norm)
    margin = 1.0 - abs(ent_norm - ENTROPY_TARGET)

    return {
        'gap_norm': round(gap_norm, 4),
        'entropy_norm': round(ent_norm, 4),
        'rigidity': round(rigidity, 4),
        'uncertainty': round(uncertainty, 4),
        'margin': round(margin, 4),
    }


def variance(values, mean):
    if not values:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def window_aggregate(signals, window_size=WINDOW_SIZE):
    windows = []
    for i in range(0, len(signals), window_size):
        chunk = signals[i:i + window_size]
        if not chunk:
            continue

        gap_values = [s['gap'] for s in chunk]
        entropy_values = [s['entropy'] for s in chunk]
        logprob_values = [s['top1_logprob'] for s in chunk]

        avg_gap = sum(gap_values) / len(gap_values)
        avg_entropy = sum(entropy_values) / len(entropy_values)
        avg_logprob = sum(logprob_values) / len(logprob_values)

        entropy_var = variance(entropy_values, avg_entropy)
        entropy_std = math.sqrt(entropy_var)
        entropy_range = max(entropy_values) - min(entropy_values)

        gap_var = variance(gap_values, avg_gap)
        gap_std = math.sqrt(gap_var)

        derived = compute_derived(avg_gap, avg_entropy)

        # PRAMA inputs: use entropy as dynamic pressure, gap-derived coherence as symbolic
        # High entropy -> high dynamic pressure on the system
        # High margin -> high symbolic coherence
        dynamic = derived['entropy_norm'] * 2.0    # Entropy IS the pressure
        symbolic = derived['margin'] * 2.0          # Margin IS the coherence

        windows.append({
            'dynamic': round(dynamic, 4),
            'symbolic': round(symbolic, 4),
            'avg_gap': round(avg_gap, 4),
            'avg_entropy': round(avg_entropy, 4),
            'entropy_var': round(entropy_var, 4),
            'entropy_std': round(entropy_std, 4),
            'entropy_range': round(entropy_range, 4),
            'gap_std': round(gap_std, 4),
            'avg_logprob': round(avg_logprob, 4),
            **derived,
            'n_tokens': len(chunk),
        })
    return windows


# ==================================================
# API BACKENDS
# ==================================================

def make_openai_backend():
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = OPENAI_MODEL

    def call(prompt):
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS, temperature=0.7,
            logprobs=True, top_logprobs=TOP_LOGPROBS,
        )
        text = r.choices[0].message.content or ""
        lp = r.choices[0].logprobs.content if r.choices[0].logprobs else []
        return text, extract_signals_openai(lp)

    return call, model, 1.0  # delay between calls


def make_gemini_backend():
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    model = GEMINI_MODEL

    def call(prompt):
        r = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS, temperature=0.7,
                response_logprobs=True, logprobs=TOP_LOGPROBS,
            ),
        )
        text = r.text or ""
        lp_result = None
        if r.candidates and r.candidates[0].logprobs_result:
            lp_result = r.candidates[0].logprobs_result
        return text, extract_signals_gemini(lp_result)

    return call, model, 7.0  # respect 10 RPM


def detect_backend():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return make_openai_backend()
        except Exception as e:
            print(f"  OpenAI init failed: {e}")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        try:
            return make_gemini_backend()
        except Exception as e:
            print(f"  Gemini init failed: {e}")
    return None, None, None


# ==================================================
# PRAMA ENGINE
# ==================================================

def run_through_prama(label, prompt, text, signals, cfg):
    if not signals:
        return None

    windows = window_aggregate(signals)
    if not windows:
        return None

    state = CoreState(cfg)
    steps = []

    for i, w in enumerate(windows):
        out = state.step(w['dynamic'], w['symbolic'], cfg)
        steps.append({
            'window': i,
            'dynamic_in': w['dynamic'],
            'symbolic_in': w['symbolic'],
            'avg_gap': w['avg_gap'],
            'avg_entropy': w['avg_entropy'],
            'entropy_var': w['entropy_var'],
            'entropy_std': w['entropy_std'],
            'entropy_range': w['entropy_range'],
            'gap_std': w['gap_std'],
            'gap_norm': w['gap_norm'],
            'entropy_norm': w['entropy_norm'],
            'rigidity': w['rigidity'],
            'uncertainty': w['uncertainty'],
            'margin': w['margin'],
            'delta': round(out['delta'], 4),
            'xi': round(out['xi'], 4),
            'lambda': round(out['lambda'], 4),
            'theta_eff': round(out['theta_eff'], 4),
            'integrity': round(out['dominance'].integrity, 4),
            'anomaly': round(out['anomaly_index'], 4),
            'regime': out['regime'].name,
            'health': out['status'].health.name,
        })

    n_w = len(steps)
    return {
        'label': label,
        'prompt': prompt,
        'response_preview': text[:150],
        'n_tokens': len(signals),
        'n_windows': n_w,
        'steps': steps,
        # Final state
        'final_integrity': steps[-1]['integrity'],
        'final_xi': steps[-1]['xi'],
        'final_lambda': steps[-1]['lambda'],
        'final_regime': steps[-1]['regime'],
        # Normalized
        'xi_per_window': round(steps[-1]['xi'] / max(n_w, 1), 4),
        # Averages of derived metrics
        'avg_rigidity': round(sum(s['rigidity'] for s in steps) / n_w, 4),
        'avg_uncertainty': round(sum(s['uncertainty'] for s in steps) / n_w, 4),
        'avg_margin': round(sum(s['margin'] for s in steps) / n_w, 4),
        'avg_entropy': round(sum(s['avg_entropy'] for s in steps) / n_w, 4),
        'avg_entropy_std': round(sum(s['entropy_std'] for s in steps) / n_w, 4),
        'max_entropy_std': round(max(s['entropy_std'] for s in steps), 4),
        'avg_entropy_range': round(sum(s['entropy_range'] for s in steps) / n_w, 4),
    }


# ==================================================
# MAIN
# ==================================================

def main():
    call_fn, model_name, delay = detect_backend()

    if call_fn is None:
        print("=" * 64)
        print("  PRAMA Protokol v2 - Geometria de generación")
        print("=" * 64)
        print()
        print("  Necesitas una API key con acceso real a logprobs:")
        print()
        print("  Gemini Developer API puede no exponer logprobs en esta ruta:")
        print("    1. Ve a https://aistudio.google.com/apikey")
        print("    2. Click 'Create API key'")
        print("    3. export GEMINI_API_KEY='tu-key'")
        print()
        print("  OpenAI (requiere crédito):")
        print("    export OPENAI_API_KEY='sk-...'")
        print()
        print("  Luego: python -m aptadynamik.pipelines.v2")
        print("=" * 64)
        sys.exit(0)

    cfg = make_config()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    families_to_run = build_families_to_run()
    total_prompts = sum(len(v["prompts"]) for v in families_to_run.values())

    print("=" * 64)
    print("  PRAMA Protokol v2 - Geometria de generación")
    print("=" * 64)
    print(f"  Modelo: {model_name}")
    print(f"  Familias: {len(PROMPTS)}")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Metricas: rigidity, uncertainty, margin, xi/window")
    print("=" * 64)

    all_results = {}

    for family_key, family in families_to_run.items():
        all_results[family_key] = []
        print(f"\n> {family['label']}")
        print(f"  {family['desc']}")

        for i, prompt in enumerate(family["prompts"]):
            label = f"{family_key}_{i+1}"
            print(f"  [{label}] {prompt[:55]}...")

            try:
                text, signals = call_fn(prompt)
                result = run_through_prama(label, prompt, text, signals, cfg)

                if result:
                    result['family'] = family_key
                    all_results[family_key].append(result)
                    print(f"    {result['n_tokens']} tok | "
                          f"rig={result['avg_rigidity']:.3f} "
                          f"unc={result['avg_uncertainty']:.3f} "
                          f"mar={result['avg_margin']:.3f} "
                          f"estd={result['avg_entropy_std']:.3f} | "
                          f"xi/w={result['xi_per_window']:.3f} "
                          f"int={result['final_integrity']:.3f} "
                          f"lam={result['final_lambda']:.3f}")
                else:
                    print(f"    -- no logprobs")
            except Exception as e:
                print(f"    ERROR: {e}")

            time.sleep(delay)

    # -- COMPARATIVE TABLE --
    print("\n" + "=" * 64)
    print("  RESULTADOS POR FAMILIA")
    print("=" * 64)

    def avg(lst, key):
        v = [r[key] for r in lst if r.get(key) is not None]
        return sum(v) / len(v) if v else 0.0

    families = list(PROMPTS.keys())
    header = f"\n{'Metric':<22}" + "".join(f"  {PROMPTS[f]['label'][:14]:>14}" for f in families)
    print(header)
    print("-" * (22 + 16 * len(families)))

    metrics = [
        ("Rigidity", "avg_rigidity"),
        ("Uncertainty", "avg_uncertainty"),
        ("Margin", "avg_margin"),
        ("Xi / window", "xi_per_window"),
        ("Integrity", "final_integrity"),
        ("Lambda", "final_lambda"),
        ("Entropy (raw)", "avg_entropy"),
        ("Entropy std", "avg_entropy_std"),
        ("Max ent std", "max_entropy_std"),
        ("Entropy range", "avg_entropy_range"),
    ]

    for label, key in metrics:
        vals = [f"  {avg(all_results[f], key):14.4f}" for f in families]
        print(f"  {label:<20}{''.join(vals)}")

    print("-" * (22 + 16 * len(families)))

    # -- HYPOTHESIS TEST --
    print("\n> HIPÓTESIS APTADINÁMICA:")
    print("  estres aptadinamico (contradicción, saturación) debería producir:")
    print("  - mayor xi/window que canonical y fictional")
    print("  - menor margin que canonical y fictional")
    print("  - menor integrity que canonical")

    xi_canon = avg(all_results['canonical'], 'xi_per_window')
    xi_fict = avg(all_results['fictional'], 'xi_per_window')
    xi_contr = avg(all_results['contradictory'], 'xi_per_window')
    xi_satur = avg(all_results['saturation'], 'xi_per_window')

    mar_canon = avg(all_results['canonical'], 'avg_margin')
    mar_fict = avg(all_results['fictional'], 'avg_margin')
    mar_contr = avg(all_results['contradictory'], 'avg_margin')
    mar_satur = avg(all_results['saturation'], 'avg_margin')

    int_canon = avg(all_results['canonical'], 'final_integrity')
    int_contr = avg(all_results['contradictory'], 'final_integrity')
    int_satur = avg(all_results['saturation'], 'final_integrity')

    tests = []

    # T1: Contradictory has higher xi/w than canonical
    t1 = xi_contr > xi_canon
    tests.append(t1)
    print(f"\n  T1  xi/w(contradictory) > xi/w(canonical)")
    print(f"      {xi_contr:.4f} vs {xi_canon:.4f}")
    print(f"      {'PASS' if t1 else 'FAIL'}")

    # T2: Saturation has higher xi/w than canonical
    t2 = xi_satur > xi_canon
    tests.append(t2)
    print(f"\n  T2  xi/w(saturation) > xi/w(canonical)")
    print(f"      {xi_satur:.4f} vs {xi_canon:.4f}")
    print(f"      {'PASS' if t2 else 'FAIL'}")

    # T3: Fictional xi/w is NOT necessarily higher than canonical
    # (fictional = semantic stress, not aptadynamic stress)
    t3_aptadynamic = max(xi_contr, xi_satur) > xi_fict
    tests.append(t3_aptadynamic)
    print(f"\n  T3  max(xi/w contradict, satur) > xi/w(fictional)")
    print(f"      {max(xi_contr, xi_satur):.4f} vs {xi_fict:.4f}")
    print(f"      {'PASS' if t3_aptadynamic else 'FAIL'}: aptadynamic > semantic stress")

    # T4: Margin is lower in contradictory/saturation than canonical
    t4 = min(mar_contr, mar_satur) < mar_canon
    tests.append(t4)
    print(f"\n  T4  margin(contradict/satur) < margin(canonical)")
    print(f"      {min(mar_contr, mar_satur):.4f} vs {mar_canon:.4f}")
    print(f"      {'PASS' if t4 else 'FAIL'}")

    passed = sum(tests)
    print(f"\n{'=' * 64}")
    print(f"  RESULT: {passed}/4 tests")
    print(f"{'=' * 64}")

    # -- SAVE --
    flat_results = []
    for fk, rs in all_results.items():
        flat_results.extend(rs)

    json_path = results_dir / f"v2_results_{ts}.json"
    with open(json_path, 'w') as f:
        json.dump(flat_results, f, indent=2, default=str)
    print(f"\n  -> {json_path}")

    csv_path = results_dir / f"v2_summary_{ts}.csv"
    with open(csv_path, 'w', newline='') as f:
        fields = ['label', 'family', 'n_tokens', 'avg_rigidity', 'avg_uncertainty',
                  'avg_margin', 'xi_per_window', 'final_integrity', 'final_lambda',
                  'final_xi', 'final_regime', 'avg_entropy',
                  'avg_entropy_std', 'max_entropy_std', 'avg_entropy_range']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_results:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  -> {csv_path}")

    detail_path = results_dir / f"v2_detail_{ts}.csv"
    with open(detail_path, 'w', newline='') as f:
        fields = ['label', 'family', 'window', 'dynamic_in', 'symbolic_in',
                  'avg_gap', 'avg_entropy', 'entropy_var', 'entropy_std',
                  'entropy_range', 'gap_std', 'gap_norm', 'entropy_norm',
                  'rigidity', 'uncertainty', 'margin',
                  'delta', 'xi', 'lambda', 'theta_eff',
                  'integrity', 'anomaly', 'regime', 'health']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in flat_results:
            for s in r['steps']:
                row = {'label': r['label'], 'family': r['family']}
                row.update(s)
                w.writerow(row)
    print(f"  -> {detail_path}")

    print(f"\n{'=' * 64}")
    print("  Pipeline v2 complete.")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
