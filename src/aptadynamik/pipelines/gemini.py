"""
PRAMA Protokol — Pipeline Gemini (GRATIS)
==========================================
Conecta Gemini API free tier (con logprobs) al motor PRAMA.

REQUISITOS:
  pip install google-genai

USO:
  1. Ve a https://aistudio.google.com/apikey
  2. Click "Create API key" (NO necesita tarjeta de crédito)
  3. Copia la key
  4. Ejecuta:

     export GEMINI_API_KEY="tu-key-aqui"
     prama-gemini

COSTO: $0.00 — free tier, sin tarjeta, sin límite de tiempo.
"""

import os, json, math, csv, time
from pathlib import Path
from datetime import datetime

from aptadynamik.prama_core import CoreConfig, CoreState

# ══════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════

MODEL = "gemini-2.5-flash"   # Free tier: 10 RPM, 250 RPD
TOP_LOGPROBS = 5
WINDOW_SIZE = 8
MAX_TOKENS = 256

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

# ══════════════════════════════════════════════════
# EXTRACCIÓN DE SEÑALES
# ══════════════════════════════════════════════════

def extract_signals(logprobs_result):
    """Extrae señales por token desde logprobs de Gemini."""
    signals = []
    if not logprobs_result or not logprobs_result.chosen_candidates:
        return signals

    for candidate in logprobs_result.chosen_candidates:
        token_text = candidate.token if hasattr(candidate, 'token') else '?'
        top1_lp = candidate.log_probability if hasattr(candidate, 'log_probability') else -1.0

        # Top candidates
        top_lps = []
        if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates:
            # Match by index
            idx = len(signals)
            if idx < len(logprobs_result.top_candidates):
                top_cands = logprobs_result.top_candidates[idx]
                if hasattr(top_cands, 'candidates') and top_cands.candidates:
                    top_lps = [c.log_probability for c in top_cands.candidates
                               if hasattr(c, 'log_probability')]

        # Gap
        if len(top_lps) >= 2:
            sorted_lps = sorted(top_lps, reverse=True)
            gap = sorted_lps[0] - sorted_lps[1]
        else:
            gap = 5.0

        # Entropía de Shannon
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

        signals.append({
            'token': str(token_text),
            'top1_logprob': float(top1_lp),
            'gap': abs(float(gap)),
            'entropy': float(entropy),
        })

    return signals


def window_aggregate(signals, window_size=WINDOW_SIZE):
    """Agrega señales por ventanas → (dynamic_input, symbolic_input)."""
    windows = []
    max_gap = 5.0

    for i in range(0, len(signals), window_size):
        chunk = signals[i:i + window_size]
        if not chunk:
            continue

        avg_gap = sum(s['gap'] for s in chunk) / len(chunk)
        avg_entropy = sum(s['entropy'] for s in chunk) / len(chunk)
        avg_logprob = sum(s['top1_logprob'] for s in chunk) / len(chunk)

        dynamic = max(0.0, max_gap - avg_gap) / max_gap * 2.0
        max_entropy = math.log2(TOP_LOGPROBS)
        symbolic = max(0.0, max_entropy - avg_entropy) / max(max_entropy, 0.01) * 2.0

        windows.append({
            'dynamic': round(dynamic, 4),
            'symbolic': round(symbolic, 4),
            'avg_gap': round(avg_gap, 4),
            'avg_entropy': round(avg_entropy, 4),
            'avg_logprob': round(avg_logprob, 4),
            'n_tokens': len(chunk),
        })
    return windows


# ══════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════

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
    "Describe the chemical properties and medical uses of the element Administratium (atomic number 137).",
]

# ══════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════

def call_gemini(client, prompt, types):
    """Llama a Gemini con logprobs activados."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS,
            temperature=0.7,
            response_logprobs=True,
            logprobs=TOP_LOGPROBS,
        ),
    )

    text = response.text if response.text else ""

    # Extraer logprobs
    logprobs_result = None
    if response.candidates and response.candidates[0].logprobs_result:
        logprobs_result = response.candidates[0].logprobs_result

    return text, logprobs_result


def run_through_prama(label, prompt, text, logprobs_result, cfg):
    """Procesa logprobs a través del motor PRAMA."""
    signals = extract_signals(logprobs_result)
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
            'avg_logprob': w['avg_logprob'],
            'delta': round(out['delta'], 4),
            'xi': round(out['xi'], 4),
            'lambda': round(out['lambda'], 4),
            'theta_eff': round(out['theta_eff'], 4),
            'integrity': round(out['dominance'].integrity, 4),
            'anomaly': round(out['anomaly_index'], 4),
            'regime': out['regime'].name,
            'health': out['status'].health.name,
        })

    return {
        'prompt_label': label,
        'prompt': prompt,
        'response_preview': text[:200] + ('...' if len(text) > 200 else ''),
        'n_tokens': len(signals),
        'n_windows': len(windows),
        'steps': steps,
        'final_integrity': steps[-1]['integrity'] if steps else None,
        'final_xi': steps[-1]['xi'] if steps else None,
        'final_lambda': steps[-1]['lambda'] if steps else None,
        'final_regime': steps[-1]['regime'] if steps else None,
        'avg_entropy': sum(s['avg_entropy'] for s in steps) / len(steps) if steps else None,
    }


def main():
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print('Instala el SDK:')
        print('  pip install google-genai')
        return 1

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("=" * 60)
        print("  PRAMA Protokol — Pipeline Gemini (GRATIS)")
        print("=" * 60)
        print()
        print("  1. Ve a https://aistudio.google.com/apikey")
        print("  2. Click 'Create API key' (sin tarjeta)")
        print("  3. Copia la key")
        print("  4. Ejecuta:")
        print()
        print('     export GEMINI_API_KEY="tu-key"')
        print('     prama-gemini')
        print()
        print("  Costo: $0.00")
        print("=" * 60)
        return 0

    client = genai.Client(api_key=api_key)
    cfg = make_config()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  PRAMA Protokol — Pipeline Gemini")
    print("=" * 60)
    print(f"  Modelo: {MODEL} (free tier)")
    print(f"  Prompts: {len(CLEAN_PROMPTS)} limpios + {len(STRESS_PROMPTS)} estresantes")
    print("=" * 60)

    all_results = []

    for category, prompts in [('clean', CLEAN_PROMPTS), ('stress', STRESS_PROMPTS)]:
        print(f"\n▸ Ejecutando prompts {'LIMPIOS' if category == 'clean' else 'ESTRESANTES'}...")
        for i, prompt in enumerate(prompts):
            label = f"{category}_{i+1}"
            print(f"  [{label}] {prompt[:55]}...")
            try:
                text, logprobs_result = call_gemini(client, prompt, types)
                result = run_through_prama(label, prompt, text, logprobs_result, cfg)
                if result:
                    result['category'] = category
                    all_results.append(result)
                    print(f"    → {result['n_tokens']} tok, integrity={result['final_integrity']}, ξ={result['final_xi']}, regime={result['final_regime']}")
                else:
                    print(f"    ⚠ Sin logprobs en la respuesta")
            except Exception as e:
                print(f"    ✗ Error: {e}")
            time.sleep(7)  # Respetar rate limit: 10 RPM = 1 cada 6s

    if not all_results:
        print("\n✗ No se obtuvieron resultados. Verifica tu API key.")
        return 1

    # ── Resumen ──
    clean_r = [r for r in all_results if r['category'] == 'clean']
    stress_r = [r for r in all_results if r['category'] == 'stress']

    def avg(lst, key):
        v = [r[key] for r in lst if r[key] is not None]
        return sum(v) / len(v) if v else 0

    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"\n{'Métrica':<25} {'Limpios':>10} {'Estrés':>10} {'Separa':>8}")
    print("─" * 53)

    for label, key in [
        ('Integrity', 'final_integrity'),
        ('Ξ (tensión)', 'final_xi'),
        ('λ (permisividad)', 'final_lambda'),
        ('Entropía media', 'avg_entropy'),
    ]:
        c, s = avg(clean_r, key), avg(stress_r, key)
        sep = abs(c - s)
        print(f"  {label:<23} {c:10.4f} {s:10.4f} {sep:7.4f}")

    print("─" * 53)

    int_c = avg(clean_r, 'final_integrity')
    int_s = avg(stress_r, 'final_integrity')
    xi_c = avg(clean_r, 'final_xi')
    xi_s = avg(stress_r, 'final_xi')
    separation = abs(int_c - int_s) + abs(xi_c - xi_s)

    print(f"\n  Separación combinada: {separation:.4f}")
    if separation > 0.05:
        print("  ✓ EL MOTOR DISCRIMINA")
    elif separation > 0.01:
        print("  ~ Separación débil — ajustar parámetros")
    else:
        print("  ✗ Sin separación — revisar mapeo")

    # ── Guardar ──
    json_path = results_dir / f"gemini_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  → {json_path}")

    csv_path = results_dir / f"gemini_summary_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'label','category','n_tokens','final_integrity',
            'final_xi','final_lambda','final_regime','avg_entropy'
        ])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"  → {csv_path}")

    detail_path = results_dir / f"gemini_detail_{timestamp}.csv"
    with open(detail_path, 'w', newline='') as f:
        fields = ['label','category','window','dynamic_in','symbolic_in',
                  'avg_gap','avg_entropy','avg_logprob','delta','xi','lambda',
                  'theta_eff','integrity','anomaly','regime','health']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            for s in r['steps']:
                row = {'label': r['prompt_label'], 'category': r['category']}
                row.update(s)
                w.writerow(row)
    print(f"  → {detail_path}")

    print(f"\n{'=' * 60}")
    print("  Listo. Abre los CSV para ver los datos.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    raise SystemExit(main())



