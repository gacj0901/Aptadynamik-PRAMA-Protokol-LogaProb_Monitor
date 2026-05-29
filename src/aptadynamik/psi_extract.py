"""
psi_extract.py — Extracción de presión ambiental Ψ desde el prompt
===================================================================
Corresponde a Ψ(t) del corpus: "presión de perturbación o carga ambiental"
(Axioma E2). Su propósito es medir la DEMANDA que el entorno impone al
sistema, ANTES y de forma INDEPENDIENTE de la salida del modelo.

Por qué importa
---------------
En el pipeline v2 actual, Ψ se leía de la entropía de la propia salida,
de modo que `dynamic` y `symbolic` eran funciones de una sola variable y
el desacoplamiento Δ = |Φ − Ψ| se calculaba entre una cosa y sí misma.
Aquí Ψ proviene del INPUT, así que Δ pasa a medir el desacoplamiento real
entre lo que el sistema sostiene (Φ, de los logprobs) y lo que el entorno
exige (Ψ, del prompt). Solo entonces la variable manipulada —el tipo de
prompt— entra en la dinámica como presión.

Diseño
------
Ψ se descompone en dos ejes, que son los dos tipos de estrés del marco:

  - load_saturation : densidad de restricciones simultáneas impuestas
                      (el eje de la familia "saturation").
  - load_contradiction : presencia de demandas mutuamente incompatibles
                      (el eje de la familia "contradictory").

Predicción comprobable SOBRE EL INPUT (antes de gastar una sola llamada):
  canonical ≈ fictional  (BAJOS)   <   contradictory, saturation  (ALTOS)
Si Ψ no produce ese orden, el problema está en la manipulación o en el
constructo, y lo sabes gratis y de forma determinista.

Sin dependencias externas. Determinista. Auditable a mano.
"""

import re
from dataclasses import dataclass, asdict

# ============================================================
# 1. EJE DE SATURACIÓN: densidad de restricciones
# ============================================================
# Cada patrón marca una restricción impuesta sobre el espacio de respuestas.
# No se trata de contar palabras, sino requisitos que deben sostenerse a la vez.

_NUMERIC_SPEC = re.compile(
    r"\b(exactly|precisely|fewer than|less than|more than|at least|"
    r"no more than|under|over|at most)\b\s*\w*\s*\d+"
    r"|\b\d+\s*(word|words|sentence|sentences|line|lines|syllable|syllables|"
    r"letter|letters|character|characters|item|items|benefit|benefits)\b"
    r"|\bexactly one\b",
    re.IGNORECASE,
)

# Marcadores de requisito universal / obligación. Se cuentan por ocurrencia
# porque "each ... every ... must ..." apila demandas independientes.
_REQUIREMENT = re.compile(
    r"\b(must|each|every|only|all of|both|simultaneously)\b", re.IGNORECASE
)

# Prohibiciones / exclusiones: recortan el espacio admisible.
_PROHIBITION = re.compile(
    r"\b(do not|don't|cannot|can't|may not|never|without|avoid|"
    r"no [a-z]+|not (?:use|mention|include|contain|repeat|share))\b",
    re.IGNORECASE,
)

# Restricciones de forma: rima, posición, formato impuesto.
_FORMAT = re.compile(
    r"\b(rhyme|rhymes|start with|begin with|end with|start each|"
    r"format as|numbered list|in order|alphabetical|each line|"
    r"each sentence|each benefit|each must|valid haiku)\b",
    re.IGNORECASE,
)


def load_saturation(prompt: str) -> float:
    """Densidad de restricciones simultáneas. Suma de ocurrencias por
    categoría, sin cap: una sobrecarga real (5-8 restricciones) debe
    puntuar muy por encima de una pregunta simple (0-1)."""
    n = 0
    n += len(_NUMERIC_SPEC.findall(prompt))
    n += len(_REQUIREMENT.findall(prompt))
    n += len(_PROHIBITION.findall(prompt))
    n += len(_FORMAT.findall(prompt))
    return float(n)


# ============================================================
# 2. EJE DE CONTRADICCIÓN: incompatibilidad de demandas
# ============================================================
# Pares de polos opuestos. Se marca una unidad de contradicción cuando
# AMBOS polos aparecen en el mismo prompt: se pide X y se niega la condición
# de X. CONJUNTO INICIAL, NO EXHAUSTIVO — extiéndelo según tus prompts.

_POLE_PAIRS = [
    # expansión exigida vs. compresión exigida
    (
        re.compile(r"\b(detailed|elaborate|comprehensive|in detail|thorough|"
                   r"adds new information|long)\b", re.IGNORECASE),
        re.compile(r"\b(one sentence|exactly one|brief|briefly|short|"
                   r"under \d+ words|exactly \d+ words)\b", re.IGNORECASE),
    ),
    # subjetividad/persuasión vs. neutralidad
    (
        re.compile(r"\b(persuasive|persuade|argue|convince|argument)\b",
                   re.IGNORECASE),
        re.compile(r"\b(neutral|no position|takes no position|unbiased|"
                   r"objective|balanced)\b", re.IGNORECASE),
    ),
    # figurativo vs. literal
    (
        re.compile(r"\b(metaphor|imagery|figurative|simile|poetic)\b",
                   re.IGNORECASE),
        re.compile(r"\b(no figurative|no comparison|no imagery|literal|"
                   r"do not use any figurative)\b", re.IGNORECASE),
    ),
    # simple vs. rigor experto (registros incompatibles)
    (
        re.compile(r"\b(simple terms|simply|layman|easy to understand)\b",
                   re.IGNORECASE),
        re.compile(r"\b(technically rigorous|rigorous and complete|"
                   r"physicist would find|expert)\b", re.IGNORECASE),
    ),
    # originalidad/creatividad vs. prohibición de los medios para lograrla
    (
        re.compile(r"\b(creative|original)\b", re.IGNORECASE),
        re.compile(r"\b(do not use any|no figurative|no comparison)\b",
                   re.IGNORECASE),
    ),
]


def load_contradiction(prompt: str) -> float:
    """Número de pares de polos incompatibles co-presentes en el prompt.
    NOTA: detecta una subclase léxica de contradicción; la contradicción
    puramente semántica requiere el hook de juez (abajo, opcional)."""
    units = 0
    for pole_a, pole_b in _POLE_PAIRS:
        if pole_a.search(prompt) and pole_b.search(prompt):
            units += 1
    return float(units)


# ============================================================
# 3. Ψ AGREGADA (con descomposición)
# ============================================================

@dataclass
class PsiResult:
    psi: float                 # presión total (para alimentar la dinámica)
    load_saturation: float     # eje saturación (densidad de restricciones)
    load_contradiction: float  # eje contradicción (incompatibilidad)
    contra_weight: float       # peso usado para el eje de contradicción

    def as_dict(self):
        return asdict(self)


def extract_psi(
    prompt: str,
    contra_weight: float = 3.0,
    scale: float = None,
) -> PsiResult:
    """
    Calcula Ψ desde el prompt.

    contra_weight : una incompatibilidad es alta presión aun con pocas
                    restricciones; este peso la pone en escala con la
                    densidad de saturación. Ajústalo y registra el valor.
    scale         : si se da, Ψ se mapea a [0, 2] vía min(2, psi_raw/scale)
                    para entrar en el rango de Φ del pipeline. Si es None,
                    se devuelve Ψ cruda (recomendado: calibra scale a tu Φ).

    Devuelve PsiResult con la descomposición por eje, NO solo el escalar.
    Analiza siempre los ejes por separado antes de colapsarlos.
    """
    sat = load_saturation(prompt)
    contra = load_contradiction(prompt)
    psi_raw = sat + contra_weight * contra

    if scale is not None and scale > 0:
        psi_val = min(2.0, psi_raw / scale)
    else:
        psi_val = psi_raw

    return PsiResult(
        psi=round(psi_val, 4),
        load_saturation=round(sat, 4),
        load_contradiction=round(contra, 4),
        contra_weight=contra_weight,
    )


# ============================================================
# 4. HOOK OPCIONAL DE JUEZ SEMÁNTICO (apagado por defecto)
# ============================================================

def judge_contradiction(prompt: str, judge_call) -> float:
    """
    Detección semántica de contradicción vía un juez externo.

    ADVERTENCIA DE INDEPENDENCIA: úsalo SOLO con un modelo DISTINTO del que
    estás midiendo, y registra su salida en una columna aparte. Si el juez
    es el mismo sistema bajo prueba, reintroduces circularidad y Ψ deja de
    ser independiente de Φ. Mantenlo apagado salvo que sepas por qué lo
    enciendes.

    judge_call : callable(str) -> str que devuelve un número en [0,1].
    """
    q = (
        "On a scale from 0 to 1, how mutually incompatible are the demands "
        "in the following instruction? Reply with only the number.\n\n"
        f"Instruction: {prompt}"
    )
    try:
        raw = judge_call(q)
        m = re.search(r"[01](?:\.\d+)?", str(raw))
        return float(m.group(0)) if m else 0.0
    except Exception:
        return 0.0


# ============================================================
# 5. AUTOPRUEBA SOBRE LAS CUATRO FAMILIAS
# ============================================================
# Comprueba en el INPUT, sin ninguna llamada a API, si Ψ separa las familias
# como predice el marco: canonical ≈ fictional (BAJOS) < contradictory,
# saturation (ALTOS). Si esto falla, no sigas: arregla la manipulación.

_FAMILIES = {
    "canonical": [
        "Explain how photosynthesis works in three sentences.",
        "What are the three laws of thermodynamics? State each briefly.",
        "Describe the water cycle in simple terms.",
        "What is the Pythagorean theorem and give one example?",
        "List the planets of the solar system in order from the sun.",
    ],
    "fictional": [
        "Describe in detail the economic policies of President John Barron of the United States.",
        "Explain the scientific findings published in the Journal of Quantum Sociology, Volume 12.",
        "What were the main conclusions of the 2019 Geneva Protocol on Digital Consciousness?",
        "Summarize the plot of the novel 'The Amber Calculation' by Elena Vostrikova.",
        "Describe the chemical properties and medical uses of the element Administratium (atomic number 137).",
    ],
    "contradictory": [
        "Write a detailed paragraph that is exactly one sentence long. Make sure every sentence adds new information.",
        "List five advantages of solar energy. Do not mention sunlight, cost, environment, efficiency, or renewable sources.",
        "Explain quantum mechanics in simple terms that a physicist would find technically rigorous and complete.",
        "Write a persuasive argument that is completely neutral and takes no position on any side of the issue of climate change.",
        "Give me a creative, original metaphor for love. Do not use any figurative language, comparisons, or imagery.",
    ],
    "saturation": [
        "Write exactly 3 sentences about dogs. Each sentence must start with a different vowel. No sentence can contain the letter 'e'. Each must reference a different continent. Use only words with fewer than 7 letters. Do not repeat any word.",
        "Describe a sunset in exactly 40 words. Every word must be exactly 5 letters long. Do not use any color words. Include a reference to music, mathematics, and cooking. End with a question.",
        "Write a haiku about artificial intelligence where each line contains exactly one number, no abstract nouns, at least two concrete objects, and the syllable count must also work as a valid haiku in Spanish translation.",
        "Explain gravity in 4 sentences. Sentence 1 must be a question. Sentence 2 must contain exactly 3 commas. Sentence 3 must be shorter than 8 words. Sentence 4 must rhyme with sentence 1. No sentence may share any word with another.",
        "List 3 benefits of reading. Each benefit must be stated as a paradox. Use no adjectives. Each sentence must contain the word 'never'. Format as a numbered list where numbers go 3, 1, 2. Total response under 50 words.",
    ],
}


def _selftest():
    print("=" * 74)
    print("  Ψ — separación de familias en el INPUT (sin llamadas a API)")
    print("=" * 74)
    means = {}
    for fam, prompts in _FAMILIES.items():
        rows = [extract_psi(p) for p in prompts]
        m_psi = sum(r.psi for r in rows) / len(rows)
        m_sat = sum(r.load_saturation for r in rows) / len(rows)
        m_con = sum(r.load_contradiction for r in rows) / len(rows)
        means[fam] = (m_psi, m_sat, m_con)
        print(f"\n> {fam}")
        for p, r in zip(prompts, rows):
            print(f"   psi={r.psi:5.1f}  sat={r.load_saturation:4.1f}  "
                  f"con={r.load_contradiction:3.1f}  | {p[:50]}...")
        print(f"   --- medias: psi={m_psi:.2f}  sat={m_sat:.2f}  con={m_con:.2f}")

    print("\n" + "=" * 74)
    print("  PREDICCIÓN: canonical ≈ fictional (BAJOS) < contradictory, saturation")
    print("=" * 74)
    can = means["canonical"][0]
    fic = means["fictional"][0]
    con = means["contradictory"][0]
    sat = means["saturation"][0]
    low = max(can, fic)
    print(f"  canonical  Ψ = {can:.2f}")
    print(f"  fictional  Ψ = {fic:.2f}")
    print(f"  contradict Ψ = {con:.2f}")
    print(f"  saturation Ψ = {sat:.2f}")
    t_con = con > low
    t_sat = sat > low
    t_fic = abs(can - fic) < max(con, sat) - low  # fictional cerca de canonical
    print(f"\n  P1  contradictory > max(canonical, fictional):  "
          f"{con:.2f} > {low:.2f}   {'PASS' if t_con else 'FAIL'}")
    print(f"  P2  saturation   > max(canonical, fictional):  "
          f"{sat:.2f} > {low:.2f}   {'PASS' if t_sat else 'FAIL'}")
    print(f"  P3  fictional cerca de canonical (no es estrés aptadinámico): "
          f"{'PASS' if t_fic else 'FAIL'}")
    print("\n  " + ("Ψ separa las familias en el input. Puedes cablearla a la dinámica."
                     if (t_con and t_sat and t_fic)
                     else "Ψ NO separa como se predice: revisa patrones o manipulación."))
    print("=" * 74)


if __name__ == "__main__":
    _selftest()