"""
psi_xi_partial.py — El análisis decisivo
=========================================
Pregunta única: ¿la tensión interna acumulada (xi/w) sigue a la presión
ambiental (Ψ) UNA VEZ que se controla la longitud de la respuesta (tokens)?

Esa es la afirmación central del marco: Ξ ∝ Ψ. Si la correlación parcial
r(Ψ, xi | tokens) es ~0, el vínculo no está en estos datos —y la separación
de familias que se ve en bruto es longitud/formato, no presión.

Sin dependencias externas. Los 100 registros (5 corridas × 20 prompts) están
embebidos tal como aparecen en la consola. Para precisión completa y más
corridas, sustituye DATA por la lectura de tus v2_summary_*.csv (función
load_csv al final).
"""

import math

# family, tokens, psi, rig, unc, mar, estd(entropy_std), xiw, integ, lam
DATA = [
 ("can",103,0.0,0.909,0.000,0.591,0.257,0.133,0.930,0.309),
 ("can",128,1.0,0.913,0.003,0.574,0.265,0.120,0.799,0.313),
 ("can",128,0.0,0.832,0.009,0.634,0.412,0.140,0.848,0.202),
 ("can",128,0.0,0.931,0.000,0.569,0.211,0.124,0.912,0.243),
 ("can",45,1.0,0.988,0.000,0.512,0.059,0.048,0.895,0.899),
 ("fic",49,0.0,0.701,0.046,0.735,0.462,0.083,0.849,0.778),
 ("fic",39,0.0,0.635,0.043,0.779,0.656,0.087,0.911,0.858),
 ("fic",103,0.0,0.566,0.073,0.787,0.531,0.186,0.851,0.017),
 ("fic",101,0.0,0.677,0.033,0.776,0.633,0.186,0.810,0.143),
 ("fic",42,0.0,0.707,0.033,0.721,0.531,0.087,0.932,0.800),
 ("con",128,5.0,0.735,0.018,0.726,0.564,0.166,0.978,0.096),
 ("con",47,1.0,0.810,0.000,0.690,0.549,0.067,0.932,0.856),
 ("con",128,3.0,0.769,0.021,0.682,0.460,0.155,0.983,0.085),
 ("con",128,4.0,0.383,0.195,0.858,0.640,0.219,0.929,-0.594),
 ("con",30,7.0,0.430,0.158,0.895,0.733,0.111,0.967,0.855),
 ("sat",27,12.0,0.393,0.166,0.910,0.694,0.128,0.953,0.818),
 ("sat",53,9.0,0.154,0.418,0.864,0.569,0.160,0.851,0.589),
 ("sat",21,6.0,0.449,0.103,0.911,0.719,0.125,0.976,0.890),
 ("sat",41,9.0,0.527,0.185,0.755,0.464,0.132,0.891,0.693),
 ("sat",48,12.0,0.648,0.023,0.825,0.767,0.102,0.943,0.774),
 # run 2
 ("can",103,0.0,0.908,0.002,0.585,0.248,0.130,0.876,0.318),
 ("can",128,1.0,0.926,0.003,0.568,0.231,0.111,0.892,0.343),
 ("can",128,0.0,0.795,0.027,0.659,0.422,0.137,0.856,0.134),
 ("can",128,0.0,0.924,0.004,0.568,0.218,0.121,0.851,0.281),
 ("can",45,1.0,0.989,0.000,0.511,0.059,0.047,0.894,0.900),
 ("fic",50,0.0,0.671,0.052,0.703,0.472,0.097,0.848,0.758),
 ("fic",56,0.0,0.562,0.105,0.789,0.542,0.152,0.814,0.625),
 ("fic",128,0.0,0.636,0.055,0.756,0.554,0.176,0.935,-0.169),
 ("fic",107,0.0,0.615,0.048,0.792,0.676,0.188,0.925,-0.019),
 ("fic",48,0.0,0.674,0.035,0.759,0.620,0.101,0.926,0.759),
 ("con",128,5.0,0.509,0.093,0.841,0.672,0.209,0.984,-0.855),
 ("con",50,1.0,0.747,0.029,0.681,0.450,0.108,0.808,0.723),
 ("con",128,3.0,0.781,0.024,0.661,0.381,0.151,0.980,0.063),
 ("con",128,4.0,0.400,0.172,0.884,0.672,0.227,0.980,-0.691),
 ("con",24,7.0,0.384,0.147,0.932,0.674,0.109,0.982,0.915),
 ("sat",34,12.0,0.279,0.221,0.908,0.771,0.116,0.953,0.816),
 ("sat",53,9.0,0.140,0.417,0.888,0.589,0.155,0.853,0.591),
 ("sat",20,6.0,0.419,0.109,0.926,0.728,0.097,0.981,0.934),
 ("sat",37,9.0,0.468,0.180,0.819,0.522,0.105,0.930,0.829),
 ("sat",33,12.0,0.591,0.068,0.806,0.599,0.087,0.959,0.866),
 # run 3
 ("can",106,0.0,0.913,0.003,0.578,0.228,0.133,0.892,0.271),
 ("can",128,1.0,0.917,0.001,0.580,0.293,0.117,0.864,0.317),
 ("can",128,0.0,0.838,0.013,0.632,0.378,0.139,0.939,0.162),
 ("can",128,0.0,0.919,0.006,0.567,0.205,0.122,0.921,0.282),
 ("can",45,1.0,0.989,0.000,0.511,0.059,0.047,0.893,0.900),
 ("fic",58,0.0,0.684,0.036,0.723,0.517,0.108,0.814,0.649),
 ("fic",43,0.0,0.627,0.095,0.730,0.474,0.060,0.951,0.883),
 ("fic",83,0.0,0.543,0.111,0.787,0.604,0.167,0.796,0.314),
 ("fic",104,0.0,0.662,0.034,0.773,0.616,0.191,0.915,0.102),
 ("fic",57,0.0,0.712,0.043,0.699,0.437,0.132,0.895,0.550),
 ("con",128,5.0,0.602,0.075,0.797,0.589,0.202,0.953,-0.545),
 ("con",36,1.0,0.859,0.000,0.641,0.360,0.086,0.913,0.848),
 ("con",128,3.0,0.803,0.014,0.663,0.417,0.155,0.920,0.066),
 ("con",128,4.0,0.349,0.197,0.905,0.706,0.229,0.956,-1.466),
 ("con",24,7.0,0.543,0.050,0.864,0.807,0.087,0.984,0.942),
 ("sat",31,12.0,0.213,0.297,0.956,0.664,0.115,0.971,0.859),
 ("sat",55,9.0,0.085,0.503,0.826,0.564,0.140,0.857,0.661),
 ("sat",46,6.0,0.720,0.041,0.732,0.565,0.123,0.903,0.704),
 ("sat",42,9.0,0.454,0.160,0.857,0.577,0.108,0.904,0.764),
 ("sat",40,12.0,0.627,0.037,0.803,0.647,0.093,0.945,0.843),
 # run 4
 ("can",104,0.0,0.898,0.000,0.602,0.281,0.136,0.901,0.307),
 ("can",128,1.0,0.922,0.003,0.569,0.251,0.115,0.802,0.314),
 ("can",102,0.0,0.757,0.023,0.679,0.473,0.135,0.818,0.326),
 ("can",128,0.0,0.930,0.000,0.570,0.215,0.121,0.939,0.284),
 ("can",45,1.0,0.988,0.000,0.512,0.059,0.047,0.895,0.899),
 ("fic",39,0.0,0.678,0.050,0.725,0.568,0.054,0.955,0.920),
 ("fic",66,0.0,0.637,0.066,0.762,0.522,0.173,0.847,0.326),
 ("fic",76,0.0,0.527,0.102,0.829,0.567,0.176,0.692,0.345),
 ("fic",109,0.0,0.603,0.061,0.799,0.608,0.182,0.896,0.045),
 ("fic",57,0.0,0.471,0.182,0.792,0.540,0.125,0.748,0.596),
 ("con",128,5.0,0.698,0.025,0.765,0.624,0.186,0.998,-0.119),
 ("con",43,1.0,0.733,0.060,0.699,0.493,0.049,0.937,0.909),
 ("con",128,3.0,0.815,0.005,0.670,0.487,0.153,0.881,-0.045),
 ("con",128,4.0,0.428,0.163,0.864,0.661,0.218,0.964,-0.321),
 ("con",29,7.0,0.549,0.073,0.847,0.621,0.096,0.974,0.884),
 ("sat",33,12.0,0.371,0.254,0.829,0.514,0.137,0.921,0.756),
 ("sat",50,9.0,0.138,0.413,0.878,0.629,0.155,0.852,0.607),
 ("sat",21,6.0,0.482,0.085,0.906,0.737,0.123,0.977,0.890),
 ("sat",33,9.0,0.575,0.063,0.794,0.604,0.125,0.926,0.781),
 ("sat",42,12.0,0.655,0.013,0.795,0.740,0.130,0.883,0.724),
 # run 5
 ("can",96,0.0,0.903,0.000,0.597,0.253,0.132,0.841,0.353),
 ("can",128,1.0,0.929,0.001,0.565,0.241,0.114,0.751,0.323),
 ("can",128,0.0,0.870,0.001,0.626,0.425,0.140,0.896,0.164),
 ("can",128,0.0,0.911,0.008,0.573,0.233,0.121,0.921,0.293),
 ("can",45,1.0,0.989,0.000,0.511,0.059,0.047,0.893,0.900),
 ("fic",43,0.0,0.724,0.034,0.712,0.509,0.088,0.900,0.793),
 ("fic",52,0.0,0.710,0.027,0.738,0.544,0.123,0.847,0.678),
 ("fic",85,0.0,0.541,0.072,0.820,0.589,0.192,0.962,0.085),
 ("fic",103,0.0,0.573,0.079,0.816,0.650,0.204,0.979,-0.248),
 ("fic",48,0.0,0.539,0.104,0.817,0.594,0.102,0.893,0.791),
 ("con",128,5.0,0.656,0.038,0.776,0.630,0.189,0.993,-0.094),
 ("con",38,1.0,0.809,0.005,0.677,0.445,0.072,0.934,0.889),
 ("con",128,3.0,0.790,0.025,0.651,0.413,0.141,0.883,0.154),
 ("con",128,4.0,0.434,0.161,0.880,0.677,0.222,0.966,-0.903),
 ("con",23,7.0,0.369,0.175,0.904,0.617,0.113,0.977,0.907),
 ("sat",29,12.0,0.346,0.170,0.915,0.721,0.090,0.972,0.897),
 ("sat",52,9.0,0.085,0.531,0.805,0.550,0.129,0.866,0.694),
 ("sat",20,6.0,0.493,0.093,0.891,0.781,0.152,0.922,0.845),
 ("sat",32,9.0,0.408,0.155,0.852,0.641,0.063,0.976,0.941),
 ("sat",33,12.0,0.612,0.047,0.815,0.540,0.089,0.954,0.862),
]

COLS = ["family","tokens","psi","rig","unc","mar","estd","xiw","integ","lam"]
def col(rows, name): return [r[COLS.index(name)] for r in rows]

def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sxy = sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx == 0 or syy == 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def partial(x, y, z):
    """r(x,y | z) — primer orden."""
    rxy = pearson(x,y); rxz = pearson(x,z); ryz = pearson(y,z)
    den = math.sqrt((1-rxz**2)*(1-ryz**2))
    return float('nan') if den == 0 else (rxy - rxz*ryz)/den

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0]*len(v); i = 0
        while i < len(v):
            j = i
            while j+1 < len(v) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    return pearson(rank(x), rank(y))

def pval(r, n):
    if abs(r) >= 1: return 0.0
    t = r*math.sqrt((n-2)/(1-r**2))
    # aproximación normal de dos colas (suficiente para n=100)
    z = abs(t)
    return 2*(1 - 0.5*(1+math.erf(z/math.sqrt(2))))

def report(name, x, y, z=None, n=None):
    n = n or len(x)
    r = pearson(x,y); rs = spearman(x,y)
    line = f"  {name:<46} r={r:+.3f}  r2={r*r:.3f}  rho={rs:+.3f}  p={pval(r,n):.3f}"
    if z is not None:
        rp = partial(x,y,z)
        line += f"  | parcial r={rp:+.3f}"
    print(line)

print("="*78)
print("  EL ANÁLISIS DECISIVO — ¿Ξ sigue a Ψ una vez quitada la longitud?")
print("="*78)
print(f"  n = {len(DATA)} generaciones (5 corridas × 20 prompts)")

psi = col(DATA,"psi"); xiw = col(DATA,"xiw"); tok = col(DATA,"tokens")
rig = col(DATA,"rig"); unc = col(DATA,"unc"); estd = col(DATA,"estd"); mar = col(DATA,"mar")

print("\n--- Cuánto se confunde la presión con la longitud ---")
report("r(Psi, tokens)", psi, tok)

print("\n--- La hipótesis central: Psi -> xi/w ---")
report("r(Psi, xi/w)            [bruto]", psi, xiw)
report("r(Psi, xi/w | tokens)   [decisivo]", psi, xiw, tok)
report("r(Psi, xi/w | rigidez)  [control confianza]", psi, xiw, rig)

print("\n--- Qué SÍ rastrea Psi (para contexto) ---")
report("r(Psi, uncertainty)", psi, unc)
report("r(Psi, uncertainty | tokens)", psi, unc, tok)
report("r(Psi, rigidity)", psi, rig)
report("r(Psi, rigidity | tokens)", psi, rig, tok)
report("r(Psi, entropy_std)", psi, estd)
report("r(Psi, entropy_std | tokens)", psi, estd, tok)

print("\n--- Dosis-respuesta DENTRO de cada familia estresada ---")
print("    (Psi varía; longitud y tipo casi constantes -> test más limpio)")
for fam, label in [("con","contradictory"),("sat","saturation")]:
    rows = [r for r in DATA if r[0]==fam]
    p = col(rows,"psi"); x = col(rows,"xiw"); t = col(rows,"tokens")
    rb = pearson(p,x); rp = partial(p,x,t)
    print(f"  {label:<14} n={len(rows)}  r(Psi,xi)={rb:+.3f}  parcial|tok={rp:+.3f}")

# medias por familia para la lectura
print("\n--- Medias por familia (recordatorio) ---")
print(f"  {'fam':<6}{'Psi':>7}{'xi/w':>9}{'tok':>7}{'rig':>8}{'unc':>8}")
for fam in ["can","fic","con","sat"]:
    rows=[r for r in DATA if r[0]==fam]
    print(f"  {fam:<6}{sum(col(rows,'psi'))/len(rows):>7.2f}"
          f"{sum(col(rows,'xiw'))/len(rows):>9.3f}"
          f"{sum(col(rows,'tokens'))/len(rows):>7.0f}"
          f"{sum(col(rows,'rig'))/len(rows):>8.3f}"
          f"{sum(col(rows,'unc'))/len(rows):>8.3f}")
print("="*78)


# --- Para tus CSV reales (precisión completa, más corridas) ---
def load_csv(paths):
    """Sustituye DATA por esto: lee tus v2_summary_*.csv y devuelve filas
    en el mismo orden de columnas que COLS. Requiere columnas: family,
    n_tokens, psi, avg_rigidity, avg_uncertainty, avg_margin,
    avg_entropy_std, xi_per_window, final_integrity, final_lambda."""
    import csv
    out = []
    for p in paths:
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    out.append((
                        row["family"][:3], float(row["n_tokens"]), float(row["psi"]),
                        float(row["avg_rigidity"]), float(row["avg_uncertainty"]),
                        float(row["avg_margin"]), float(row["avg_entropy_std"]),
                        float(row["xi_per_window"]), float(row["final_integrity"]),
                        float(row["final_lambda"]),
                    ))
                except (KeyError, ValueError):
                    continue
    return out