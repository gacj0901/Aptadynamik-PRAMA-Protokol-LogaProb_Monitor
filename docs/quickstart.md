# PRAMA Protokol - Arranque

PRAMA monitorea viabilidad estructural de trayectorias de generacion LLM a partir de senales de logprobs y entropia.

## Estructura principal

```text
src/aptadynamik/prama_core.py                  Motor PRAMA
src/aptadynamik/pipelines/gemini.py            Pipeline Gemini -> PRAMA -> results
src/aptadynamik/verification/scenarios.py      Verificacion offline sin API
docs/quickstart.md                             Este arranque
```

## Paso 1 - Verificar Python

```bash
python --version
```

Si dice Python 3.9 o superior, estas listo.

## Paso 2 - Instalar el paquete local

Desde la raiz del repositorio:

```bash
python -m pip install -e .
```

Para usar Gemini:

```bash
python -m pip install -r requirements.txt
```

## Paso 3 - Probar el motor sin internet

```bash
prama-verify
```

Tambien puedes ejecutar:

```bash
python examples/run_offline_verify.py
```

Debe dar `4/4 tests passed`. La salida completa se guarda en `results/results.json`.

## Paso 4 - Obtener API key de Gemini

1. Ve a https://aistudio.google.com/apikey
2. Inicia sesion con tu cuenta de Google
3. Crea una API key
4. Copia la key

## Paso 5 - Ejecutar el pipeline Gemini

Mac/Linux:

```bash
export GEMINI_API_KEY="tu-key-aqui"
prama-gemini
```

Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="tu-key-aqui"
prama-gemini
```

Tambien puedes ejecutar:

```bash
python examples/run_gemini_demo.py
```

El pipeline tarda alrededor de dos minutos porque espera entre prompts para respetar limites de rate.

## Paso 6 - Leer resultados

Los archivos se crean en `results/`:

- `gemini_summary_*.csv`: una fila por prompt.
- `gemini_detail_*.csv`: ventanas de trayectoria por prompt.
- `gemini_results_*.json`: datos completos.

Observa `final_integrity`, `final_xi`, `final_lambda`, `final_regime` y la serie por ventanas como senales de viabilidad estructural de la trayectoria generada.

---

*AptadynamiK - PRAMA Protokol - G.A.C.J. (c) 2026*
