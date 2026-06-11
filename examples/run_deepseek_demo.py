from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aptadynamik.pipelines.deepseek import DeepSeekConfig, run_deepseek_session


PROMPTS = [
    "Hola. Responde en una frase breve.",
    "Explica en dos frases que es la coherencia estructural.",
    "Como sabes que tu respuesta mantiene coherencia?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small DeepSeek logprob smoke session.")
    parser.add_argument("--output-dir", default="results/deepseek_smoke")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-logprobs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = DeepSeekConfig(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_logprobs=args.top_logprobs,
    )
    raw = run_deepseek_session(PROMPTS, config=config)
    raw_path = output_dir / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    prama_dir = output_dir / "prama"
    print(f"Wrote {raw_path}")
    print("Process with:")
    print(f"python scripts\\prama_components_runner.py --from-raw {raw_path} --output-dir {prama_dir} --calib-window 1")


if __name__ == "__main__":
    main()
