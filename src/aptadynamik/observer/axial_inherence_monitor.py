from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EPSILON = 1e-12


def cos_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("cos_similarity requires non-empty vectors of equal length")
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) ** 2 for x in a))
    nb = math.sqrt(sum(float(y) ** 2 for y in b))
    if na <= EPSILON or nb <= EPSILON:
        return 0.0
    return dot / (na * nb)


def compute_iota(cur_vec: Sequence[float], hist_vecs: Sequence[Sequence[float]]) -> float:
    if not hist_vecs:
        return 1.0
    max_sim = max(cos_similarity(cur_vec, hist_vec) for hist_vec in hist_vecs)
    return max(0.0, min(1.0, 1.0 - max_sim))


def dominances(iota: float, contradiction_rate: float, drop_fraction: float) -> Tuple[float, float, float]:
    i = max(0.0, min(1.0, float(iota)))
    k = max(0.0, min(1.0, 1.0 - float(contradiction_rate)))
    r = max(0.0, min(1.0, float(drop_fraction)))
    return i, k, r


def dominant_axis(ikr: Sequence[float]) -> str:
    if len(ikr) != 3:
        raise ValueError("dominant_axis requires an (i, k, r) triple")
    axes = ("i", "k", "r")
    return axes[max(range(3), key=lambda idx: float(ikr[idx]))]


def _axis_sequence(seq: Sequence[Any]) -> List[str]:
    axes: List[str] = []
    for item in seq:
        if isinstance(item, str):
            if item not in {"i", "k", "r"}:
                raise ValueError(f"unknown axis '{item}'")
            axes.append(item)
        else:
            axes.append(dominant_axis(item))
    return axes


def rotational_mobility(seq: Sequence[Any], win: int = 9, warmup: int = 9) -> List[float]:
    axes = _axis_sequence(seq)
    mobility: List[float] = []
    for idx in range(len(axes)):
        if idx + 1 < warmup:
            mobility.append(1.0)
            continue
        window = axes[max(0, idx - win + 1) : idx + 1]
        if len(window) < 2:
            mobility.append(1.0)
            continue
        diversity = len(set(window)) / 3.0
        transitions = sum(1 for left, right in zip(window, window[1:]) if left != right) / (len(window) - 1)
        mobility.append(max(0.01, min(1.0, 0.3 * diversity + 0.7 * transitions)))
    return mobility


def fatigue_series(ikr_seq: Sequence[Sequence[float]], win: int = 9, warmup: int = 9) -> List[float]:
    mobility = rotational_mobility(ikr_seq, win=win, warmup=warmup)
    return [max(0.0, -math.log(max(value, 0.01)) / math.log(100.0)) for value in mobility]


def precedence_lead(
    ikr_seq: Sequence[Sequence[float]],
    fn_seq: Sequence[float],
    fat_thr: float = 0.4,
    warmup: int = 9,
) -> Optional[int]:
    fatigue = fatigue_series(ikr_seq, warmup=warmup)
    fatigue_turn = next((idx for idx, value in enumerate(fatigue) if idx + 1 >= warmup and value >= fat_thr), None)
    if fatigue_turn is None:
        return None
    if not fn_seq:
        return None
    baseline = mean([float(value) for value in fn_seq[: max(1, min(warmup, len(fn_seq)))]] )
    drop_threshold = min(0.5, baseline * 0.8)
    function_turn = next((idx for idx, value in enumerate(fn_seq) if float(value) <= drop_threshold), None)
    if function_turn is None:
        return None
    return function_turn - fatigue_turn


def bootstrap_ci(values: Sequence[float], n: int = 2000) -> Dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = random.Random(0)
    samples = []
    for _ in range(n):
        draw = [clean[rng.randrange(len(clean))] for _ in clean]
        samples.append(mean(draw))
    samples.sort()
    lo_idx = int(0.025 * (len(samples) - 1))
    hi_idx = int(0.975 * (len(samples) - 1))
    return {"mean": mean(clean), "lo": samples[lo_idx], "hi": samples[hi_idx], "n": len(clean)}


def _mean_logprob(token_logprobs: Sequence[float]) -> Optional[float]:
    values = [float(value) for value in token_logprobs]
    if not values:
        return None
    return mean(values)


def run_axial_inherence_session(
    adapters: Any,
    task: Dict[str, Any],
    condition: str = "sustained",
    max_turns: int = 40,
) -> Dict[str, Any]:
    context_messages = [{"role": "system", "content": task.get("system", "Answer the task directly.")}]
    if task.get("prompt"):
        context_messages.append({"role": "user", "content": task["prompt"]})
    kb_state: Dict[str, Any] = {}
    hist_vecs: List[List[float]] = []
    turns: List[Dict[str, Any]] = []
    ikr_seq: List[Tuple[float, float, float]] = []
    fn_seq: List[float] = []

    for turn_index in range(max_turns):
        user_message = adapters.interlocutor_turn(turns, condition)
        context_messages.append({"role": "user", "content": user_message})
        generated = adapters.generate_turn(context_messages)
        text = str(generated.get("text", ""))
        cur_vec = adapters.embed(text)
        iota = compute_iota(cur_vec, hist_vecs)
        contradiction_rate, kb_state = adapters.kb_consistency(text, kb_state)
        drop_fraction, kb_state = adapters.kb_retraction(text, kb_state)
        i, k, r = dominances(iota, contradiction_rate, drop_fraction)
        f = float(adapters.exo_judge(text, {"task": task, "condition": condition, "turn": turn_index}))
        ikr_seq.append((i, k, r))
        fn_seq.append(f)
        token_logprobs = generated.get("token_logprobs") or []
        turns.append(
            {
                "turn": turn_index,
                "i": i,
                "k": k,
                "r": r,
                "f": f,
                "dominant": dominant_axis((i, k, r)),
                "mean_logprob": _mean_logprob(token_logprobs),
                "n_tokens": len(token_logprobs),
                "finish_reason": generated.get("finish_reason"),
            }
        )
        hist_vecs.append(list(cur_vec))
        context_messages.append({"role": "assistant", "content": text})

    fatigue = fatigue_series(ikr_seq)
    for turn, fatigue_value in zip(turns, fatigue):
        turn["fatigue"] = fatigue_value
    return {
        "task_id": task.get("task_id", task.get("id", "task")),
        "condition": condition,
        "turns": turns,
        "fatigue": fatigue,
        "precedence_lead": precedence_lead(ikr_seq, fn_seq),
    }


def run_axial_inherence_study(
    adapters: Any,
    tasks: Sequence[Dict[str, Any]],
    reps: int = 30,
    condition: str = "sustained",
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = []
    for rep in range(reps):
        for task in tasks:
            session = run_axial_inherence_session(adapters, task, condition=condition)
            session["rep"] = rep
            sessions.append(session)

    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for session in sessions:
                handle.write(json.dumps(session) + "\n")

    leads = [session["precedence_lead"] for session in sessions if session.get("precedence_lead") is not None]
    final_fatigues = [session["fatigue"][-1] for session in sessions if session.get("fatigue")]
    return {
        "condition": condition,
        "n_sessions": len(sessions),
        "fatigue_mean": mean(final_fatigues) if final_fatigues else None,
        "fatigue_max": max(final_fatigues) if final_fatigues else None,
        "precedence_lead_ci": bootstrap_ci(leads),
        "valid_leads": len(leads),
        "sessions": sessions,
    }
