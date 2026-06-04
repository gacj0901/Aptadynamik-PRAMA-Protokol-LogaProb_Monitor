import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from aptadynamik.prama_components import measure
from aptadynamik.observer.report_writer import ReportWriter
from aptadynamik.observer.session_recorder import SessionRecorder

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_LOGPROBS = int(os.getenv("PRAMA_MONITOR_TOP_LOGPROBS", "5"))
WINDOW_SIZE = int(os.getenv("PRAMA_MONITOR_WINDOW_SIZE", "16"))
MAX_TOKENS = int(os.getenv("PRAMA_MONITOR_MAX_TOKENS", "512"))
RESULTS_DIR = Path("results")

app = FastAPI(title="PRAMA Monitor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, SessionRecorder] = {}


class ChatRequest(BaseModel):
    session_id: str
    user_message: str


class SessionRequest(BaseModel):
    session_id: str


def entropy_from_logprobs(logprobs: List[float]) -> float:
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-15) for p in normalized)


def extract_openai_tokens(logprobs_content) -> List[Dict]:
    tokens = []
    for item in logprobs_content or []:
        top_items = getattr(item, "top_logprobs", None) or []
        top_lps = [float(t.logprob) for t in top_items if hasattr(t, "logprob")]
        sorted_lps = sorted(top_lps, reverse=True)
        gap = sorted_lps[0] - sorted_lps[1] if len(sorted_lps) >= 2 else 5.0
        tokens.append(
            {
                "token": getattr(item, "token", ""),
                "top1_logprob": float(getattr(item, "logprob", 0.0)),
                "top_logprobs": top_lps,
                "gap": abs(float(gap)),
                "entropy": entropy_from_logprobs(top_lps),
            }
        )
    return tokens


def synthetic_tokens(text: str) -> List[Dict]:
    words = text.split()
    tokens = []
    for idx, word in enumerate(words):
        entropy = 0.45 + 0.1 * math.sin(idx / 4.0)
        gap = 1.4 + 0.2 * math.cos(idx / 5.0)
        top1 = -0.2 - 0.05 * math.sin(idx / 3.0)
        tokens.append(
            {
                "token": word,
                "top1_logprob": top1,
                "top_logprobs": [top1, top1 - gap],
                "gap": abs(gap),
                "entropy": max(0.0, entropy),
            }
        )
    return tokens


def compute_windows(tokens: List[Dict], window_size: int = WINDOW_SIZE) -> List[Dict]:
    windows = []
    max_entropy = math.log2(TOP_LOGPROBS)
    max_gap = 5.0
    for start in range(0, len(tokens), window_size):
        chunk = tokens[start : start + window_size]
        if not chunk:
            continue
        entropies = [float(t.get("entropy", 0.0)) for t in chunk]
        gaps = [float(t.get("gap", 0.0)) for t in chunk]
        entropy_raw = sum(entropies) / len(entropies)
        entropy_norm = min(1.0, max(0.0, entropy_raw / max(max_entropy, 1e-12)))
        gap_raw = sum(gaps) / len(gaps)
        gap_norm = min(1.0, max(0.0, gap_raw / max_gap))
        entropy_mean = entropy_raw
        entropy_std = math.sqrt(sum((v - entropy_mean) ** 2 for v in entropies) / len(entropies))
        entropy_range = max(entropies) - min(entropies)
        windows.append(
            {
                "window_index": len(windows),
                "entropy_raw": round(entropy_raw, 6),
                "entropy_norm": round(entropy_norm, 6),
                "gap_norm": round(gap_norm, 6),
                "rigidity": round(gap_norm * (1.0 - entropy_norm), 6),
                "uncertainty": round(entropy_norm * (1.0 - gap_norm), 6),
                "entropy_std": round(entropy_std, 6),
                "entropy_range": round(entropy_range, 6),
                "n_tokens_in_window": len(chunk),
            }
        )
    return windows


def compute_live_prama_events(tokens: List[Dict], turn_index: int, window_size: int = WINDOW_SIZE) -> List[Dict]:
    token_chunks = []
    for start in range(0, len(tokens), window_size):
        chunk = tokens[start : start + window_size]
        if not chunk:
            continue
        token_chunks.append(
            {
                "turn_index": len(token_chunks),
                "tokens": chunk,
            }
        )
    if not token_chunks:
        return []
    calib_window = min(3, len(token_chunks))
    result = measure(token_chunks, calib_window=calib_window)
    events = []

    def rounded(value, digits: int = 6):
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return round(numeric, digits)

    def bool_value(value, default: bool = False) -> bool:
        if value is None:
            return default
        return bool(value)

    for row in result["turns"]:
        micro_health = rounded(row.get("micro_health"))
        macro_health = rounded(row.get("macro_health"))
        acople = rounded(row.get("acople"))
        micro_drop = rounded(row.get("micro_drop"))
        micro_raw = rounded(row.get("micro_raw"))
        micro_excess = rounded(row.get("micro_excess"))
        viability_margin = rounded(row.get("viability_margin"))
        boundary_pressure = rounded(row.get("boundary_pressure"))
        activity_raw = rounded(row.get("activity_raw"))
        activity_structural = rounded(row.get("activity_structural"))
        activity_effective = rounded(row.get("activity_effective"))
        acople_effective = rounded(row.get("acople_effective"))
        delta_instant = rounded(row.get("delta_instant"))
        xi_norm = rounded(row.get("xi_norm"))
        lambda_remaining = rounded(row.get("lambda_remaining"))
        theta_dynamic = rounded(row.get("theta_dynamic"))
        events.append(
            {
                "type": "prama",
                "turn_index": turn_index,
                "window_index": row.get("turn_index"),
                "micro_raw": micro_raw,
                "micro_health": micro_health,
                "macro_health": macro_health,
                "activity_raw": activity_raw,
                "activity_structural": activity_structural,
                "activity_effective": activity_effective,
                "acople": acople,
                "acople_effective": acople_effective,
                "delta_instant": delta_instant,
                "xi_norm": xi_norm,
                "lambda_remaining": lambda_remaining,
                "theta_dynamic": theta_dynamic,
                "viability_margin": viability_margin,
                "threshold_crossed": bool_value(row.get("threshold_crossed")),
                "xi_exceeds_theta": bool_value(row.get("xi_exceeds_theta")),
                "boundary_side": row.get("boundary_side", "UNRESOLVED"),
                "boundary_pressure": boundary_pressure,
                "viability_status": row.get("viability_status", "UNRESOLVED"),
                "distance_to_threshold": rounded(row.get("distance_to_threshold")),
                "micro_excess": micro_excess,
                # Legacy visual aliases. These names keep their existing frontend meaning.
                "micro": micro_health,
                "macro": macro_health,
                "viability": viability_margin,
                # Legacy visual fallback; PRAMA Components v0.2.2 core does not compute these observer metrics.
                "rig": 0.0,
                "eco": 0.0,
                "alu": 0.0,
                "turb": 0.0,
                "collapse_threshold": result.get("collapse_threshold"),
                "regime_label": result.get("regime_label"),
                "regime_description": result.get("regime_description"),
                "recovery_observed": result.get("recovery_observed"),
                "first_crossing_turn": result.get("first_crossing_turn"),
                "threshold_crossing_ratio": result.get("threshold_crossing_ratio"),
                "persistent_crossing_ratio": result.get("persistent_crossing_ratio"),
                "post_crossing_recovery_turns": result.get("post_crossing_recovery_turns", []),
                "trajectory_assessment": result.get("trajectory_assessment"),
                "live_mode": "token_window",
            }
        )
    final = events[-1].copy()
    final["type"] = "final_prama"
    return events + [final]


def call_openai(recorder: SessionRecorder, user_message: str):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. The conversation is monitored locally by PRAMA metrics.",
        }
    ]
    messages.extend(recorder.messages_for_model())
    messages.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=recorder.model,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
    )
    choice = response.choices[0]
    assistant_message = choice.message.content or ""
    logprobs_content = choice.logprobs.content if choice.logprobs else []
    return assistant_message, extract_openai_tokens(logprobs_content), choice.finish_reason


def call_dry(recorder: SessionRecorder, user_message: str):
    assistant_message = (
        "Local dry-run response. I received your message and generated a deterministic "
        "stand-in answer so PRAMA Monitor can record geometry metrics without an API key. "
        f"Your message was: {user_message}"
    )
    return assistant_message, synthetic_tokens(assistant_message), "stop"


def resolve_session(session_id: str) -> SessionRecorder:
    recorder = SESSIONS.get(session_id)
    if recorder is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return recorder


@app.post("/session/start")
def start_session():
    recorder = SessionRecorder.create(model=MODEL)
    SESSIONS[recorder.session_id] = recorder
    return recorder.live_summary()


@app.post("/chat")
def chat(request: ChatRequest):
    recorder = resolve_session(request.session_id)
    if recorder.status != "active":
        raise HTTPException(status_code=400, detail="Session is closed")

    if os.environ.get("OPENAI_API_KEY"):
        assistant_message, tokens, finish_reason = call_openai(recorder, request.user_message)
    else:
        assistant_message, tokens, finish_reason = call_dry(recorder, request.user_message)
    windows = compute_windows(tokens)
    turn = recorder.append_turn(
        request.user_message,
        assistant_message,
        tokens,
        windows,
        finish_reason=finish_reason,
    )
    summary = recorder.live_summary()
    prama_events = compute_live_prama_events(tokens, turn.get("turn_index", max(summary.get("turn_count", 1) - 1, 0)))
    final_prama = prama_events[-1] if prama_events else {}
    prama_session_state = {
        "regime_label": final_prama.get("regime_label"),
        "regime_description": final_prama.get("regime_description"),
        "recovery_observed": final_prama.get("recovery_observed"),
        "first_crossing_turn": final_prama.get("first_crossing_turn"),
        "threshold_crossing_ratio": final_prama.get("threshold_crossing_ratio"),
        "persistent_crossing_ratio": final_prama.get("persistent_crossing_ratio"),
        "post_crossing_recovery_turns": final_prama.get("post_crossing_recovery_turns", []),
        "trajectory_assessment": final_prama.get("trajectory_assessment"),
    }
    summary_with_prama = {**summary, **prama_session_state}

    def stream():
        step = 48
        emitted_prama = 0
        total_chunks = max(1, math.ceil(len(assistant_message) / step))
        for i in range(0, len(assistant_message), step):
            yield json.dumps({"type": "chunk", "text": assistant_message[i : i + step]}) + "\n"
            target_events = math.floor(((i // step) + 1) * max(len(prama_events) - 1, 0) / total_chunks)
            while emitted_prama < target_events:
                yield json.dumps(prama_events[emitted_prama]) + "\n"
                emitted_prama += 1
        if prama_events:
            while emitted_prama < len(prama_events):
                yield json.dumps(prama_events[emitted_prama]) + "\n"
                emitted_prama += 1
        else:
            yield json.dumps(
                {
                    "type": "final_prama",
                    "turn_index": turn.get("turn_index", max(summary.get("turn_count", 1) - 1, 0)),
                    "viability": None,
                    "threshold_crossed": False,
                    "boundary_side": "UNRESOLVED",
                    "boundary_pressure": None,
                    "viability_status": "UNRESOLVED",
                    "distance_to_threshold": None,
                    "micro_raw": None,
                    "micro_health": None,
                    "activity_raw": None,
                    "activity_structural": None,
                    "activity_effective": None,
                    "acople_effective": None,
                    "delta_instant": None,
                    "xi_norm": None,
                    "lambda_remaining": None,
                    "theta_dynamic": None,
                    "viability_margin": None,
                    "xi_exceeds_theta": False,
                    "micro": None,
                    "macro": None,
                    "macro_health": None,
                    "acople": None,
                    "micro_drop": None,
                    # Legacy visual fallback; PRAMA Components v0.2.2 core does not compute these observer metrics.
                    "rig": 0.0,
                    "eco": 0.0,
                    "alu": 0.0,
                    "turb": 0.0,
                    "regime_label": None,
                    "regime_description": None,
                    "recovery_observed": None,
                    "first_crossing_turn": None,
                    "threshold_crossing_ratio": None,
                    "persistent_crossing_ratio": None,
                    "post_crossing_recovery_turns": [],
                    "trajectory_assessment": None,
                    "live_mode": "turn_final_only",
                }
            ) + "\n"
        yield json.dumps({"type": "turn_summary", "turn": turn["summary"], "session": summary_with_prama}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/session/stop")
def stop_session(request: SessionRequest):
    recorder = resolve_session(request.session_id)
    recorder.stop()
    return recorder.live_summary()


@app.post("/session/report")
def session_report(request: SessionRequest):
    recorder = resolve_session(request.session_id)
    result = ReportWriter(
        RESULTS_DIR,
        temperature=0.7,
        max_tokens=MAX_TOKENS,
        top_logprobs=TOP_LOGPROBS,
        window_size=WINDOW_SIZE,
    ).write(recorder)
    return result


@app.get("/session/{session_id}/summary")
def session_summary(session_id: str):
    return resolve_session(session_id).live_summary()


@app.get("/download")
def download(path: str = Query(...)):
    requested = Path(path)
    if requested.is_absolute():
        candidate = requested.resolve()
    else:
        candidate = (Path.cwd() / requested).resolve()
    results_root = (Path.cwd() / RESULTS_DIR).resolve()
    try:
        candidate.relative_to(results_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate)
