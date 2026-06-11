from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4


@dataclass
class DeepSeekConfig:
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.2
    max_tokens: int = 512
    top_logprobs: int = 5
    api_key_env: str = "DEEPSEEK_API_KEY"


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _valid_logprob(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric == -9999.0:
        return None
    return numeric


def _softmax_entropy_norm(logprobs: Sequence[float]) -> float:
    values = [_valid_logprob(value) for value in logprobs]
    clean = [value for value in values if value is not None]
    if len(clean) < 2:
        return 0.0
    max_lp = max(clean)
    weights = [math.exp(value - max_lp) for value in clean]
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    probs = [weight / total for weight in weights]
    entropy = -sum(prob * math.log(prob + 1e-15) for prob in probs)
    return max(0.0, min(1.0, entropy / math.log(len(probs))))


def _candidate_logprobs(item: Any) -> List[float]:
    chosen = _valid_logprob(_get_attr(item, "logprob"))
    top_items = _get_attr(item, "top_logprobs", []) or []
    values: List[float] = []
    for candidate in top_items:
        value = _valid_logprob(_get_attr(candidate, "logprob"))
        if value is not None:
            values.append(value)
    if chosen is not None and chosen not in values:
        values.insert(0, chosen)
    return values


def _gap_from_logprobs(logprobs: Sequence[float]) -> float:
    clean = sorted([value for value in (_valid_logprob(v) for v in logprobs) if value is not None], reverse=True)
    if len(clean) < 2:
        return 0.0
    return float(clean[0] - clean[1])


def make_deepseek_client(config: Optional[DeepSeekConfig] = None):
    cfg = config or DeepSeekConfig()
    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"{cfg.api_key_env} is not defined.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK to use DeepSeek: python -m pip install openai") from exc
    return OpenAI(api_key=api_key, base_url=cfg.base_url)


def deepseek_chat_completion(messages: Sequence[Dict[str, str]], config: Optional[DeepSeekConfig] = None):
    cfg = config or DeepSeekConfig()
    client = make_deepseek_client(cfg)
    return client.chat.completions.create(
        model=cfg.model,
        messages=list(messages),
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        logprobs=True,
        top_logprobs=cfg.top_logprobs,
    )


def deepseek_response_to_raw_turn(response: Any, turn_index: int, user_message: str) -> Dict[str, Any]:
    choices = _get_attr(response, "choices", []) or []
    if not choices:
        raise ValueError("DeepSeek response has no choices.")
    choice = choices[0]
    message = _get_attr(choice, "message")
    assistant_message = _get_attr(message, "content", "") or ""
    logprobs = _get_attr(choice, "logprobs")
    if logprobs is None:
        raise ValueError("DeepSeek response is missing choice.logprobs.")
    content = _get_attr(logprobs, "content", []) or []
    if not content:
        raise ValueError("DeepSeek response choice.logprobs.content is empty.")

    tokens: List[Dict[str, Any]] = []
    for item in content:
        top_logprobs = _candidate_logprobs(item)
        top1 = _valid_logprob(_get_attr(item, "logprob"))
        if top1 is None:
            continue
        tokens.append(
            {
                "token": _get_attr(item, "token", ""),
                "top1_logprob": top1,
                "top_logprobs": top_logprobs,
                "gap": _gap_from_logprobs(top_logprobs),
                "entropy": _softmax_entropy_norm(top_logprobs),
            }
        )

    if not tokens:
        raise ValueError("DeepSeek response has no valid token logprobs after filtering.")

    return {
        "turn_index": int(turn_index),
        "user_message": user_message,
        "assistant_message": assistant_message,
        "finish_reason": _get_attr(choice, "finish_reason"),
        "token_count": len(tokens),
        "tokens": tokens,
    }


def deepseek_response_to_tokens(response: Any) -> List[Dict[str, Any]]:
    return deepseek_response_to_raw_turn(response, turn_index=0, user_message="")["tokens"]


def deepseek_response_to_turn(response: Any, turn_index: int, user_message: str) -> Dict[str, Any]:
    return deepseek_response_to_raw_turn(response, turn_index=turn_index, user_message=user_message)


def run_deepseek_session(
    user_messages: Sequence[str],
    config: Optional[DeepSeekConfig] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = config or DeepSeekConfig()
    sid = session_id or uuid4().hex
    messages: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []
    resolved_model = cfg.model

    for turn_index, user_message in enumerate(user_messages):
        messages.append({"role": "user", "content": user_message})
        response = deepseek_chat_completion(messages, cfg)
        resolved_model = _get_attr(response, "model", resolved_model) or resolved_model
        turn = deepseek_response_to_raw_turn(response, turn_index=turn_index, user_message=user_message)
        turns.append(turn)
        messages.append({"role": "assistant", "content": turn["assistant_message"]})

    return {
        "session_id": sid,
        "model": resolved_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "deepseek",
        "turns": turns,
    }
