from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def default_prama_regime_state() -> Dict[str, Any]:
    return {
        "regime_label": None,
        "regime_description": None,
        "trajectory_assessment": None,
        "recovery_observed": None,
        "first_crossing_turn": None,
        "threshold_crossing_ratio": None,
        "persistent_crossing_ratio": None,
        "post_crossing_recovery_turns": [],
    }


@dataclass
class SessionRecorder:
    session_id: str
    model: str
    created_at: str = field(default_factory=utc_now)
    closed_at: Optional[str] = None
    generated_at: Optional[str] = None
    output_prefix: Optional[str] = None
    output_dir: Optional[str] = None
    output_folder_name: Optional[str] = None
    prama_regime_state: Dict[str, Any] = field(default_factory=default_prama_regime_state)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"

    @classmethod
    def create(cls, model: str, session_id: Optional[str] = None) -> "SessionRecorder":
        return cls(session_id=session_id or uuid4().hex, model=model)

    def append_turn(
        self,
        user_message: str,
        assistant_message: str,
        tokens: List[Dict[str, Any]],
        windows: List[Dict[str, Any]],
        turn_summary: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        turn_index = len(self.turns)
        summary = turn_summary or summarize_windows(windows)
        turn = {
            "turn_index": turn_index,
            "timestamp": utc_now(),
            "user_message": user_message,
            "assistant_message": assistant_message,
            "finish_reason": finish_reason,
            "token_count": len(tokens),
            "tokens": tokens,
            "windows": windows,
            "summary": summary,
        }
        self.turns.append(turn)
        self.update_prama_regime_state(summary)
        return turn

    def update_prama_regime_state(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        current = {**default_prama_regime_state(), **self.prama_regime_state}
        changed = False
        for key in current:
            if key not in payload:
                continue
            value = payload.get(key)
            if key == "post_crossing_recovery_turns":
                current[key] = value if isinstance(value, list) else []
            else:
                current[key] = value
            changed = True
        if changed:
            self.prama_regime_state = current

    def stop(self) -> None:
        if self.closed_at is None:
            self.closed_at = utc_now()
        self.status = "closed"

    @property
    def started_at(self) -> str:
        return self.created_at

    @property
    def stopped_at(self) -> Optional[str]:
        return self.closed_at

    def duration_seconds(self) -> float:
        start = parse_time(self.created_at)
        end = parse_time(self.closed_at) or datetime.now(timezone.utc)
        if start is None:
            return 0.0
        return round(max(0.0, (end - start).total_seconds()), 6)

    def total_tokens(self) -> int:
        return sum(int(turn.get("token_count", 0)) for turn in self.turns)

    def total_windows(self) -> int:
        return sum(len(turn.get("windows", [])) for turn in self.turns)

    def messages_for_model(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for turn in self.turns:
            messages.append({"role": "user", "content": turn["user_message"]})
            messages.append({"role": "assistant", "content": turn["assistant_message"]})
        return messages

    def live_summary(self) -> Dict[str, Any]:
        all_windows = [window for turn in self.turns for window in turn["windows"]]
        summary = summarize_windows(all_windows)
        total_tokens = self.total_tokens()
        total_windows = self.total_windows()
        prama_regime_state = {**default_prama_regime_state(), **self.prama_regime_state}
        return {
            "session_id": self.session_id,
            "model": self.model,
            "status": self.status,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "generated_at": self.generated_at,
            "output_dir": self.output_dir,
            "output_folder_name": self.output_folder_name,
            "started_at": self.created_at,
            "stopped_at": self.closed_at,
            "duration_seconds": self.duration_seconds(),
            "turn_count": len(self.turns),
            "total_tokens": total_tokens,
            "total_windows": total_windows,
            "token_count": total_tokens,
            "window_count": total_windows,
            **summary,
            **prama_regime_state,
        }

    def to_dict(self) -> Dict[str, Any]:
        prama_regime_state = {**default_prama_regime_state(), **self.prama_regime_state}
        return {
            "session_id": self.session_id,
            "model": self.model,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "generated_at": self.generated_at,
            "output_dir": self.output_dir,
            "output_folder_name": self.output_folder_name,
            "started_at": self.created_at,
            "stopped_at": self.closed_at,
            "status": self.status,
            "prama_regime_state": prama_regime_state,
            **prama_regime_state,
            "turns": self.turns,
            "summary": self.live_summary(),
        }


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_windows(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not windows:
        return {
            "avg_entropy_raw": 0.0,
            "avg_entropy_norm": 0.0,
            "avg_gap_norm": 0.0,
            "avg_rigidity": 0.0,
            "avg_uncertainty": 0.0,
            "max_entropy_std": 0.0,
            "max_entropy_range": 0.0,
        }
    return {
        "avg_entropy_raw": round(mean([float(w.get("entropy_raw", 0.0)) for w in windows]), 6),
        "avg_entropy_norm": round(mean([float(w.get("entropy_norm", 0.0)) for w in windows]), 6),
        "avg_gap_norm": round(mean([float(w.get("gap_norm", 0.0)) for w in windows]), 6),
        "avg_rigidity": round(mean([float(w.get("rigidity", 0.0)) for w in windows]), 6),
        "avg_uncertainty": round(mean([float(w.get("uncertainty", 0.0)) for w in windows]), 6),
        "max_entropy_std": round(max(float(w.get("entropy_std", 0.0)) for w in windows), 6),
        "max_entropy_range": round(max(float(w.get("entropy_range", 0.0)) for w in windows), 6),
    }
