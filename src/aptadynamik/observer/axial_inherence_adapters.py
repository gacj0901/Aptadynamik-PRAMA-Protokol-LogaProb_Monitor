from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Tuple


class AxialInherenceAdapters:
    def generate_turn(self, context_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def kb_consistency(self, turn_text: str, kb_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        raise NotImplementedError

    def kb_retraction(self, turn_text: str, kb_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        raise NotImplementedError

    def exo_judge(self, turn_text: str, task_context: Dict[str, Any]) -> float:
        raise NotImplementedError

    def interlocutor_turn(self, session_so_far: List[Dict[str, Any]], condition: str) -> str:
        raise NotImplementedError


class MockAxialInherenceAdapters(AxialInherenceAdapters):
    def __init__(self, decline_delay: int = 24) -> None:
        self.turn_index = 0
        self.decline_delay = decline_delay

    def generate_turn(self, context_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        condition = context_messages[-1]["content"].split("condition=")[-1].strip()
        turn = self.turn_index
        self.turn_index += 1
        if "echo" in condition or "stagnated" in condition:
            text = "stable consistent continuation " * 4
        else:
            phase = turn % 3
            if phase == 0:
                text = f"novel branch {turn} introduces a fresh angle"
            elif phase == 1:
                text = f"consistent branch {turn} preserves the main claim"
            else:
                text = f"release branch {turn} retracts an optional detail"
        return {
            "text": text.strip(),
            "token_logprobs": [-0.2 for _ in text.split()],
            "finish_reason": "stop",
        }

    def embed(self, text: str) -> List[float]:
        lower = text.lower()
        if "stable consistent" in lower:
            return [1.0, 0.0, 0.0]
        if "novel" in lower:
            match = re.search(r"(\d+)", lower)
            turn = int(match.group(1)) if match else 0
            angle = turn * 2.399963
            return [math.cos(angle), math.sin(angle), 0.0]
        if "consistent" in lower:
            return [0.0, 0.0, 1.0]
        if "release" in lower or "retract" in lower:
            return [1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0]
        return [0.5, 0.5, 0.5]

    def kb_consistency(self, turn_text: str, kb_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        lower = turn_text.lower()
        if "release" in lower or "retract" in lower:
            rate = 0.35
        elif "consistent" in lower or "stable" in lower:
            rate = 0.02
        elif "novel" in lower:
            rate = 0.80
        else:
            rate = 0.08
        return rate, kb_state

    def kb_retraction(self, turn_text: str, kb_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        lower = turn_text.lower()
        if "release" in lower or "retract" in lower:
            drop = 0.92
        elif "stable" in lower:
            drop = 0.04
        else:
            drop = 0.12
        return drop, kb_state

    def exo_judge(self, turn_text: str, task_context: Dict[str, Any]) -> float:
        condition = str(task_context.get("condition", ""))
        turn = int(task_context.get("turn", 0))
        if "decline" in condition and turn >= self.decline_delay:
            return 0.25
        return 0.95

    def interlocutor_turn(self, session_so_far: List[Dict[str, Any]], condition: str) -> str:
        return f"continue; condition={condition}"
