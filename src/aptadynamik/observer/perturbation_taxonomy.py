from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List


class PerturbationType(str, Enum):
    CONTROL_NEUTRAL = "control_neutral"
    CONCRETE_CONTENT = "concrete_content"
    ABSTRACT_CONTENT = "abstract_content"
    MINIMAL_STRUCTURAL = "minimal_structural"


REQUIRED_ITEM_FIELDS = {
    "item_id",
    "topic",
    "baseline_prompt",
    "tracked_commitment",
    "perturbations",
}


def validate_perturbation_item(item: Dict[str, Any]) -> None:
    missing = sorted(REQUIRED_ITEM_FIELDS - set(item))
    if missing:
        raise ValueError(f"diagnostic item missing required field(s): {', '.join(missing)}")

    perturbations = item.get("perturbations")
    if not isinstance(perturbations, dict):
        raise ValueError(f"diagnostic item {item.get('item_id', '<unknown>')} has invalid perturbations mapping")

    required_perturbations = {
        PerturbationType.CONTROL_NEUTRAL.value,
        PerturbationType.CONCRETE_CONTENT.value,
        PerturbationType.ABSTRACT_CONTENT.value,
        "minimal_structural_rule",
    }
    missing_perturbations = sorted(required_perturbations - set(perturbations))
    if missing_perturbations:
        raise ValueError(
            f"diagnostic item {item.get('item_id', '<unknown>')} missing perturbation(s): "
            + ", ".join(missing_perturbations)
        )

    for key in required_perturbations:
        value = perturbations.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"diagnostic item {item.get('item_id', '<unknown>')} has empty perturbation '{key}'")


def load_protocol(path: str | Path) -> Dict[str, Any]:
    protocol_path = Path(path)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{protocol_path} must be JSON-compatible YAML for this dependency-free scaffold"
        ) from exc

    required = {
        "study_id",
        "model_list",
        "temperature",
        "max_tokens",
        "top_logprobs",
        "window_size",
        "arms",
        "diagnostic_items",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"protocol missing required field(s): {', '.join(missing)}")

    arms = set(protocol["arms"])
    expected_arms = {member.value for member in PerturbationType}
    if arms != expected_arms:
        raise ValueError(f"protocol arms must be exactly: {', '.join(sorted(expected_arms))}")

    if not isinstance(protocol["model_list"], list) or not protocol["model_list"]:
        raise ValueError("protocol model_list must be a non-empty list")
    if not isinstance(protocol["diagnostic_items"], list) or not protocol["diagnostic_items"]:
        raise ValueError("protocol diagnostic_items must be a non-empty list")

    for item in protocol["diagnostic_items"]:
        validate_perturbation_item(item)

    return protocol


def iter_trials(protocol: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    models: List[str] = protocol["model_list"]
    items: List[Dict[str, Any]] = protocol["diagnostic_items"]
    arms: List[str] = protocol["arms"]

    for model in models:
        for item in items:
            for arm in arms:
                perturbation_type = PerturbationType(arm)
                perturbations = item["perturbations"]
                trial = {
                    "trial_id": f"{protocol['study_id']}::{model}::{item['item_id']}::{arm}",
                    "model": model,
                    "topic": item["topic"],
                    "item_id": item["item_id"],
                    "perturbation_type": perturbation_type.value,
                    "baseline_prompt": item["baseline_prompt"],
                    "tracked_commitment": item["tracked_commitment"],
                }
                if perturbation_type is PerturbationType.MINIMAL_STRUCTURAL:
                    trial["perturbation_rule"] = perturbations["minimal_structural_rule"]
                    trial["perturbation_text"] = ""
                else:
                    trial["perturbation_text"] = perturbations[perturbation_type.value]
                    trial["perturbation_rule"] = ""
                yield trial
