import importlib.util
from pathlib import Path

from aptadynamik.observer.perturbation_taxonomy import (
    PerturbationType,
    iter_trials,
    load_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols" / "minimal_structural_perturbations.yaml"
RUNNER_PATH = ROOT / "scripts" / "prama_perturbation_study_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("prama_perturbation_study_runner", RUNNER_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_protocol_loads_correctly():
    protocol = load_protocol(PROTOCOL_PATH)

    assert protocol["study_id"] == "prama_minimal_structural_perturbation_v0"
    assert protocol["diagnostic_items"]


def test_all_perturbation_types_are_recognized():
    values = {member.value for member in PerturbationType}

    assert values == {
        "control_neutral",
        "concrete_content",
        "abstract_content",
        "minimal_structural",
    }


def test_iter_trials_includes_text_or_rule():
    protocol = load_protocol(PROTOCOL_PATH)
    trials = list(iter_trials(protocol))

    assert trials
    assert any(trial["perturbation_rule"] for trial in trials if trial["perturbation_type"] == "minimal_structural")
    assert all("trial_id" in trial for trial in trials)


def test_dry_run_generates_trials_csv_and_study_design(tmp_path):
    RUNNER.run_dry_run(PROTOCOL_PATH, tmp_path)

    assert (tmp_path / "trials.csv").exists()
    assert (tmp_path / "study_design.md").exists()


def test_study_design_includes_hypotheses_and_falsification_criteria(tmp_path):
    RUNNER.run_dry_run(PROTOCOL_PATH, tmp_path)
    design = (tmp_path / "study_design.md").read_text(encoding="utf-8")

    assert "H_A" in design
    assert "H_B" in design
    assert "H0" in design
    assert "Falsification Criteria" in design
