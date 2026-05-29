from aptadynamik.pipelines import v2
from aptadynamik.psi_extract import extract_psi


def _family_mean(family, key):
    values = [getattr(extract_psi(prompt), key) for prompt in v2.PROMPTS[family]["prompts"]]
    return sum(values) / len(values)


def test_extract_psi_separates_low_and_high_pressure_families():
    canonical = _family_mean("canonical", "psi")
    fictional = _family_mean("fictional", "psi")
    contradictory = _family_mean("contradictory", "psi")
    saturation = _family_mean("saturation", "psi")
    low = max(canonical, fictional)

    assert contradictory > low
    assert saturation > low
    assert fictional <= canonical + 1.0


def test_saturation_load_is_higher_than_canonical():
    assert _family_mean("saturation", "load_saturation") > _family_mean("canonical", "load_saturation")


def test_contradiction_load_is_higher_than_canonical():
    assert _family_mean("contradictory", "load_contradiction") > _family_mean(
        "canonical", "load_contradiction"
    )


def test_v2_result_exports_prompt_pressure_fields():
    signals = [
        {"top1_logprob": -0.1, "gap": 0.2, "entropy": 0.1},
        {"top1_logprob": -0.2, "gap": 0.4, "entropy": 0.5},
        {"top1_logprob": -0.3, "gap": 0.7, "entropy": 0.9},
    ]
    prompt = v2.PROMPTS["saturation"]["prompts"][0]

    result = v2.run_through_prama("saturation_test", prompt, "demo", signals, v2.make_config())

    assert "psi" in result
    assert "load_saturation" in result
    assert "load_contradiction" in result
    assert "contra_weight" in result
    assert result["psi"] > 0
    assert result["load_saturation"] > 0


def test_prompt_pressure_tests_evaluate_family_table():
    table = {
        "canonical": {"psi": 0.4, "load_saturation": 0.4, "load_contradiction": 0.0},
        "fictional": {"psi": 0.0, "load_saturation": 0.0, "load_contradiction": 0.0},
        "contradictory": {"psi": 4.0, "load_saturation": 1.0, "load_contradiction": 1.0},
        "saturation": {"psi": 9.6, "load_saturation": 9.0, "load_contradiction": 0.2},
    }

    tests = v2.evaluate_prompt_pressure_tests(table)

    assert all(tests.values())
