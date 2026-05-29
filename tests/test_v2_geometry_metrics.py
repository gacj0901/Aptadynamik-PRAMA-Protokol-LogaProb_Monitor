from aptadynamik.pipelines import v2


def test_compute_derived_returns_geometry_metrics():
    derived = v2.compute_derived(avg_gap=2.0, avg_entropy=1.0)

    assert set(["gap_norm", "entropy_norm", "rigidity", "uncertainty", "margin"]).issubset(derived)
    assert 0.0 <= derived["gap_norm"] <= 1.0
    assert 0.0 <= derived["entropy_norm"] <= 1.0
    assert 0.0 <= derived["rigidity"] <= 1.0
    assert 0.0 <= derived["uncertainty"] <= 1.0
    assert 0.0 <= derived["margin"] <= 1.0


def test_window_aggregate_includes_intra_window_variation_metrics():
    signals = [
        {"top1_logprob": -0.1, "gap": 0.2, "entropy": 0.1},
        {"top1_logprob": -0.2, "gap": 0.4, "entropy": 0.5},
        {"top1_logprob": -0.3, "gap": 0.7, "entropy": 0.9},
    ]

    windows = v2.window_aggregate(signals, window_size=3)

    assert len(windows) == 1
    window = windows[0]
    assert "entropy_var" in window
    assert "entropy_std" in window
    assert "entropy_range" in window
    assert "gap_std" in window
    assert window["entropy_std"] > 0
    assert window["entropy_range"] == 0.8
    assert window["gap_std"] > 0


def test_build_families_to_run_respects_prompt_limit(monkeypatch):
    monkeypatch.setenv("PRAMA_PROMPT_LIMIT", "1")

    families = v2.build_families_to_run()

    assert set(families) == set(v2.PROMPTS)
    assert all(len(family["prompts"]) == 1 for family in families.values())


def test_geometry_tests_evaluate_synthetic_family_table():
    table = {
        "canonical": {
            "avg_rigidity": 0.90,
            "avg_entropy_std": 0.05,
            "avg_entropy_range": 0.10,
        },
        "fictional": {
            "avg_rigidity": 0.40,
            "avg_entropy_std": 0.20,
            "avg_entropy_range": 0.30,
        },
        "contradictory": {
            "avg_rigidity": 0.30,
            "avg_entropy_std": 0.35,
            "avg_entropy_range": 0.55,
        },
        "saturation": {
            "avg_rigidity": 0.20,
            "avg_entropy_std": 0.60,
            "avg_entropy_range": 0.80,
        },
    }

    tests = v2.evaluate_geometry_tests(table)

    assert tests["G1_entropy_std_saturation_gt_canonical"]
    assert tests["G2_entropy_range_saturation_gt_canonical"]
    assert tests["G3_structural_entropy_std_gt_semantic"]
    assert tests["G4_structural_entropy_range_gt_semantic"]
    assert tests["G5_canonical_rigidity_highest"]
