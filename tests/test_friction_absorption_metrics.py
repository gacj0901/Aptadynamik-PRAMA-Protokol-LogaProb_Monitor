from aptadynamik.observer.friction_absorption_metrics import (
    classify_absorption_or_friction,
    compute_commitment_shift,
    compute_elaboration,
    compute_recombination,
    compute_surprise_from_prama_turn,
)


def test_classify_absorption_for_near_zero_commitment_high_recombination_low_surprise_high_elaboration():
    result = classify_absorption_or_friction(delta_C=0.01, R=0.85, S=0.12, E=20)

    assert result == "absorption"


def test_classify_friction_for_nonzero_commitment_low_recombination_high_surprise():
    result = classify_absorption_or_friction(delta_C=0.4, R=0.2, S=0.75, E=6)

    assert result == "friction"


def test_commitment_shift_is_small_for_similar_text():
    delta = compute_commitment_shift("libraries share public knowledge", "public libraries share knowledge")

    assert delta < 0.5


def test_recombination_uses_history_similarity_proxy():
    score = compute_recombination("ocean currents redistribute heat", ["bridges use arches", "currents redistribute heat"])

    assert score > 0.5


def test_surprise_uses_prama_turn_summary_entropy():
    turn = {"summary": {"avg_entropy_norm": 0.42, "max_entropy_range": 0.7}}

    assert compute_surprise_from_prama_turn(turn) == 0.42


def test_elaboration_counts_tokens():
    assert compute_elaboration("one two three") == 3
