import pytest

from aptadynamik.observer.axial_inherence_adapters import (
    AxialInherenceAdapters,
    MockAxialInherenceAdapters,
)


def test_base_axial_inherence_adapters_methods_raise_not_implemented():
    adapters = AxialInherenceAdapters()

    with pytest.raises(NotImplementedError):
        adapters.generate_turn([])
    with pytest.raises(NotImplementedError):
        adapters.embed("text")
    with pytest.raises(NotImplementedError):
        adapters.kb_consistency("text", {})
    with pytest.raises(NotImplementedError):
        adapters.kb_retraction("text", {})
    with pytest.raises(NotImplementedError):
        adapters.exo_judge("text", {})
    with pytest.raises(NotImplementedError):
        adapters.interlocutor_turn([], "condition")


def test_mock_axial_inherence_adapters_generate_required_turn_shape():
    adapters = MockAxialInherenceAdapters()
    user = adapters.interlocutor_turn([], "rotating")
    result = adapters.generate_turn([{"role": "user", "content": user}])

    assert isinstance(result["text"], str)
    assert isinstance(result["token_logprobs"], list)
    assert result["finish_reason"] == "stop"


def test_mock_axial_inherence_adapters_decline_after_delay():
    adapters = MockAxialInherenceAdapters(decline_delay=2)

    assert adapters.exo_judge("ok", {"condition": "stagnated_echo_decline", "turn": 1}) == 0.95
    assert adapters.exo_judge("ok", {"condition": "stagnated_echo_decline", "turn": 2}) < 0.5
