import unittest

from aptadynamik.observer.axial_inherence_adapters import (
    AxialInherenceAdapters,
    MockAxialInherenceAdapters,
)


class TestAxialInherenceAdapters(unittest.TestCase):
    def test_base_axial_inherence_adapters_methods_raise_not_implemented(self):
        adapters = AxialInherenceAdapters()

        with self.assertRaises(NotImplementedError):
            adapters.generate_turn([])
        with self.assertRaises(NotImplementedError):
            adapters.embed("text")
        with self.assertRaises(NotImplementedError):
            adapters.kb_consistency("text", {})
        with self.assertRaises(NotImplementedError):
            adapters.kb_retraction("text", {})
        with self.assertRaises(NotImplementedError):
            adapters.exo_judge("text", {})
        with self.assertRaises(NotImplementedError):
            adapters.interlocutor_turn([], "condition")

    def test_mock_axial_inherence_adapters_generate_required_turn_shape(self):
        adapters = MockAxialInherenceAdapters()
        user = adapters.interlocutor_turn([], "rotating")
        result = adapters.generate_turn([{"role": "user", "content": user}])

        self.assertIsInstance(result["text"], str)
        self.assertIsInstance(result["token_logprobs"], list)
        self.assertEqual(result["finish_reason"], "stop")

    def test_mock_axial_inherence_adapters_decline_after_delay(self):
        adapters = MockAxialInherenceAdapters(decline_delay=2)

        self.assertEqual(adapters.exo_judge("ok", {"condition": "stagnated_echo_decline", "turn": 1}), 0.95)
        self.assertLess(adapters.exo_judge("ok", {"condition": "stagnated_echo_decline", "turn": 2}), 0.5)


if __name__ == "__main__":
    unittest.main()
