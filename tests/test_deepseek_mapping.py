import unittest
from types import SimpleNamespace

from aptadynamik.pipelines import deepseek
from aptadynamik.pipelines.deepseek import (
    DeepSeekConfig,
    deepseek_response_to_raw_turn,
    run_deepseek_session,
)


def candidate(logprob):
    return SimpleNamespace(logprob=logprob)


def token_item(token, logprob, top_logprobs):
    return SimpleNamespace(token=token, logprob=logprob, top_logprobs=[candidate(lp) for lp in top_logprobs])


def response_with_tokens(items, text="DeepSeek response.", finish_reason="stop", model="deepseek-v4-flash"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
                logprobs=SimpleNamespace(content=items),
            )
        ],
    )


class TestDeepSeekMapping(unittest.TestCase):
    def test_response_to_raw_turn_produces_compatible_tokens(self):
        response = response_with_tokens(
            [
                token_item("A", -0.1, [-0.2, -0.8, -9999.0]),
                token_item("B", -0.4, [-0.4, -1.2, -1.8]),
            ],
            text="AB",
        )
        turn = deepseek_response_to_raw_turn(response, turn_index=3, user_message="hello")
        self.assertEqual(turn["turn_index"], 3)
        self.assertEqual(turn["user_message"], "hello")
        self.assertEqual(turn["assistant_message"], "AB")
        self.assertEqual(turn["finish_reason"], "stop")
        self.assertEqual(turn["token_count"], 2)
        self.assertEqual(len(turn["tokens"]), 2)
        self.assertIn("top1_logprob", turn["tokens"][0])
        self.assertIn("gap", turn["tokens"][0])
        self.assertIn("entropy", turn["tokens"][0])

    def test_excludes_sentinel_from_top_logprobs(self):
        response = response_with_tokens([token_item("A", -0.1, [-9999.0, -0.7])])
        turn = deepseek_response_to_raw_turn(response, turn_index=0, user_message="hello")
        self.assertNotIn(-9999.0, turn["tokens"][0]["top_logprobs"])
        self.assertEqual(turn["tokens"][0]["top_logprobs"][0], -0.1)

    def test_gap_and_entropy_are_calculated(self):
        response = response_with_tokens([token_item("A", -0.1, [-0.1, -0.6, -1.0])])
        turn = deepseek_response_to_raw_turn(response, turn_index=0, user_message="hello")
        token = turn["tokens"][0]
        self.assertAlmostEqual(token["gap"], 0.5)
        self.assertGreater(token["entropy"], 0.0)
        self.assertLessEqual(token["entropy"], 1.0)

    def test_gap_and_entropy_are_zero_with_one_valid_logprob(self):
        response = response_with_tokens([token_item("A", -0.1, [-9999.0])])
        turn = deepseek_response_to_raw_turn(response, turn_index=0, user_message="hello")
        self.assertEqual(turn["tokens"][0]["gap"], 0.0)
        self.assertEqual(turn["tokens"][0]["entropy"], 0.0)

    def test_fails_if_logprobs_missing(self):
        response = SimpleNamespace(
            model="deepseek-v4-flash",
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"), finish_reason="stop", logprobs=None)],
        )
        with self.assertRaisesRegex(ValueError, "missing choice.logprobs"):
            deepseek_response_to_raw_turn(response, turn_index=0, user_message="hello")

    def test_fails_if_logprobs_content_empty(self):
        response = response_with_tokens([])
        with self.assertRaisesRegex(ValueError, "content is empty"):
            deepseek_response_to_raw_turn(response, turn_index=0, user_message="hello")

    def test_run_deepseek_session_with_mock_completion(self):
        original = deepseek.deepseek_chat_completion

        def fake_completion(messages, config=None):
            return response_with_tokens(
                [token_item("ok", -0.2, [-0.2, -0.9])],
                text=f"turn {len([m for m in messages if m['role'] == 'user'])}",
            )

        try:
            deepseek.deepseek_chat_completion = fake_completion
            raw = run_deepseek_session(
                ["one", "two"],
                config=DeepSeekConfig(model="deepseek-chat"),
                session_id="deepseek-test-session",
            )
        finally:
            deepseek.deepseek_chat_completion = original

        self.assertEqual(raw["session_id"], "deepseek-test-session")
        self.assertEqual(raw["model"], "deepseek-v4-flash")
        self.assertEqual(raw["provider"], "deepseek")
        self.assertEqual(len(raw["turns"]), 2)
        self.assertEqual(raw["turns"][0]["assistant_message"], "turn 1")
        self.assertEqual(raw["turns"][1]["assistant_message"], "turn 2")


if __name__ == "__main__":
    unittest.main()
