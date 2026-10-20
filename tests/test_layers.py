import unittest

from assistant.layers import (
    ANSWER_VOCAB,
    VOCAB,
    encode_text,
    generate,
    probabilities,
    run_demo,
    softmax,
)


class LayerTests(unittest.TestCase):
    def test_softmax_is_a_distribution(self):
        weights = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertGreater(weights[2], weights[0])

    def test_later_token_cannot_see_the_future(self):
        result = run_demo()
        self.assertEqual(result["tokens"], ["EFT", "ücreti"])
        self.assertEqual(result["attention"][0][1], 0.0)
        self.assertAlmostEqual(sum(result["attention"][1]), 1.0)
        self.assertGreater(result["attention"][1][1], result["attention"][1][0])

    def test_lower_temperature_concentrates_the_same_choice(self):
        result = run_demo()
        self.assertEqual(len(result["logits"]), len(VOCAB))
        self.assertAlmostEqual(sum(result["warm"]), 1.0)
        self.assertGreater(max(result["cool"]), max(result["warm"]))
        cool = probabilities(result["logits"], 0.2)
        self.assertEqual(VOCAB[cool.index(max(cool))], result["choice"])

    def test_prompt_runs_through_the_block(self):
        traced = encode_text("bakiyem ne kadar")
        self.assertGreater(traced["n_tokens"], 0)
        self.assertEqual(len(traced["final"]), 8)

    def test_any_sentence_chooses_a_word_from_logits(self):
        result = generate("bakiyem ne kadar")
        self.assertEqual(result["tokens"], ["bakiyem", "ne", "kadar"])
        self.assertEqual(result["attention"][0][1], 0.0)
        self.assertAlmostEqual(sum(result["attention"][-1]), 1.0)
        self.assertEqual(len(result["logits"]), len(ANSWER_VOCAB))
        self.assertIn(result["choice"], ANSWER_VOCAB)
        self.assertEqual(
            result["choice"],
            ANSWER_VOCAB[max(range(len(result["logits"])), key=result["logits"].__getitem__)],
        )


if __name__ == "__main__":
    unittest.main()
