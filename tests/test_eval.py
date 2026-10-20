import unittest

from evaluate import load_cases, score


class EvaluationTests(unittest.TestCase):
    def test_scripted_conversations_match_expected_paths_and_facts(self):
        cases = load_cases()
        self.assertGreaterEqual(len(cases), 20)
        failed = [item for item in score(cases, live=False) if not item["ok"]]
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()
