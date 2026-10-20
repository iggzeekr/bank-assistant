import unittest

from assistant.pipeline import Answer
from assistant.present import present


def answer(path, payload=None, evidence=""):
    return Answer(
        text="",
        path=path,
        model="m",
        latency_ms=1,
        sources=["eft.md"] if path == "rag" and evidence else [],
        prompt_tokens=0,
        completion_tokens=0,
        evidence=evidence,
        payload=payload,
    )


class PresentTests(unittest.TestCase):
    def test_balance_is_a_sentence(self):
        said = present(
            "bakiyem ne kadar",
            answer(
                "tool",
                {
                    "action": "lookup_balance",
                    "found": True,
                    "available_tl": 12450.75,
                    "account_masked": "TR** **** **** 1001",
                },
            ),
        )
        self.assertIn("12.450,75 TL", said)
        self.assertIn("TR**", said)

    def test_other_customer_amount_is_not_spoken(self):
        said = present(
            "T1802",
            answer(
                "tool",
                {"action": "lookup_transfer", "id": "T1802", "found": False, "reason": "bu kayıt sana ait değil"},
            ),
        )
        self.assertIn("sana ait değil", said)
        self.assertNotIn("75", said)

    def test_missing_note_is_explicit(self):
        said = present("altın", answer("rag", {"action": "no_document"}))
        self.assertEqual(said, "Bu bilgi belgelerde yok.")

    def test_fee_paragraph_is_shown_as_written(self):
        note = "EFT ücreti 5.000 TL altındaki transferlerde 15 TL."
        said = present("ücret", answer("rag", evidence=note))
        self.assertEqual(said, note)


if __name__ == "__main__":
    unittest.main()
