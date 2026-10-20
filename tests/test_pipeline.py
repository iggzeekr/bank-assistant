import json
import unittest
from pathlib import Path

from assistant.memory import Memory, redact
from assistant.pipeline import Assistant, build_messages
from assistant.retrieve import Index
from assistant.router import choose_path
from assistant.tools import (
    find_transfer_id,
    load_accounts,
    load_payees,
    load_transfers,
    lookup_balance,
    lookup_transfer,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def __init__(self):
        self.models = []

    def complete(self, messages, model):
        self.models.append((model, messages))
        from assistant.llm import Completion

        return Completion(text=f"yanıt:{model}", prompt_tokens=10, completion_tokens=4)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = Index.from_dir(ROOT / "knowledge")
        self.transfers = load_transfers(ROOT / "data" / "transfers.json")
        self.accounts = load_accounts(ROOT / "data" / "accounts.json")
        self.payees = load_payees(ROOT / "data" / "payees.json")
        self.profile = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))

    def assistant(self):
        transfers = {key: dict(value) for key, value in self.transfers.items()}
        llm = FakeLLM()
        bot = Assistant(
            index=self.index,
            transfers=transfers,
            accounts=self.accounts,
            payees=self.payees,
            memory=Memory(self.profile),
            llm=llm,
            large_model="large-model",
            small_model="small-model",
        )
        return bot, llm

    def test_redacts_card_numbers(self):
        cleaned = redact("kartım 4242 4242 4242 4242 oldu")
        self.assertNotIn("4242", cleaned)
        self.assertIn("çıkarıldı", cleaned)

    def test_retrieval_finds_eft_rules(self):
        hits = self.index.search("EFT saat kaça kadar yapılabilir")
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].source, "eft.md")

    def test_transfer_tool_returns_masked_account(self):
        result = lookup_transfer(self.transfers, "T2048")
        self.assertEqual(result["status"], "gerçekleşti")
        self.assertEqual(result["amount_tl"], 2500)
        self.assertIn("**", result["recipient_account_masked"])
        self.assertNotIn("iban", result)
        self.assertEqual(find_transfer_id("T2048 transferim gerçekleşti mi"), "T2048")

    def test_balance_tool_returns_available_amount_only(self):
        result = lookup_balance(self.accounts, "M1001")
        self.assertEqual(result["available_tl"], 12450.75)
        self.assertIn("**", result["account_masked"])
        self.assertNotIn("iban", result)

    def test_short_greeting_uses_small_model(self):
        self.assertEqual(choose_path("merhaba", None, 0.0), "small")
        bot, llm = self.assistant()
        answer = bot.ask("merhaba")
        self.assertEqual(answer.path, "small")
        self.assertEqual(llm.models[0][0], "small-model")
        self.assertEqual(answer.sources, [])

    def test_eft_question_uses_documents(self):
        bot, llm = self.assistant()
        answer = bot.ask("EFT saat kaça kadar yapılabilir")
        self.assertEqual(answer.path, "rag")
        self.assertEqual(llm.models[0][0], "large-model")
        self.assertIn("eft.md", answer.sources)

    def test_follow_up_stays_on_transfer_fee(self):
        bot, _llm = self.assistant()
        bot.ask("EFT saat kaça kadar yapılabilir")
        answer = bot.ask("peki ücreti ne kadar")
        self.assertEqual(answer.path, "rag")
        self.assertIn("eft.md", answer.sources)

    def test_lost_card_uses_card_note(self):
        bot, _llm = self.assistant()
        answer = bot.ask("kartımı kaybettim ne yapmalıyım")
        self.assertEqual(answer.path, "rag")
        self.assertIn("kart.md", answer.sources)

    def test_transfer_question_calls_tool_then_model(self):
        bot, llm = self.assistant()
        answer = bot.ask("T2048 transferim gerçekleşti mi")
        self.assertEqual(answer.path, "tool")
        self.assertEqual(llm.models[0][0], "large-model")
        self.assertIn("gerçekleşti", llm.models[0][1][1]["content"])
        self.assertIn("bireysel", llm.models[0][1][1]["content"])
        self.assertIn("2500", llm.models[0][1][1]["content"])

    def test_balance_question_calls_account_tool(self):
        bot, llm = self.assistant()
        answer = bot.ask("bakiyem ne kadar")
        self.assertEqual(answer.path, "tool")
        self.assertIn("12450.75", llm.models[0][1][1]["content"])
        messages = build_messages(
            self.profile,
            [],
            "bakiyem ne kadar",
            [],
            lookup_balance(self.accounts, "M1001"),
        )
        self.assertIn("Ayşe", messages[1]["content"])
        self.assertNotIn("TR12", answer.text)

    def test_memory_keeps_recent_turns_only(self):
        memory = Memory(self.profile, limit=2)
        memory.add("user", "bir")
        memory.add("assistant", "iki")
        memory.add("user", "üç")
        self.assertEqual([turn["content"] for turn in memory.turns], ["iki", "üç"])

    def test_eft_dialogue_collects_then_waits_for_confirmation(self):
        bot, llm = self.assistant()
        opening = bot.ask("EFT yapmak istiyorum")
        self.assertEqual(opening.path, "dialog")
        self.assertEqual(bot.draft.stage, "collect")
        self.assertIn("K1", llm.models[0][1][1]["content"])
        proposal = bot.ask("2500 K1")
        self.assertEqual(proposal.path, "dialog")
        self.assertEqual(bot.draft.stage, "confirm")
        self.assertIn("prepare_eft", llm.models[1][1][1]["content"])
        self.assertNotIn("T2050", bot.transfers)

    def test_confirmed_eft_is_recorded_only_after_yes(self):
        bot, llm = self.assistant()
        bot.ask("2500 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertEqual(bot.draft.stage, "confirm")
        answer = bot.ask("evet")
        self.assertEqual(answer.path, "dialog")
        self.assertIn("submit_eft", llm.models[-1][1][1]["content"])
        self.assertIn("T2050", bot.transfers)
        self.assertEqual(bot.transfers["T2050"]["status"], "onay bekliyor")
        self.assertEqual(bot.draft.stage, "idle")

    def test_over_limit_eft_is_not_offered_for_confirmation(self):
        bot, _llm = self.assistant()
        bot.ask("200000 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertEqual(bot.draft.stage, "collect")
        self.assertFalse(bot.draft.prepared["allowed"])
        self.assertIn("günlük limit aşılıyor", bot.draft.prepared["reasons"])
        self.assertNotIn("T2050", bot.transfers)

    def test_cancel_clears_the_draft(self):
        bot, _llm = self.assistant()
        bot.ask("2500 K1 için EFT yapmak istiyorum")
        bot.ask("vazgeç")
        self.assertEqual(bot.draft.stage, "idle")
        self.assertNotIn("T2050", bot.transfers)

    def test_transfer_list_does_not_require_a_reference(self):
        bot, llm = self.assistant()
        answer = bot.ask("transferlerim")
        self.assertEqual(answer.path, "tool")
        self.assertIn("T2048", llm.models[0][1][1]["content"])

    def test_missing_reference_asks_which_transfer(self):
        bot, llm = self.assistant()
        answer = bot.ask("transferim nerede")
        self.assertEqual(answer.path, "dialog")
        sent = llm.models[0][1][1]["content"]
        self.assertIn("clarify_transfer", sent)
        self.assertIn("T2048", sent)
        self.assertIn("T2049", sent)
        follow = bot.ask("T2048")
        self.assertEqual(follow.path, "tool")
        self.assertIn("gerçekleşti", llm.models[1][1][1]["content"])

    def test_full_iban_is_removed_before_the_model(self):
        bot, llm = self.assistant()
        answer = bot.ask("IBAN TR330006100519786457841326 hesabına yolla")
        self.assertEqual(answer.path, "dialog")
        sent = llm.models[0][1][1]["content"]
        self.assertNotIn("330006100519786457841326", sent)
        self.assertIn("IBAN çıkarıldı", sent)

    def test_embedding_matches_a_different_word_for_the_same_fee(self):
        plain = Index.from_dir(ROOT / "knowledge")
        self.assertFalse(any(chunk.source == "eft.md" for chunk, _score in plain.search("masraf ne kadar")))
        embedded = Index.from_dir(ROOT / "knowledge", embedder=_FeeSynonyms())
        hits = embedded.search("masraf ne kadar")
        self.assertEqual(hits[0][0].source, "eft.md")
        self.assertEqual(embedded.method, "embedding")


class _FeeSynonyms:
    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if "masraf" in lowered or "ücret" in lowered or "eft" in lowered:
                vectors.append([1.0, 0.0])
            elif "kart" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return vectors


if __name__ == "__main__":
    unittest.main()
