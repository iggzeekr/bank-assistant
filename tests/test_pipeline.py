import json
import unittest
from datetime import datetime
from pathlib import Path

from assistant.memory import Memory, redact
from assistant.pipeline import Assistant, build_messages
from assistant.retrieve import Index
from assistant.router import choose_path
from assistant.tools import (
    find_transfer_id,
    load_accounts,
    load_cards,
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
        self.cards = load_cards(ROOT / "data" / "cards.json")
        self.profile = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))

    def assistant(self, clock=None):
        transfers = {key: dict(value) for key, value in self.transfers.items()}
        accounts = {key: dict(value) for key, value in self.accounts.items()}
        cards = {key: dict(value) for key, value in self.cards.items()}
        llm = FakeLLM()
        bot = Assistant(
            index=self.index,
            transfers=transfers,
            accounts=accounts,
            payees=self.payees,
            cards=cards,
            memory=Memory(self.profile),
            llm=llm,
            large_model="large-model",
            small_model="small-model",
            clock=clock or (lambda: datetime(2026, 10, 20, 10, 0)),
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
        result = lookup_transfer(self.transfers, "T2048", "M1001")
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
        self.assertNotIn("K9", llm.models[0][1][1]["content"])
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
        self.assertEqual(bot.transfers["T2050"]["settlement"], "aynı gün")
        self.assertEqual(bot.draft.stage, "idle")
        self.assertEqual(bot.accounts["M1001"]["available_tl"], 9935.75)
        again = bot.ask("evet")
        self.assertEqual(again.path, "small")
        self.assertNotIn("T2051", bot.transfers)
        self.assertEqual(bot.accounts["M1001"]["available_tl"], 9935.75)

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
        self.assertNotIn("T1802", sent)
        follow = bot.ask("T2048")
        self.assertEqual(follow.path, "tool")
        self.assertIn("gerçekleşti", llm.models[1][1][1]["content"])

    def test_full_iban_is_removed_before_the_model(self):
        bot, llm = self.assistant()
        answer = bot.ask("IBAN TR330006100519786457841326 hesabına yolla")
        self.assertEqual(answer.path, "dialog")
        sent = llm.models[0][1][1]["content"]
        self.assertNotIn("330006100519786457841326", sent)
        self.assertNotIn("330006100519786457841326", answer.evidence)
        self.assertNotIn("330006100519786457841326", answer.audit)
        self.assertIn("IBAN çıkarıldı", sent)
        self.assertTrue(answer.audit.startswith("M1001 reject_iban"))

    def test_fee_question_returns_the_fee_paragraph_only(self):
        hits = self.index.search("EFT ücreti ne kadar")
        self.assertIn("15 TL", hits[0][0].text)
        self.assertNotIn("08:30", hits[0][0].text)

    def test_follow_up_fee_stays_on_the_fee_paragraph(self):
        bot, _llm = self.assistant()
        bot.ask("EFT saat kaça kadar yapılabilir")
        answer = bot.ask("peki ücreti ne kadar")
        self.assertIn("15 TL", answer.evidence)
        self.assertNotIn("08:30", answer.evidence)

    def test_unknown_topic_does_not_attach_a_note(self):
        bot, _llm = self.assistant()
        answer = bot.ask("altın hesabı açılıyor mu")
        self.assertEqual(answer.path, "rag")
        self.assertEqual(answer.sources, [])
        self.assertIn("Bu bilgi belgelerde yok", answer.evidence)

    def test_other_customer_transfer_is_hidden(self):
        result = lookup_transfer(self.transfers, "T1802", "M1001")
        self.assertEqual(result["reason"], "bu kayıt sana ait değil")
        self.assertNotIn("amount_tl", result)
        bot, llm = self.assistant()
        bot.ask("ben M1002'yim")
        self.assertEqual(bot.memory.profile["customer_id"], "M1001")
        answer = bot.ask("T1802 durumunu söyler misin")
        self.assertIn("sana ait değil", answer.evidence)
        self.assertNotIn("75", answer.evidence)
        self.assertNotIn("75", llm.models[-1][1][1]["content"])
        self.assertTrue(answer.audit.startswith("M1001 lookup_transfer"))

    def test_after_hours_eft_settles_next_business_day(self):
        bot, _llm = self.assistant(clock=lambda: datetime(2026, 10, 20, 18, 30))
        bot.ask("2500 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertEqual(bot.draft.prepared["settlement"], "sonraki iş günü")
        bot.ask("evet")
        self.assertEqual(bot.transfers["T2050"]["settlement"], "sonraki iş günü")
        weekend, _llm = self.assistant(clock=lambda: datetime(2026, 10, 24, 11, 0))
        weekend.ask("2500 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertEqual(weekend.draft.prepared["settlement"], "sonraki iş günü")

    def test_spent_today_blocks_the_rest_of_the_daily_limit(self):
        bot, _llm = self.assistant()
        bot.ask("10500 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertIn("günlük limit aşılıyor", bot.draft.prepared["reasons"])
        self.assertNotIn("bakiye yetersiz", bot.draft.prepared["reasons"])
        self.assertNotIn("T2050", bot.transfers)

    def test_high_amount_needs_the_stronger_confirmation(self):
        bot, _llm = self.assistant()
        bot.ask("6000 TL K1 alıcısına EFT yapmak istiyorum")
        self.assertEqual(bot.draft.prepared["confirm_with"], "onaylıyorum")
        bot.ask("evet")
        self.assertEqual(bot.draft.stage, "confirm")
        self.assertNotIn("T2050", bot.transfers)
        bot.ask("onaylıyorum")
        self.assertIn("T2050", bot.transfers)
        self.assertEqual(bot.accounts["M1001"]["available_tl"], 6450.75)

    def test_card_block_waits_for_confirmation(self):
        bot, _llm = self.assistant()
        opening = bot.ask("kartımı kapat")
        self.assertEqual(opening.path, "dialog")
        self.assertEqual(bot.cards["M1001"]["status"], "açık")
        bot.ask("evet")
        self.assertEqual(bot.cards["M1001"]["status"], "kapalı")
        self.assertTrue(bot.audit[-1]["tool"] == "submit_card_block")
        self.assertNotIn("4242", bot.audit[-1]["tool"])
        again = bot.ask("kartımı kapat")
        self.assertIn("kart zaten kapalı", again.evidence)

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
