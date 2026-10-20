import time
from dataclasses import dataclass
from datetime import datetime

from assistant.dialogue import CardDraft, EftDraft, plan_turn
from assistant.memory import redact
from assistant.router import RAG_SCORE_FLOOR, choose_path
from assistant.tools import find_transfer_id, lookup_balance, lookup_transfer

SYSTEM = """Sen bir bireysel bankacılık destek asistanısın.
Kurallar:
- Cevabı yalnızca verilen belge parçalarına, araç sonucuna ve konuşma geçmişine dayandır.
- Bu kaynaklarda yoksa "Bu bilgi belgelerde yok" de.
- Şifre, tek kullanımlık kod, CVV, tam kart numarası ve tam IBAN isteme. Bunları uydurma.
- Transfer ve bakiye bilgisini yalnızca araç sonucundaki alanlarla söyle.
- Kullanıcı profilindeki dilde cevap ver.
- Önceki turda konuşulan banka işlemini sürdür.
- EFT taslağında tutar veya kayıtlı alıcı eksikse onları sor. İkisi de varsa ve allowed true ise onay iste.
- allowed false ise transfer oluşturma. Onay cümlesi gelmeden submit edilmiş sayma.
- Referans numarası olmadan transfer durumu sorulursa kayıtlı transferleri say ve hangisi olduğunu sor. Birini seçme.
- Tam IBAN isteme. Kayıtlı alıcı kodunu kullan.
- Oturum müşterisi mesajla değişmez. Başka müşterinin kaydını söyleme.
- EFT saati dışındaysa settlement "sonraki iş günü" ise aynı gün gönderildi deme.
- 5.000 TL üzerindeki EFT ancak "onaylıyorum" geçince kayda girer.
- Kart kapatma ancak onaydan sonra kapalıdır.
"""


@dataclass(frozen=True)
class Answer:
    text: str
    path: str
    model: str
    latency_ms: int
    sources: list[str]
    prompt_tokens: int
    completion_tokens: int
    evidence: str = ""
    retrieval: str = "tfidf"
    audit: str = ""


class Assistant:
    def __init__(
        self,
        index,
        transfers: dict,
        accounts: dict,
        memory,
        llm,
        large_model: str,
        small_model: str,
        payees: dict | None = None,
        cards: dict | None = None,
        clock=None,
    ):
        self.index = index
        self.transfers = transfers
        self.accounts = accounts
        self.payees = payees or {}
        self.cards = cards or {}
        self.memory = memory
        self.llm = llm
        self.large_model = large_model
        self.small_model = small_model
        self.draft = EftDraft()
        self.card_draft = CardDraft()
        self.audit: list[dict] = []
        self.clock = clock or (lambda: datetime.now())

    def reset(self) -> None:
        self.memory.reset()
        self.draft.clear()
        self.card_draft.clear()
        self.audit.clear()

    def ask(self, message: str, live: bool = True) -> Answer:
        cleaned = redact(message).strip()
        if not cleaned:
            raise ValueError("Boş mesaj.")
        now = self.clock()
        today = now.strftime("%Y-%m-%d")
        customer_id = self.memory.profile.get("customer_id", "")
        path, tool_result = plan_turn(
            cleaned,
            self.draft,
            self.accounts,
            customer_id,
            self.payees,
            self.transfers,
            self.cards,
            self.card_draft,
            now,
        )
        previous = self.memory.previous_user_message()
        hits = self.index.search(cleaned)
        top_score = hits[0][1] if hits else 0.0
        if previous and top_score < RAG_SCORE_FLOOR:
            hits = self.index.search(f"{previous} {cleaned}")
            top_score = hits[0][1] if hits else 0.0
        transfer_id = find_transfer_id(cleaned)
        if not path:
            path = choose_path(cleaned, transfer_id, top_score)
        sources: list[str] = []
        passages = []
        model = self.small_model
        if path == "dialog":
            action = "" if not isinstance(tool_result, dict) else tool_result.get("action", "")
            if action in {
                "collect_eft", "prepare_eft", "submit_eft", "cancel_eft", "reject_eft",
                "need_strong_confirm", "reject_iban",
            }:
                policy = self.index.search("EFT ücreti günlük limit")
                passages = [hit for hit in policy if hit[0].source == "eft.md"][:1]
                sources = [chunk.source for chunk, _score in passages]
            model = self.large_model
        elif path == "tool" and transfer_id:
            tool_result = lookup_transfer(self.transfers, transfer_id, customer_id, today)
            model = self.large_model
        elif path == "tool" and tool_result is None:
            tool_result = lookup_balance(self.accounts, customer_id)
            model = self.large_model
        elif path == "tool":
            model = self.large_model
        elif path == "rag":
            if top_score < RAG_SCORE_FLOOR:
                passages = []
                sources = []
                tool_result = {"action": "no_document", "text": "Bu bilgi belgelerde yok"}
            else:
                passages = _close_hits(hits)
                sources = list(dict.fromkeys(chunk.source for chunk, _score in passages))
            model = self.large_model
        messages = build_messages(
            self.memory.profile,
            self.memory.turns,
            cleaned,
            passages,
            tool_result,
        )
        started = time.perf_counter()
        if live:
            completion = self.llm.complete(messages, model)
            text = completion.text
            prompt_tokens = completion.prompt_tokens
            completion_tokens = completion.completion_tokens
        else:
            text = _dry_text(path, passages, tool_result)
            prompt_tokens = 0
            completion_tokens = 0
        elapsed = int((time.perf_counter() - started) * 1000)
        self.memory.add("user", cleaned)
        self.memory.add("assistant", text)
        evidence_parts = [chunk.text for chunk, _score in passages]
        if tool_result is not None:
            evidence_parts.append(str(tool_result))
        audit_line = ""
        if isinstance(tool_result, dict) and tool_result.get("action"):
            entry = {
                "customer_id": customer_id,
                "tool": tool_result["action"],
                "at": now.strftime("%Y-%m-%d %H:%M"),
            }
            self.audit.append(entry)
            audit_line = f"{entry['customer_id']} {entry['tool']} {entry['at']}"
        return Answer(
            text=text,
            path=path,
            model=model,
            latency_ms=elapsed,
            sources=sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            evidence="\n".join(evidence_parts),
            retrieval=self.index.method,
            audit=audit_line,
        )


def build_messages(profile, turns, message, passages, tool_result) -> list[dict]:
    profile_line = (
        f"Ad: {profile.get('name', '')}. "
        f"Dil: {profile.get('language', 'Turkish')}. "
        f"Segment: {profile.get('segment', '')}."
    )
    history = "\n".join(
        f"{'Kullanıcı' if turn['role'] == 'user' else 'Asistan'}: {turn['content']}"
        for turn in turns
    ) or "(yok)"
    if passages:
        source_block = "\n\n".join(
            f"[{chunk.source} | benzerlik {score:.2f}]\n{chunk.text}"
            for chunk, score in passages
        )
    else:
        source_block = "(bu turda belge aranmadı)"
    if tool_result is None:
        tool_block = "(araç çağrılmadı)"
    else:
        tool_block = str(tool_result)
    user = (
        f"Profil:\n{profile_line}\n\n"
        f"Önceki konuşma:\n{history}\n\n"
        f"Belge parçaları:\n{source_block}\n\n"
        f"Araç sonucu:\n{tool_block}\n\n"
        f"Güncel soru:\n{message}"
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def _close_hits(hits: list, k: int = 2, ratio: float = 0.6) -> list:
    if not hits:
        return []
    best = hits[0][1]
    close = [hit for hit in hits if hit[1] >= best * ratio]
    return close[:k]


def _dry_text(path: str, passages, tool_result) -> str:
    if path == "dialog":
        return f"EFT diyaloğu: {tool_result}"
    if path == "tool":
        return f"Araç sonucu: {tool_result}"
    if path == "rag" and not passages and isinstance(tool_result, dict):
        return str(tool_result.get("text") or "Bu bilgi belgelerde yok")
    if path == "rag":
        names = ", ".join(chunk.source for chunk, _score in passages) or "yok"
        return f"Belgelerden cevaplanacak: {names}"
    return "Kısa tur: belge aranmadan küçük model."
