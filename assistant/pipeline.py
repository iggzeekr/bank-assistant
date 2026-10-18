import time
from dataclasses import dataclass

from assistant.dialogue import EftDraft, plan_turn
from assistant.memory import redact
from assistant.router import choose_path
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
- Tam IBAN isteme. Kayıtlı alıcı kodunu kullan.
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


class Assistant:
    def __init__(self, index, transfers: dict, accounts: dict, memory, llm, large_model: str, small_model: str, payees: dict | None = None):
        self.index = index
        self.transfers = transfers
        self.accounts = accounts
        self.payees = payees or {}
        self.memory = memory
        self.llm = llm
        self.large_model = large_model
        self.small_model = small_model
        self.draft = EftDraft()

    def reset(self) -> None:
        self.memory.reset()
        self.draft.clear()

    def ask(self, message: str, live: bool = True) -> Answer:
        cleaned = redact(message).strip()
        if not cleaned:
            raise ValueError("Boş mesaj.")
        customer_id = self.memory.profile.get("customer_id", "")
        path, tool_result = plan_turn(
            cleaned,
            self.draft,
            self.accounts,
            customer_id,
            self.payees,
            self.transfers,
        )
        previous = self.memory.previous_user_message()
        query = f"{previous} {cleaned}".strip()
        hits = self.index.search(query)
        top_score = hits[0][1] if hits else 0.0
        transfer_id = find_transfer_id(cleaned)
        if not path:
            path = choose_path(cleaned, transfer_id, top_score)
        sources: list[str] = []
        passages = []
        model = self.small_model
        if path == "dialog":
            policy = self.index.search("EFT ücreti günlük limit")
            passages = [hit for hit in policy if hit[0].source == "eft.md"][:1]
            sources = [chunk.source for chunk, _score in passages]
            model = self.large_model
        elif path == "tool" and transfer_id:
            tool_result = lookup_transfer(self.transfers, transfer_id)
            model = self.large_model
        elif path == "tool" and tool_result is None:
            tool_result = lookup_balance(self.accounts, customer_id)
            model = self.large_model
        elif path == "tool":
            model = self.large_model
        elif path == "rag":
            passages = hits[:2]
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
        return Answer(
            text=text,
            path=path,
            model=model,
            latency_ms=elapsed,
            sources=sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
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


def _dry_text(path: str, passages, tool_result) -> str:
    if path == "dialog":
        return f"EFT diyaloğu: {tool_result}"
    if path == "tool":
        return f"Araç sonucu: {tool_result}"
    if path == "rag":
        names = ", ".join(chunk.source for chunk, _score in passages) or "yok"
        return f"Belgelerden cevaplanacak: {names}"
    return "Kısa tur: belge aranmadan küçük model."
