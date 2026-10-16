import re

CARD_PATTERN = re.compile(r"(?:\d[ -]?){13,19}")


def redact(text: str) -> str:
    """Remove card-like digit runs before the text reaches the model."""
    return CARD_PATTERN.sub("[kart numarası çıkarıldı]", text)


class Memory:
    def __init__(self, profile: dict, limit: int = 6):
        self.profile = profile
        self.limit = limit
        self.turns: list[dict] = []

    def previous_user_message(self) -> str:
        for turn in reversed(self.turns):
            if turn["role"] == "user":
                return turn["content"]
        return ""

    def add(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.limit:
            self.turns = self.turns[-self.limit :]

    def reset(self) -> None:
        self.turns.clear()
