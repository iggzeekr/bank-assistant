from assistant.tools import asks_balance

RAG_SCORE_FLOOR = 0.12
SOCIAL = {
    "merhaba",
    "selam",
    "selamlar",
    "teşekkürler",
    "teşekkür ederim",
    "sağ ol",
    "sağol",
    "evet",
    "onaylıyorum",
    "onayla",
}


def is_social(message: str) -> bool:
    return message.strip().casefold() in SOCIAL


def choose_path(message: str, transfer_id: str | None, retrieval_score: float) -> str:
    """Pick one of tool, small, or rag.

    A transfer reference or a balance question calls the banking tool.
    A greeting, thanks, or a bare confirmation skips retrieval.
    Any other question is answered from the notes, or refused when
    no note matches.
    """
    if transfer_id or asks_balance(message):
        return "tool"
    if is_social(message) and retrieval_score < RAG_SCORE_FLOOR:
        return "small"
    return "rag"
