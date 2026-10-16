from assistant.tools import asks_balance

SMALL_WORD_LIMIT = 2
RAG_SCORE_FLOOR = 0.12


def choose_path(message: str, transfer_id: str | None, retrieval_score: float) -> str:
    """Pick one of tool, small, or rag.

    A transfer reference or a balance question calls the banking tool.
    A short message with almost no document overlap skips retrieval
    and uses the smaller model. Policy questions are answered from the
    EFT, card, and privacy notes.
    """
    if transfer_id or asks_balance(message):
        return "tool"
    word_count = len(message.split())
    if word_count <= SMALL_WORD_LIMIT and retrieval_score < RAG_SCORE_FLOOR:
        return "small"
    return "rag"
