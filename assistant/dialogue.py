import re

from assistant.tools import (
    STRONG_CONFIRM_OVER_TL,
    find_transfer_id,
    lookup_balance,
    lookup_transfer,
    owned_transfer_ids,
    prepare_card_block,
    prepare_eft,
    submit_card_block,
    submit_eft,
)

START = re.compile(
    r"\b(?:eft|havale)\b.{0,40}\b(?:yapmak|göndermek|atmak|talimat)\b|\bpara göndermek\b",
    re.IGNORECASE,
)
LIST = re.compile(r"\btransferlerim\b|\bson transfer", re.IGNORECASE)
AMBIGUOUS = re.compile(r"\b(?:transferim|havalem)\b", re.IGNORECASE)
PAYEE = re.compile(r"\bK\d\b", re.IGNORECASE)
AMOUNT = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d{2,7})\b")
STRONG = re.compile(r"\bonaylıyorum\b", re.IGNORECASE)
WEAK = re.compile(r"\b(?:evet|onayla)\b", re.IGNORECASE)
CANCEL = re.compile(r"\b(?:vazgeç|iptal|hayır)\b", re.IGNORECASE)
CARD_CLOSE = re.compile(r"\bkart\w*\b.{0,30}\bkapat\w*\b|\bkapat\w*\b.{0,30}\bkart\w*\b", re.IGNORECASE)
IDENTITY = re.compile(r"\bM\d{4}\b", re.IGNORECASE)
IBAN_MARK = "[IBAN çıkarıldı]"


class EftDraft:
    def __init__(self):
        self.active = False
        self.amount = None
        self.payee_id = None
        self.stage = "idle"
        self.prepared = None

    def clear(self) -> None:
        self.active = False
        self.amount = None
        self.payee_id = None
        self.stage = "idle"
        self.prepared = None


class CardDraft:
    def __init__(self):
        self.stage = "idle"
        self.prepared = None

    def clear(self) -> None:
        self.stage = "idle"
        self.prepared = None


def wants_new_eft(message: str) -> bool:
    return START.search(message) is not None


def wants_transfer_list(message: str) -> bool:
    return LIST.search(message) is not None and find_transfer_id(message) is None


def wants_transfer_clarification(message: str) -> bool:
    return AMBIGUOUS.search(message) is not None and find_transfer_id(message) is None


def parse_amount(message: str) -> int | None:
    without_ids = PAYEE.sub(" ", message)
    without_ids = re.sub(r"\bT\d{4}\b", " ", without_ids, flags=re.IGNORECASE)
    match = AMOUNT.search(without_ids)
    if match is None:
        return None
    return int(match.group(1).replace(".", ""))


def parse_payee(message: str) -> str | None:
    match = PAYEE.search(message)
    if match is None:
        return None
    return match.group(0).upper()


def confirmation_level(message: str) -> str:
    if STRONG.search(message):
        return "strong"
    if WEAK.search(message):
        return "weak"
    return ""


def _own_payees(payees: dict, customer_id: str) -> list[dict]:
    return [
        {"id": row["id"], "label": row["label"], "account_masked": row["account_masked"]}
        for row in payees.values()
        if row.get("customer_id") == customer_id
    ]


def plan_turn(
    message,
    draft,
    accounts,
    customer_id,
    payees,
    transfers,
    cards,
    card_draft,
    now,
) -> tuple[str, dict | None]:
    """Return the path and the tool payload for this turn.

    An open EFT draft keeps slot-filling on the dialog path. A finished
    draft is submitted only after the confirmation that amount requires.
    Card blocking uses its own draft. The customer id is never read from the message.
    """
    today = now.strftime("%Y-%m-%d")

    if IBAN_MARK in message:
        return "dialog", {
            "action": "reject_iban",
            "allowed": False,
            "reason": "Tam IBAN alınmaz. Kayıtlı alıcı kodu kullanılır.",
            "redacted": IBAN_MARK,
        }

    if draft.stage == "confirm" and CANCEL.search(message):
        draft.clear()
        return "dialog", {"action": "cancel_eft"}

    level = confirmation_level(message)
    if draft.stage == "confirm" and level:
        amount = None if not draft.prepared else draft.prepared.get("amount_tl")
        if amount is not None and amount > STRONG_CONFIRM_OVER_TL and level != "strong":
            return "dialog", {
                "action": "need_strong_confirm",
                "allowed": False,
                "amount_tl": amount,
                "reason": "5.000 TL üzeri için onaylıyorum gerekir",
            }
        if not draft.prepared or not draft.prepared.get("allowed"):
            draft.clear()
            return "dialog", {"action": "reject_eft", "allowed": False}
        created = submit_eft(transfers, accounts, customer_id, draft.prepared, now)
        draft.clear()
        return "dialog", created

    if card_draft.stage == "confirm" and not wants_new_eft(message):
        if CANCEL.search(message):
            card_draft.clear()
            return "dialog", {"action": "cancel_card_block"}
        if level:
            if not card_draft.prepared or not card_draft.prepared.get("allowed"):
                card_draft.clear()
                return "dialog", {"action": "reject_card_block", "allowed": False}
            closed = submit_card_block(cards, customer_id)
            card_draft.clear()
            return "dialog", closed

    amount = parse_amount(message)
    payee_id = parse_payee(message)
    starts = wants_new_eft(message)
    if starts or (draft.active and (amount is not None or payee_id is not None)):
        card_draft.clear()
        draft.active = True
        if amount is not None:
            draft.amount = amount
        if payee_id is not None:
            draft.payee_id = payee_id
        if draft.amount is None or draft.payee_id is None:
            draft.stage = "collect"
            balance = lookup_balance(accounts, customer_id)
            return "dialog", {
                "action": "collect_eft",
                "amount_tl": draft.amount,
                "payee_id": draft.payee_id,
                "available_tl": balance.get("available_tl"),
                "payees": _own_payees(payees, customer_id),
            }
        prepared = prepare_eft(
            accounts, customer_id, payees, draft.amount, draft.payee_id, transfers, now
        )
        draft.prepared = prepared
        draft.stage = "confirm" if prepared["allowed"] else "collect"
        return "dialog", prepared

    if not draft.active and CARD_CLOSE.search(message):
        prepared = prepare_card_block(cards, customer_id)
        if prepared.get("allowed"):
            card_draft.stage = "confirm"
            card_draft.prepared = prepared
        else:
            card_draft.clear()
        return "dialog", prepared

    if wants_transfer_clarification(message):
        return "dialog", {
            "action": "clarify_transfer",
            "ask_which": True,
            "transfers": [
                lookup_transfer(transfers, row_id, customer_id, today)
                for row_id in owned_transfer_ids(transfers, customer_id)
            ],
        }

    if wants_transfer_list(message):
        return "tool", {
            "action": "list_transfers",
            "transfers": [
                lookup_transfer(transfers, row_id, customer_id, today)
                for row_id in owned_transfer_ids(transfers, customer_id)
            ],
        }

    if IDENTITY.search(message) and find_transfer_id(message) is None:
        return "dialog", {
            "action": "reject_identity",
            "customer_id": customer_id,
            "reason": "oturum müşterisi değişmez",
        }

    return "", None
