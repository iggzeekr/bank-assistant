import re

from assistant.tools import find_transfer_id, lookup_balance, lookup_transfer, prepare_eft, submit_eft

START = re.compile(
    r"\b(?:eft|havale)\b.{0,40}\b(?:yapmak|göndermek|atmak|talimat)\b|\bpara göndermek\b",
    re.IGNORECASE,
)
LIST = re.compile(r"\btransferlerim\b|\bson transfer", re.IGNORECASE)
AMBIGUOUS = re.compile(r"\b(?:transferim|havalem)\b", re.IGNORECASE)
PAYEE = re.compile(r"\bK\d\b", re.IGNORECASE)
AMOUNT = re.compile(r"\b(\d{1,3}(?:\.\d{3})+|\d{2,7})\b")
CONFIRM = re.compile(r"\b(?:evet|onaylıyorum|onayla)\b", re.IGNORECASE)
CANCEL = re.compile(r"\b(?:vazgeç|iptal|hayır)\b", re.IGNORECASE)
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


def plan_turn(message, draft, accounts, customer_id, payees, transfers) -> tuple[str, dict | None]:
    """Return the path and the tool payload for this turn.

    An open EFT draft keeps slot-filling on the dialog path. A finished
    draft is submitted only after an explicit confirmation.
    """
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

    if draft.stage == "confirm" and CONFIRM.search(message):
        if not draft.prepared or not draft.prepared.get("allowed"):
            draft.clear()
            return "dialog", {"action": "reject_eft", "allowed": False}
        created = submit_eft(transfers, draft.prepared)
        draft.clear()
        return "dialog", {"action": "submit_eft", "transfer": created}

    amount = parse_amount(message)
    payee_id = parse_payee(message)
    starts = wants_new_eft(message)
    if starts or (draft.active and (amount is not None or payee_id is not None)):
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
                "payees": [
                    {"id": row["id"], "label": row["label"], "account_masked": row["account_masked"]}
                    for row in payees.values()
                ],
            }
        prepared = prepare_eft(accounts, customer_id, payees, draft.amount, draft.payee_id)
        draft.prepared = prepared
        draft.stage = "confirm" if prepared["allowed"] else "collect"
        return "dialog", prepared

    if wants_transfer_clarification(message):
        return "dialog", {
            "action": "clarify_transfer",
            "ask_which": True,
            "transfers": [lookup_transfer(transfers, row_id) for row_id in sorted(transfers)],
        }

    if wants_transfer_list(message):
        return "tool", {
            "action": "list_transfers",
            "transfers": [lookup_transfer(transfers, row_id) for row_id in transfers],
        }

    return "", None
