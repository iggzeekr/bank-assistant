import json
import re
from datetime import datetime
from pathlib import Path

TRANSFER_ID = re.compile(r"\bT\d{4}\b", re.IGNORECASE)
BALANCE = re.compile(r"\bbakiye\w*", re.IGNORECASE)

DAILY_EFT_LIMIT_TL = 100_000
STRONG_CONFIRM_OVER_TL = 5_000


def load_transfers(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"].upper(): row for row in rows}


def load_accounts(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_payees(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"].upper(): row for row in rows}


def load_cards(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_transfer_id(message: str) -> str | None:
    match = TRANSFER_ID.search(message)
    if match is None:
        return None
    return match.group(0).upper()


def asks_balance(message: str) -> bool:
    return BALANCE.search(message) is not None


def settlement_for(now: datetime) -> str:
    """Weekdays 08:30 through 17:00 settle the same day. Later and weekends do not."""
    if now.weekday() >= 5:
        return "sonraki iş günü"
    opened = now.replace(hour=8, minute=30, second=0, microsecond=0)
    closed = now.replace(hour=17, minute=0, second=0, microsecond=0)
    if opened <= now <= closed:
        return "aynı gün"
    return "sonraki iş günü"


def _on_this_day(row: dict, day: str) -> bool:
    value = row.get("date")
    return value == "today" or value == day


def spent_today(transfers: dict[str, dict], customer_id: str, day: str) -> int:
    total = 0
    for row in transfers.values():
        if row.get("customer_id") != customer_id or row.get("type") != "EFT":
            continue
        if _on_this_day(row, day):
            total += int(row["amount_tl"])
    return total


def _shown_date(value: str, today: str | None) -> str:
    if value == "today" and today:
        return today
    return value


def lookup_transfer(
    transfers: dict[str, dict],
    transfer_id: str,
    customer_id: str,
    today: str | None = None,
) -> dict:
    row = transfers.get(transfer_id.upper())
    if row is None:
        return {"action": "lookup_transfer", "id": transfer_id.upper(), "found": False}
    if row.get("customer_id") != customer_id:
        return {
            "action": "lookup_transfer",
            "id": transfer_id.upper(),
            "found": False,
            "allowed": False,
            "reason": "bu kayıt sana ait değil",
        }
    return {
        "action": "lookup_transfer",
        "id": row["id"],
        "type": row["type"],
        "amount_tl": row["amount_tl"],
        "status": row["status"],
        "date": _shown_date(row["date"], today),
        "recipient_account_masked": row["recipient_account_masked"],
        "found": True,
    }


def owned_transfer_ids(transfers: dict[str, dict], customer_id: str) -> list[str]:
    return sorted(row_id for row_id, row in transfers.items() if row.get("customer_id") == customer_id)


def lookup_balance(accounts: dict[str, dict], customer_id: str) -> dict:
    row = accounts.get(customer_id)
    if row is None:
        return {"action": "lookup_balance", "customer_id": customer_id, "found": False}
    return {
        "action": "lookup_balance",
        "customer_id": customer_id,
        "account_masked": row["account_masked"],
        "available_tl": row["available_tl"],
        "currency": row["currency"],
        "found": True,
    }


def prepare_eft(
    accounts: dict,
    customer_id: str,
    payees: dict,
    amount: int,
    payee_id: str,
    transfers: dict,
    now: datetime,
) -> dict:
    balance = lookup_balance(accounts, customer_id)
    payee = payees.get(payee_id.upper())
    if payee is None or payee.get("customer_id") != customer_id:
        payee = None
    fee = 15 if amount < 5000 else 0
    day = now.strftime("%Y-%m-%d")
    spent = spent_today(transfers, customer_id, day)
    reasons = []
    if payee is None:
        reasons.append("kayıtlı alıcı yok")
    if spent + amount > DAILY_EFT_LIMIT_TL:
        reasons.append("günlük limit aşılıyor")
    available = balance.get("available_tl")
    if available is not None and amount + fee > available:
        reasons.append("bakiye yetersiz")
    return {
        "action": "prepare_eft",
        "amount_tl": amount,
        "fee_tl": fee,
        "payee_id": payee_id.upper(),
        "payee_label": None if payee is None else payee["label"],
        "account_masked": None if payee is None else payee["account_masked"],
        "available_tl": available,
        "spent_today_tl": spent,
        "daily_limit_tl": DAILY_EFT_LIMIT_TL,
        "settlement": settlement_for(now),
        "confirm_with": "onaylıyorum" if amount > STRONG_CONFIRM_OVER_TL else "evet",
        "allowed": not reasons,
        "reasons": reasons,
    }


def submit_eft(
    transfers: dict[str, dict],
    accounts: dict,
    customer_id: str,
    prepared: dict,
    now: datetime,
) -> dict:
    number = 2048
    for transfer_id in transfers:
        suffix = int(transfer_id[1:])
        if suffix >= number:
            number = suffix + 1
    transfer_id = f"T{number}"
    account = accounts.get(customer_id)
    available = None
    if account is not None:
        remaining = account["available_tl"] - prepared["amount_tl"] - prepared["fee_tl"]
        account["available_tl"] = float(f"{remaining:.2f}")
        available = account["available_tl"]
    row = {
        "action": "submit_eft",
        "id": transfer_id,
        "customer_id": customer_id,
        "type": "EFT",
        "amount_tl": prepared["amount_tl"],
        "fee_tl": prepared["fee_tl"],
        "status": "onay bekliyor",
        "date": now.strftime("%Y-%m-%d"),
        "settlement": prepared.get("settlement", settlement_for(now)),
        "recipient_account_masked": prepared["account_masked"],
        "available_tl": available,
        "found": True,
    }
    transfers[transfer_id] = {key: value for key, value in row.items() if key != "action"}
    return row


def prepare_card_block(cards: dict, customer_id: str) -> dict:
    row = cards.get(customer_id)
    if row is None:
        return {"action": "prepare_card_block", "allowed": False, "reason": "kart yok"}
    if row.get("status") == "kapalı":
        return {
            "action": "prepare_card_block",
            "allowed": False,
            "reason": "kart zaten kapalı",
            "card_masked": row["card_masked"],
        }
    return {
        "action": "prepare_card_block",
        "allowed": True,
        "card_masked": row["card_masked"],
        "status": row["status"],
    }


def submit_card_block(cards: dict, customer_id: str) -> dict:
    row = cards[customer_id]
    row["status"] = "kapalı"
    return {
        "action": "submit_card_block",
        "status": "kapalı",
        "card_masked": row["card_masked"],
    }
