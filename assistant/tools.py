import json
import re
from pathlib import Path

TRANSFER_ID = re.compile(r"\bT\d{4}\b", re.IGNORECASE)
BALANCE = re.compile(r"\bbakiye\w*", re.IGNORECASE)


def load_transfers(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"].upper(): row for row in rows}


def load_accounts(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_transfer_id(message: str) -> str | None:
    match = TRANSFER_ID.search(message)
    if match is None:
        return None
    return match.group(0).upper()


def asks_balance(message: str) -> bool:
    return BALANCE.search(message) is not None


def lookup_transfer(transfers: dict[str, dict], transfer_id: str) -> dict:
    row = transfers.get(transfer_id.upper())
    if row is None:
        return {"id": transfer_id.upper(), "found": False}
    return {
        "id": row["id"],
        "type": row["type"],
        "amount_tl": row["amount_tl"],
        "status": row["status"],
        "date": row["date"],
        "recipient_account_masked": row["recipient_account_masked"],
        "found": True,
    }


def lookup_balance(accounts: dict[str, dict], customer_id: str) -> dict:
    row = accounts.get(customer_id)
    if row is None:
        return {"customer_id": customer_id, "found": False}
    return {
        "customer_id": customer_id,
        "account_masked": row["account_masked"],
        "available_tl": row["available_tl"],
        "currency": row["currency"],
        "found": True,
    }
