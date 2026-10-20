"""Sentences for the local page when no model is called.

The banking decision stays in the pipeline. This module only reads the
tool payload and the retrieved paragraph.
"""


def present(message: str, answer) -> str:
    payload = answer.payload or {}
    action = payload.get("action", "")
    if action == "no_document":
        return "Bu bilgi belgelerde yok."
    if answer.path == "small":
        return _small(message)
    if answer.path == "rag":
        return answer.evidence.strip() or "Bu bilgi belgelerde yok."
    if action == "lookup_balance":
        if not payload.get("found"):
            return "Bu oturum için hesap yok."
        return (
            f"Kullanılabilir bakiyen {_money(payload['available_tl'])}. "
            f"Hesap {payload['account_masked']}."
        )
    if action == "lookup_transfer":
        if payload.get("reason"):
            return "Bu kayıt sana ait değil."
        if not payload.get("found"):
            return f"{payload.get('id', '')} numaralı transfer yok."
        return (
            f"{payload['id']} {payload['status']}. Tutar {_money(payload['amount_tl'])}. "
            f"Tarih {payload['date']}. Alıcı hesap {payload['recipient_account_masked']}."
        )
    if action == "list_transfers":
        lines = _transfer_lines(payload.get("transfers") or [])
        return "Transferlerin:\n" + "\n".join(lines)
    if action == "clarify_transfer":
        lines = _transfer_lines(payload.get("transfers") or [])
        return "Hangi transferi soruyorsun?\n" + "\n".join(lines)
    if action == "collect_eft":
        missing = []
        if payload.get("amount_tl") is None:
            missing.append("tutar")
        if not payload.get("payee_id"):
            missing.append("kayıtlı alıcı")
        payees = ", ".join(
            f"{row['id']} {row['label']}" for row in payload.get("payees") or []
        )
        balance = payload.get("available_tl")
        balance_text = f" Kullanılabilir bakiye {_money(balance)}." if balance is not None else ""
        return f"EFT için {' ve '.join(missing)} yaz.{balance_text} Kayıtlı alıcılar: {payees}."
    if action == "prepare_eft":
        if not payload.get("allowed"):
            reasons = ", ".join(payload.get("reasons") or [])
            return f"Bu EFT kayda geçemez: {reasons}."
        fee = payload.get("fee_tl") or 0
        fee_text = "Ücret yok." if fee == 0 else f"Ücret {_money(fee)}."
        label = payload.get("payee_label") or payload.get("payee_id")
        return (
            f"{_money(payload['amount_tl'])} tutarında {label} alıcısına EFT hazır. "
            f"{fee_text} İşlem {payload.get('settlement')}. "
            f"Onay için {payload.get('confirm_with')} yaz."
        )
    if action == "need_strong_confirm":
        return "Bu tutar 5.000 TL üzerinde. Kayda geçmesi için onaylıyorum yaz."
    if action == "submit_eft":
        return (
            f"{payload['id']} kayda alındı. Durum {payload['status']}. "
            f"İşlem {payload.get('settlement')}. "
            f"Kullanılabilir bakiye {_money(payload.get('available_tl'))}."
        )
    if action == "cancel_eft":
        return "EFT taslağı iptal edildi. Kayıt açılmadı."
    if action == "reject_eft":
        return "Transfer oluşturulmadı."
    if action == "reject_iban":
        return "Tam IBAN alınmaz. Kayıtlı alıcı kodunu kullan."
    if action == "reject_identity":
        return f"Oturum müşterisi değişmez. Bu oturum {payload.get('customer_id')}."
    if action == "prepare_card_block":
        if not payload.get("allowed"):
            return payload.get("reason") or "Kart kapatılamadı."
        return f"Kart {payload.get('card_masked')} kapatılacak. Onay için evet yaz."
    if action == "submit_card_block":
        return f"Kart {payload.get('card_masked')} kullanıma kapatıldı."
    if action == "cancel_card_block":
        return "Kart kapatma iptal edildi."
    return "Bu turda gösterilecek bir sonuç yok."


def _small(message: str) -> str:
    folded = message.casefold()
    if "teşekkür" in folded or "tesekkur" in folded or "sağ ol" in folded:
        return "Rica ederim."
    if folded.strip() in {"evet", "onaylıyorum", "onayla"}:
        return "Onaylanacak bir işlem yok."
    return "Merhaba. EFT, kart, bakiye veya transfer referansı yazabilirsiniz."


def _transfer_lines(rows: list[dict]) -> list[str]:
    lines = []
    for row in rows:
        if not row.get("found"):
            continue
        lines.append(f"{row['id']}: {row['status']}, {_money(row['amount_tl'])}")
    return lines or ["Kayıtlı transfer yok."]


def _money(value) -> str:
    if value is None:
        return "-"
    number = float(value)
    sign = "-" if number < 0 else ""
    number = abs(number)
    whole, frac = f"{number:.2f}".split(".")
    groups = []
    while whole:
        groups.append(whole[-3:])
        whole = whole[:-3]
    body = ".".join(reversed(groups))
    if frac == "00":
        return f"{sign}{body} TL"
    return f"{sign}{body},{frac} TL"
