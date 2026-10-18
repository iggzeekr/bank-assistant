# Bank Assistant

A local retail-banking support assistant. It answers from bank policy notes, then calls one banking tool when the customer asks for a transfer or a balance. Sample customers and transfers are fictional and stay on this machine.

What it does:

- The prompt keeps the customer segment and the last six turns, so a follow-up such as "peki ücreti ne kadar" stays on the EFT question.
- EFT, card, and privacy questions are answered from `knowledge/`. Search is TF-IDF over those files. The model sees the matching note, not the whole folder.
- A transfer reference such as `T2048` calls `lookup_transfer`. A balance question calls `lookup_balance`. "transferlerim" lists recent transfers without a reference.
- An EFT request opens a draft. The assistant asks for the missing amount and saved payee, checks the balance and the daily limit, and records the transfer only after the customer says yes.
- A short greeting with no document overlap skips search and calls `LLM_MODEL_SMALL`.
- Card-like numbers and full IBANs are removed before any text is sent. Tool results contain a masked account.

## Run

```powershell
cd C:\Users\Lenovo\Projects\desk-assistant
copy .env.example .env
```

Put a real key in `.env`. Any OpenAI-compatible endpoint works. On Groq's free tier:

```text
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your-key
LLM_MODEL=llama-3.3-70b-versatile
LLM_MODEL_SMALL=llama-3.1-8b-instant
```

```powershell
python -m unittest discover -s tests -v
python main.py --dry
python measure.py
python main.py
```

`--dry` prints the chosen path without calling a model. `measure.py` asks six banking questions in one conversation and prints latency and token counts. Empty input exits. `/reset` clears the conversation.

## Paths

| Message | Path | What is sent |
|---|---|---|
| `merhaba` | `small` | Customer profile and recent turns. No documents. |
| `EFT saat kaça kadar yapılabilir` | `rag` | The EFT note. |
| `peki ücreti ne kadar` after the EFT question | `rag` | The previous user turn is added to the search text. |
| `kartımı kaybettim ne yapmalıyım` | `rag` | The card note. |
| `T2048 transferim gerçekleşti mi` | `tool` | Transfer amount, status, date, and masked recipient account. |
| `bakiyem ne kadar` | `tool` | Masked account and available balance for customer `M1001`. |
| `EFT yapmak istiyorum` | `dialog` | Asks for the amount and a saved payee (`K1`, `K2`). |
| `2500 K1` | `dialog` | Checks balance, fee, and the 100.000 TL daily limit, then waits for confirmation. |
| `evet` | `dialog` | Records the EFT only after that confirmation. |

## Scope

This is a local flow over three policy notes, two sample transfers, two saved payees, and one sample account. It is not connected to a core banking system. The two model names save cost only when `LLM_MODEL_SMALL` is a smaller model than `LLM_MODEL`.
