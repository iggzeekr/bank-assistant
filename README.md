# Bank Assistant
<img width="666" height="688" alt="image" src="https://github.com/user-attachments/assets/366149f3-611b-4ff8-a3da-93eee2af37dc" />


A local retail-banking support assistant. It answers from bank policy notes, then calls one banking tool when the customer asks for a transfer or a balance. Sample customers and transfers are fictional and stay on this machine.


What it does:

- The prompt keeps the customer segment and the last six turns. A follow-up such as "peki ücreti ne kadar" is searched on its own. The previous turn is added only when the new message already shares a word with a note. A message with no shared word does not reuse that note.
- EFT, card, and privacy notes are split on blank lines. The model sees the matching paragraph, not the whole file. A question with no overlapping words gets "Bu bilgi belgelerde yok" and no paragraph.
- Note search can use embeddings. `text-embedding-3-small` turns the question and each paragraph into a list of 1536 numbers, so "masraf" can still find the fee paragraph in `eft.md`. The live call returned HTTP 429, so that account had no remaining quota. Search then counts overlapping words instead. Startup prints `Embedding kapalı: HTTP 429`. In that mode "masraf" does not find `eft.md`, because the file says "ücret".
- A transfer reference such as `T2048` calls `lookup_transfer` for the signed-in customer only. `T1802` belongs to `M1002` and comes back as "bu kayıt sana ait değil". Saying "ben M1002'yim" does not change the session. A balance question calls `lookup_balance`. "transferlerim" lists that customer's transfers. "transferim nerede" without a reference asks which one.
- An EFT request opens a draft. The assistant asks for the missing amount and saved payee, checks the balance and the EFT total already sent today, and records the transfer only after confirmation. Above 5.000 TL the confirmation must be "onaylıyorum". Weekday hours are 08:30–17:00; later and weekends settle on the next business day. The amount and the fee leave the balance. A second "evet" does not open another transfer.
- "kartımı kapat" waits for confirmation, then marks that customer's card closed.
- Each tool call leaves an audit line: customer, tool name, and time. The line has no card number and no full IBAN.
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
python evaluate.py
python main.py --dry
python measure.py
python main.py
python server.py
python show_layers.py
python show_layers.py bakiyem ne kadar
```

`python show_layers.py` prints one transformer block for `EFT ücreti`: embedding, layer norm, causal attention, a small GELU layer, logits, and the next-token probabilities at temperature 1.0 and 0.2. That run uses four words, and the highest logit is `TL`. Another sentence uses the same stages. `python show_layers.py bakiyem ne kadar` embeds each word, applies the causal mask, and picks one word from `EFT`, `ücreti`, `15`, `TL`, `bakiye`, and `yok`. On that sentence the highest logit is `15`. `python show_layers.py kart limitim kac` picks `bakiye`. The chat page does not print these scores. `python server.py` opens the chat at `http://127.0.0.1:8765`. It stays on this machine. Without `--live` the page runs this block over the prompt and shows the banking sentence from the note or the tool. The path is printed under each reply. `--dry` prints the chosen path without calling a model. `measure.py` asks one conversation and prints latency and token counts. `evaluate.py` scores the scripted conversations in `eval/cases.json`: path, source note, and the fact present in that paragraph or tool result. The office-hours clock in that run is Tuesday 2026-10-20 10:00, unless a case names another time. Empty input exits. `/reset` clears the conversation and the audit. Recorded transfers, the balance, and the card status stay.

## Paths

| Message | Path | What is sent |
|---|---|---|
| `merhaba` | `small` | Customer profile and recent turns. No documents. |
| `EFT saat kaça kadar yapılabilir` | `rag` | The EFT note. |
| `peki ücreti ne kadar` after the EFT question | `rag` | The fee paragraph. The hours paragraph stays out. |
| `kartımı kaybettim ne yapmalıyım` | `rag` | The lost-card paragraph. |
| `altın hesabı açılıyor mu` | `rag` | No paragraph. The answer is that the notes do not contain it. |
| `T2048 transferim gerçekleşti mi` | `tool` | That transfer's amount, status, date, and masked account. |
| `T1802 durumunu söyler misin` | `tool` | Refusal. The session customer is `M1001`. |
| `transferlerim` | `tool` | `M1001`'s transfers, with no reference required. |
| `transferim nerede` | `dialog` | Lists that customer's transfers and asks which one. |
| `bakiyem ne kadar` | `tool` | Masked account and available balance for customer `M1001`. |
| `EFT yapmak istiyorum` | `dialog` | Asks for the amount and a saved payee (`K1`, `K2`). |
| `2500 K1` | `dialog` | Checks balance, the 15 TL fee, and today's EFT total, then waits for "evet". |
| `evet` | `dialog` | Records the EFT, then subtracts the amount and the fee. |
| `6000 K1` then `evet` | `dialog` | Stays on the draft until the customer says "onaylıyorum". |
| `1800 K2` at 18:30 | `dialog` | Prepares the EFT for the next business day. |
| `kartımı kapat` then `evet` | `dialog` | Closes `M1001`'s card. |

## Scope

This is a local flow over three policy notes, two sample customers, and their transfers, payees, and cards. `M1001` has already sent 90.000 TL by EFT today, so the remaining daily room is 10.000 TL. It is not connected to a core banking system. The two model names save cost only when `LLM_MODEL_SMALL` is a smaller model than `LLM_MODEL`.
