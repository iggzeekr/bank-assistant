import argparse
import json
import os
from pathlib import Path

from assistant.llm import from_env
from assistant.memory import Memory
from assistant.pipeline import Assistant
from assistant.retrieve import Index
from assistant.tools import load_accounts, load_transfers

ROOT = Path(__file__).resolve().parent


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build(live: bool) -> Assistant:
    load_env(ROOT / ".env")
    profile = json.loads((ROOT / "data" / "profile.json").read_text(encoding="utf-8"))
    index = Index.from_dir(ROOT / "knowledge")
    transfers = load_transfers(ROOT / "data" / "transfers.json")
    accounts = load_accounts(ROOT / "data" / "accounts.json")
    llm = from_env() if live else None
    return Assistant(
        index=index,
        transfers=transfers,
        accounts=accounts,
        memory=Memory(profile),
        llm=llm,
        large_model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        small_model=os.environ.get("LLM_MODEL_SMALL", "gpt-4o-mini"),
    )


def print_answer(result) -> None:
    sources = ", ".join(result.sources) if result.sources else "-"
    print(
        f"[{result.path} | {result.model} | {result.latency_ms} ms"
        f" | kaynak: {sources} | token: {result.prompt_tokens}+{result.completion_tokens}]"
    )
    print(result.text)
    print()


def chat(live: bool) -> None:
    assistant = build(live)
    print("Bank Assistant. Çıkmak için boş satır, sıfırlamak için /reset.")
    while True:
        try:
            message = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not message:
            return
        if message == "/reset":
            assistant.memory.reset()
            print("Konuşma silindi.\n")
            continue
        print_answer(assistant.ask(message, live=live))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bireysel bankacılık destek asistanı")
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Model çağrısı yapmadan yolu, belgeyi ve araç sonucunu göster",
    )
    args = parser.parse_args()
    chat(live=not args.dry)


if __name__ == "__main__":
    main()
