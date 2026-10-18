"""Aynı konuşmada banka soruları sorar ve yolu, süreyi, token sayısını yazar."""

import argparse

from main import build, print_answer

QUESTIONS = [
    "merhaba",
    "EFT saat kaça kadar yapılabilir",
    "peki ücreti ne kadar",
    "kartımı kaybettim ne yapmalıyım",
    "T2048 transferim gerçekleşti mi",
    "bakiyem ne kadar",
    "EFT yapmak istiyorum",
    "2500 K1",
    "evet",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    assistant = build(live=not args.dry)
    for question in QUESTIONS:
        print(f"Sen: {question}")
        print_answer(assistant.ask(question, live=not args.dry))


if __name__ == "__main__":
    main()
