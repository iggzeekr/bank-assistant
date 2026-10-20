"""Print one forward pass for the demo sentence or for the words you pass."""

import sys

from assistant.layers import DEMO_SENTENCE, VOCAB, generate, run_demo


def row(values: list[float]) -> str:
    return " ".join(f"{value:7.2f}" for value in values)


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEMO_SENTENCE
    if text.casefold() == DEMO_SENTENCE.casefold():
        result = run_demo()
        vocab = VOCAB
    else:
        result = generate(text)
        vocab = result["vocab"]
    print(text)
    print("1. Token")
    print("   " + " | ".join(result["tokens"]))
    print("2. Gömme")
    for token, vector in zip(result["tokens"], result["embedded"]):
        print(f"   {token:8} {row(vector)}")
    print("3. Katman normalizasyonu")
    for token, vector in zip(result["tokens"], result["normalized"]):
        print(f"   {token:8} {row(vector)}")
    print("4. Dikkat  (satır, kendisinden sonraki tokena bakamaz)")
    header = " ".join(f"{token:>8}" for token in result["tokens"])
    print(f"   {'':8} {header}")
    for token, weights in zip(result["tokens"], result["attention"]):
        print(f"   {token:8} {row(weights)}")
    print("5. Ara katman sonrası")
    for token, vector in zip(result["tokens"], result["after_mlp"]):
        print(f"   {token:8} {row(vector)}")
    print("6. Son token için logit")
    for token, value in zip(vocab, result["logits"]):
        print(f"   {token:8} {value:7.2f}")
    print("7. Olasılık  sıcaklık 1.0 ve 0.2")
    for token, warm, cool in zip(vocab, result["warm"], result["cool"]):
        print(f"   {token:8} {warm:7.2f}  {cool:7.2f}")
    print(f"8. Katmanların seçtiği kelime: {result['choice']}")


if __name__ == "__main__":
    main()
