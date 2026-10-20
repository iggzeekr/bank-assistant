"""One transformer block, in the order drawn at https://bbycroft.net/llm.

The weights are fixed. This forward pass does not answer a customer and
it is not the model behind the chat page.
"""

import math

DEMO_SENTENCE = "EFT ücreti"
VOCAB = ["EFT", "ücreti", "15", "TL"]
ANSWER_VOCAB = ["EFT", "ücreti", "15", "TL", "bakiye", "yok"]
WIDTH = 2
GENERATE_WIDTH = 8


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    lifted = [math.exp(value - peak) for value in values]
    total = sum(lifted)
    return [item / total for item in lifted]


def probabilities(logits: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("Sıcaklık sıfırdan büyük olmalı.")
    return softmax([value / temperature for value in logits])


def layer_norm(values: list[float], epsilon: float = 1e-5) -> list[float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance + epsilon)
    return [(value - mean) / scale for value in values]


def gelu(value: float) -> float:
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def add(left: list[float], right: list[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def linear(values: list[float], weight: list[list[float]], bias: list[float]) -> list[float]:
    return [dot(row, values) + offset for row, offset in zip(weight, bias)]


def tokenize(text: str, vocab: list[str]) -> list[int]:
    index = {token.casefold(): position for position, token in enumerate(vocab)}
    ids = []
    for word in text.split():
        if word.casefold() not in index:
            raise KeyError(word)
        ids.append(index[word.casefold()])
    return ids


def attention(states: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """One causal head. Queries, keys, and values are the states themselves."""
    scale = math.sqrt(len(states[0]))
    weights = []
    mixed = []
    for index, query in enumerate(states):
        scores = []
        for key_index, key in enumerate(states):
            if key_index > index:
                scores.append(float("-inf"))
            else:
                scores.append(dot(query, key) / scale)
        row = softmax(scores)
        weights.append(row)
        mixed.append(
            [
                sum(row[source] * states[source][dim] for source in range(len(states)))
                for dim in range(len(query))
            ]
        )
    return mixed, weights


def mlp(values: list[float]) -> list[float]:
    """Expand, apply GELU, and project back to the same width."""
    hidden = values + values
    activated = [gelu(value) for value in hidden]
    half = len(values)
    return [0.1 * (activated[index] + activated[index + half]) for index in range(half)]


def _hash_embed(token: str, width: int) -> list[float]:
    values = [0.0] * width
    for index, char in enumerate(token.casefold()):
        values[(ord(char) + index) % width] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def _words(text: str, limit: int) -> list[str]:
    words = text.split() or ["<bos>"]
    if len(words) > limit:
        words = words[-limit:]
    return words


def _block(embedded: list[list[float]]) -> dict:
    normalized = [layer_norm(row) for row in embedded]
    attended, weights = attention(normalized)
    after_attention = [add(left, right) for left, right in zip(normalized, attended)]
    after_mlp = [add(row, mlp(layer_norm(row))) for row in after_attention]
    return {
        "normalized": normalized,
        "attention": weights,
        "after_attention": after_attention,
        "after_mlp": after_mlp,
    }


def encode_text(text: str, width: int = 8) -> dict:
    """Run one block over a prompt. The caller keeps the trace private."""
    words = _words(text, 500)
    embedded = [_hash_embed(word, width) for word in words]
    traced = _block(embedded)
    return {"n_tokens": len(words), "final": traced["after_mlp"][-1]}


def generate(text: str, width: int = GENERATE_WIDTH) -> dict:
    """Run the same block on any sentence and pick the next word from logits."""
    words = _words(text, 40)
    embedded = [_hash_embed(word, width) for word in words]
    traced = _block(embedded)
    final = layer_norm(traced["after_mlp"][-1])
    rows = [_hash_embed(token, width) for token in ANSWER_VOCAB]
    logits = [dot(final, row) for row in rows]
    return {
        "tokens": words,
        "embedded": embedded,
        "normalized": traced["normalized"],
        "attention": traced["attention"],
        "after_attention": traced["after_attention"],
        "after_mlp": traced["after_mlp"],
        "logits": logits,
        "warm": probabilities(logits, 1.0),
        "cool": probabilities(logits, 0.2),
        "choice": ANSWER_VOCAB[max(range(len(logits)), key=logits.__getitem__)],
        "vocab": list(ANSWER_VOCAB),
    }


def forward(token_ids: list[int], vocab: list[str]) -> dict:
    token_table = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ]
    embedded = [list(token_table[token_id]) for token_id in token_ids]
    normalized = [layer_norm(row) for row in embedded]
    attended, weights = attention(normalized)
    after_attention = [add(left, right) for left, right in zip(normalized, attended)]
    after_mlp = [add(row, mlp(layer_norm(row))) for row in after_attention]
    final = [layer_norm(row) for row in after_mlp]
    unembed = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ]
    logits = [linear(row, unembed, [0.0, 0.0, 0.0, 0.0]) for row in final]
    last = logits[-1]
    return {
        "tokens": [vocab[token_id] for token_id in token_ids],
        "embedded": embedded,
        "normalized": normalized,
        "attention": weights,
        "after_attention": after_attention,
        "after_mlp": after_mlp,
        "logits": last,
        "warm": probabilities(last, 1.0),
        "cool": probabilities(last, 0.2),
        "choice": vocab[max(range(len(last)), key=last.__getitem__)],
    }


def run_demo() -> dict:
    return forward(tokenize(DEMO_SENTENCE, VOCAB), VOCAB)
