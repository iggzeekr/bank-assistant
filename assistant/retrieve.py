import math
import re
import urllib.error
from dataclasses import dataclass
from pathlib import Path

TOKEN = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+")
STOPWORDS = {
    "ve", "bir", "bu", "için", "ne", "mi", "mı", "mu", "mü",
    "nasıl", "kaç", "ile", "da", "de", "ki", "olan", "olarak",
    "the", "a", "an", "of", "to",
}


def normalize(text: str) -> str:
    return text.replace("İ", "i").replace("I", "ı").casefold()


SUFFIXES = (
    "larımız", "lerimiz",
    "ımız", "imiz", "umuz", "ümüz",
    "ımı", "imi", "umu", "ümü",
    "ları", "leri", "lar", "ler",
    "ım", "im", "um", "üm",
    "ın", "in", "un", "ün",
    "nın", "nin", "nun", "nün",
)


def stem(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def paragraphs(text: str) -> list[str]:
    """Split a note on blank lines. A heading alone is not a chunk."""
    chunks = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
        if lines:
            chunks.append(" ".join(lines))
    return chunks


def tokenize(text: str) -> list[str]:
    words = []
    for match in TOKEN.findall(text):
        word = stem(normalize(match))
        if len(word) < 2 or word in STOPWORDS:
            continue
        words.append(word)
    return words


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    weights: dict[str, float]


class Index:
    def __init__(self, chunks: list[Chunk], embedder=None, vectors: list[list[float]] | None = None):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = vectors
        self.method = "embedding" if vectors else "tfidf"
        self.embed_error = ""

    @classmethod
    def from_dir(cls, directory: Path, embedder=None) -> "Index":
        chunk_texts = []
        for path in sorted(directory.glob("*.md")):
            for paragraph in paragraphs(path.read_text(encoding="utf-8")):
                chunk_texts.append((path.name, paragraph))
        doc_freq: dict[str, int] = {}
        tokenized = []
        for source, paragraph in chunk_texts:
            tokens = tokenize(paragraph)
            tokenized.append((source, paragraph, tokens))
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1
        count = max(len(tokenized), 1)
        chunks = []
        for source, paragraph, tokens in tokenized:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            weights = {}
            for token, seen in counts.items():
                idf = math.log((1 + count) / (1 + doc_freq[token])) + 1
                weights[token] = seen * idf
            chunks.append(Chunk(source, paragraph, weights))
        vectors = None
        if embedder is not None:
            try:
                vectors = embedder.embed_many([chunk.text for chunk in chunks])
            except (RuntimeError, urllib.error.URLError, TimeoutError, OSError) as error:
                vectors = None
                index = cls(chunks, embedder=None, vectors=None)
                index.embed_error = str(error)
                return index
        index = cls(chunks, embedder=embedder if vectors else None, vectors=vectors)
        if embedder is None:
            index.embed_error = "EMBED_API_KEY boş"
        return index

    def search(self, query: str, k: int = 2) -> list[tuple[Chunk, float]]:
        if self.vectors is not None and self.embedder is not None:
            try:
                query_vector = self.embedder.embed(query)
                ranked = []
                for chunk, vector in zip(self.chunks, self.vectors):
                    score = _cosine_vec(query_vector, vector)
                    if score > 0:
                        ranked.append((chunk, score))
                ranked.sort(key=lambda item: item[1], reverse=True)
                self.method = "embedding"
                return ranked[:k]
            except (RuntimeError, urllib.error.URLError, TimeoutError, OSError):
                self.method = "tfidf"
        return self._search_tfidf(query, k)

    def _search_tfidf(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        self.method = "tfidf"
        query_weights: dict[str, float] = {}
        for token in tokenize(query):
            query_weights[token] = query_weights.get(token, 0) + 1
        ranked = []
        for chunk in self.chunks:
            score = _cosine(query_weights, chunk.weights)
            if score > 0:
                ranked.append((chunk, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _cosine_vec(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
