import json
import os
import urllib.error
import urllib.request


class Embedder:
    """Turns text into a list of numbers through an OpenAI-compatible embeddings API."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            error.read()
            raise RuntimeError(f"HTTP {error.code}") from error
        rows = sorted(payload["data"], key=lambda row: row["index"])
        return [row["embedding"] for row in rows]


def from_env() -> Embedder | None:
    api_key = os.environ.get("EMBED_API_KEY", "").strip() or os.environ.get("LLM_API_KEY", "").strip()
    if not api_key:
        return None
    return Embedder(
        base_url=os.environ.get("EMBED_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        model=os.environ.get("EMBED_MODEL", "text-embedding-3-small"),
        timeout=10,
    )
