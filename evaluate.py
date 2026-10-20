"""Score scripted banking conversations without calling a model.

Each turn checks the chosen path. When a turn names a source or a fact,
that text must be in the note or tool result handed to the model.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from main import build

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.json"


def load_cases(path: Path = CASES) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def score(cases: list[dict], live: bool = False) -> list[dict]:
    assistant = build(live=live)
    results = []
    office_hours = lambda: datetime(2026, 10, 20, 10, 0)
    for case in cases:
        assistant.reset()
        stamp = case.get("clock")
        if stamp:
            assistant.clock = lambda stamp=stamp: datetime.fromisoformat(stamp)
        else:
            assistant.clock = office_hours
        for turn in case["turns"]:
            answer = assistant.ask(turn["user"], live=live)
            seen = answer.text if live else answer.evidence
            problems = []
            if answer.path != turn["path"]:
                problems.append(f"yol {answer.path}, beklenen {turn['path']}")
            source = turn.get("source")
            if source and source not in answer.sources:
                problems.append(f"kaynak yok: {source}")
            expected = turn.get("contains")
            if isinstance(expected, str):
                expected = [expected]
            for fact in expected or []:
                if fact not in seen and fact not in answer.evidence:
                    problems.append(f"metinde yok: {fact}")
            results.append(
                {
                    "id": case["id"],
                    "user": turn["user"],
                    "path": answer.path,
                    "ok": not problems,
                    "problems": problems,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Konuşma ölçüm seti")
    parser.add_argument("--live", action="store_true", help="Cevabı modelden de kontrol et")
    args = parser.parse_args()
    results = score(load_cases(), live=args.live)
    failed = [item for item in results if not item["ok"]]
    for item in results:
        mark = "ok" if item["ok"] else "FAIL"
        print(f"{mark:4} {item['id']:24} {item['path']:6} {item['user']}")
        for problem in item["problems"]:
            print(f"     {problem}")
    print(f"\n{len(results) - len(failed)}/{len(results)} tur geçti")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
