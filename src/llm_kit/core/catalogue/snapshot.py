"""Bundled offline snapshot of parsed source data.

Rebuild before a release with::

    python -m llm_kit.core.catalogue.snapshot
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from llm_kit.core.catalogue.sources import DEFAULT_SOURCES, ParseResult

SNAPSHOT_PATH = Path(__file__).parent / "data" / "snapshot.json.gz"


def build(path: Path = SNAPSHOT_PATH, fetch=None) -> dict[str, int]:
    from llm_kit.core.catalogue.refresh import urllib_fetcher

    fetch = fetch or urllib_fetcher
    out = {"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "sources": {}}
    counts = {}
    for src in DEFAULT_SOURCES:
        resp = fetch(src.url, {"Accept-Encoding": "gzip", "User-Agent": "django-llm-kit (snapshot)"})
        parsed = src.parse(json.loads(resp.body))
        out["sources"][src.name] = {
            "models": [[p, m, partial] for (p, m), partial in sorted(parsed.models.items())],
            "providers": parsed.providers,
        }
        counts[src.name] = len(parsed.models)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(out, f, separators=(",", ":"))
    return counts


def built_at(path: Path = SNAPSHOT_PATH) -> str | None:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f).get("built_at")


def load(path: Path = SNAPSHOT_PATH) -> dict[str, ParseResult]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for name, payload in data["sources"].items():
        pr = ParseResult(providers=payload.get("providers", {}))
        for provider, model_id, partial in payload["models"]:
            pr.models[(provider, model_id)] = partial
        result[name] = pr
    return result


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPSHOT_PATH
    for name, n in build(target).items():
        print(f"{name}: {n} models")
    print(f"wrote {target} ({target.stat().st_size // 1024} KiB)")
