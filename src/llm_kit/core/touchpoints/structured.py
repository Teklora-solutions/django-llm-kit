"""Getting a typed object out of a model reply that may wrap JSON in prose or code fences."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (fenced or bare)."""
    for candidate in [m.group(1) for m in _FENCE.finditer(text)] + [text]:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except ValueError:
            pass
        for open_, close in (("{", "}"), ("[", "]")):
            start, end = candidate.find(open_), candidate.rfind(close)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except ValueError:
                    continue
    raise ValueError("no JSON found in model reply")


def parse_as(text: str, schema: type[BaseModel] | None) -> Any:
    data = extract_json(text)
    return schema.model_validate(data) if schema is not None else data
