"""
Policy condition parsing and evaluation helpers.
"""
from __future__ import annotations

import re
from typing import Tuple


_TRUST_RE = re.compile(r"trust_score\s*([<>=]+)\s*([\d.]+)")


def parse_trust_condition(cond: str) -> Tuple[str, float] | None:
    m = _TRUST_RE.match(cond)
    if not m:
        return None
    op, val = m.groups()
    return op, float(val)


