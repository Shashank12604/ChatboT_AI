from __future__ import annotations
import re
from typing import Literal

Intent = Literal["general", "nec", "wattmonk"]

NEC_KEYWORDS = [
    r"\\bnec\\b",
    r"national\\s+electrical\\s+code",
    r"nfpa\\s*70",
    r"code\\s*(article|section|table|c)\\b",
]

WATTMONK_KEYWORDS = [
    r"\\bwattmonk\\b",
    r"permit|plan\\s*set|turnaround|sla|pricing|cad|as\\s*built",
]


def classify_intent(query: str) -> Intent:
    q = query.lower()
    for pat in NEC_KEYWORDS:
        if re.search(pat, q):
            return "nec"
    for pat in WATTMONK_KEYWORDS:
        if re.search(pat, q):
            return "wattmonk"
    return "general"