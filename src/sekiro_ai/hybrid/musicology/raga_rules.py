"""
Raga grammar utilities used by training and inference.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# Pitch classes: C=0 ... B=11 (Sa depends on tonic choice).
DEFAULT_RAGA_RULES: Dict[str, Dict] = {
    "yaman": {
        "allowed_pitch_classes": [0, 2, 4, 6, 7, 9, 11],
        "vivadi_pitch_classes": [1, 3, 5, 8, 10],
        "vadi_pitch_classes": [7],
        "samvadi_pitch_classes": [2],
        "chalan_degrees": [0, 4, 7, 11, 7, 4, 2, 0]
    },
    "bhairavi": {
        "allowed_pitch_classes": [0, 1, 3, 5, 7, 8, 10],
        "vivadi_pitch_classes": [2, 4, 6, 9, 11],
        "vadi_pitch_classes": [5],
        "samvadi_pitch_classes": [0],
        "chalan_degrees": [0, 1, 3, 5, 7, 8, 10, 8, 7, 5]
    },
    "malkauns": {
        "allowed_pitch_classes": [0, 3, 5, 8, 10],
        "vivadi_pitch_classes": [1, 2, 4, 6, 7, 9, 11],
        "vadi_pitch_classes": [5],
        "samvadi_pitch_classes": [0],
        "chalan_degrees": [0, 3, 5, 8, 10, 8, 5, 3, 0]
    },
    "bhoopali": {
        "allowed_pitch_classes": [0, 2, 4, 7, 9],
        "vivadi_pitch_classes": [1, 3, 5, 6, 8, 10, 11],
        "vadi_pitch_classes": [4],
        "samvadi_pitch_classes": [0],
        "chalan_degrees": [0, 2, 4, 7, 9, 7, 4, 2, 0]
    },
    "bhoop": {
        "allowed_pitch_classes": [0, 2, 4, 7, 9],
        "vivadi_pitch_classes": [1, 3, 5, 6, 8, 10, 11],
        "vadi_pitch_classes": [4],
        "samvadi_pitch_classes": [0],
        "chalan_degrees": [0, 2, 4, 7, 9, 7, 4, 2, 0]
    },
    "darbari": {
        "allowed_pitch_classes": [0, 2, 3, 5, 7, 8, 10],
        "vivadi_pitch_classes": [1, 4, 6, 9, 11],
        "vadi_pitch_classes": [7],
        "samvadi_pitch_classes": [0],
        "chalan_degrees": [0, 2, 3, 5, 7, 8, 7, 5, 3, 2, 0]
    },
}


DEFAULT_TAALS: Dict[str, Dict] = {
    "teental": {"beats": 16},
    "trital": {"beats": 16},
    "jhaptal": {"beats": 10},
    "ektal": {"beats": 12},
    "rupak": {"beats": 7},
    "dadra": {"beats": 6},
    "deepchandi": {"beats": 14},
    "bhajani": {"beats": 8},
}


@dataclass
class RagaGrammar:
    allowed_pitch_classes: Set[int]
    vivadi_pitch_classes: Set[int]
    vadi_pitch_classes: Set[int]
    samvadi_pitch_classes: Set[int]
    chalan_degrees: List[int]


def _normalize_pitch_classes(items: Optional[List[int]]) -> Set[int]:
    if not items:
        return set()
    return {int(v) % 12 for v in items}


def get_raga_grammar(raga_name: str, metadata_entry: Optional[Dict] = None) -> RagaGrammar:
    name = (raga_name or "").lower().strip()
    from_defaults = DEFAULT_RAGA_RULES.get(name, {})
    from_meta = metadata_entry or {}

    allowed = _normalize_pitch_classes(
        from_meta.get("allowed_pitch_classes", from_defaults.get("allowed_pitch_classes", list(range(12))))
    )
    vivadi = _normalize_pitch_classes(
        from_meta.get("vivadi_pitch_classes", from_defaults.get("vivadi_pitch_classes", []))
    )
    vadi = _normalize_pitch_classes(
        from_meta.get("vadi_pitch_classes", from_defaults.get("vadi_pitch_classes", []))
    )
    samvadi = _normalize_pitch_classes(
        from_meta.get("samvadi_pitch_classes", from_defaults.get("samvadi_pitch_classes", []))
    )
    chalan = from_meta.get("chalan_degrees", from_defaults.get("chalan_degrees", []))

    return RagaGrammar(
        allowed_pitch_classes=allowed if allowed else set(range(12)),
        vivadi_pitch_classes=vivadi,
        vadi_pitch_classes=vadi,
        samvadi_pitch_classes=samvadi,
        chalan_degrees=[int(x) % 12 for x in chalan] if chalan else []
    )


def get_taal_name_and_beats(raga_name: str, metadata_entry: Optional[Dict] = None) -> Tuple[str, int]:
    entry = metadata_entry or {}
    candidate = (entry.get("taal") or "teental").lower().strip()
    if candidate in DEFAULT_TAALS:
        return candidate, int(DEFAULT_TAALS[candidate]["beats"])
    return "teental", int(DEFAULT_TAALS["teental"]["beats"])
