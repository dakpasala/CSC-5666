"""Versioned activity-code -> category mapping (longest-prefix rules)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class CategoryMapping:
    version: str
    categories: list[str]
    rules: dict[str, str]
    unknown: str

    @classmethod
    def load(cls, path: str | Path) -> "CategoryMapping":
        spec = yaml.safe_load(Path(path).read_text())
        # YAML keys like 01 must be quoted; coerce defensively and reject ambiguity.
        rules = {str(k): v for k, v in spec["rules"].items()}
        cats = list(spec["categories"])
        bad = sorted({v for v in rules.values() if v not in cats})
        if bad:
            raise ValueError(f"rules reference undefined categories: {bad}")
        unknown = spec.get("unknown_category", "unknown")
        if unknown not in cats:
            raise ValueError("unknown category must be listed in categories")
        return cls(version=spec["version"], categories=cats, rules=rules, unknown=unknown)

    @property
    def index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.categories)}

    def map_code(self, code: str) -> str:
        code = str(code).zfill(6)
        for n in range(len(code), 0, -1):
            hit = self.rules.get(code[:n])
            if hit is not None:
                return hit
        return self.unknown

    def map_codes(self, codes: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Map a series of codes. Returns (category, matched_rule_flag)."""
        uniq = codes.astype(str).str.zfill(6).unique()
        lookup = {c: self.map_code(c) for c in uniq}
        matched = {c: any(c[:n] in self.rules for n in range(1, 7)) for c in uniq}
        z = codes.astype(str).str.zfill(6)
        return z.map(lookup), z.map(matched)
