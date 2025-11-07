# mod/universe.py
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple

_DEFAULT_UNIVERSE_PATHS: Tuple[str, ...] = (
    os.path.join("data", "universe", "universe.txt"),
    os.path.join("portfolio_agent", "universe.txt"),
)


def _load_universe_from_file(path: str) -> List[str]:
    """Load tickers from a StockRover-style export (txt/csv) returning unique symbols."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"universe_file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    tickers: List[str] = []
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if rdr.fieldnames:
                cols_lower = [c.lower() for c in rdr.fieldnames]
                if "ticker" in cols_lower:
                    col = rdr.fieldnames[cols_lower.index("ticker")]
                else:
                    col = rdr.fieldnames[0]
                for row in rdr:
                    v = row.get(col)
                    if v:
                        tickers.append(str(v).strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                tickers.extend(parts)
    seen: set[str] = set()
    uniq: List[str] = []
    for ticker in tickers:
        if ticker and ticker not in seen:
            seen.add(ticker)
            uniq.append(ticker)
    return uniq


def resolve_universe(policy: Dict, universe_file: Optional[str] = None) -> Tuple[List[str], str]:
    """
    Resolve the investment universe. The new workflow assumes a StockRover export
    (txt/csv) is provided, so we prioritise file-based inputs and only fall back
    to legacy inline lists for backwards compatibility.
    """

    candidates: List[str] = []
    if isinstance(universe_file, str) and universe_file.strip():
        candidates.append(universe_file.strip())

    data_block = policy.get("data", {}) if isinstance(policy, dict) else {}
    inputs_block = policy.get("inputs", {}) if isinstance(policy.get("inputs"), dict) else {}

    for block in (data_block, inputs_block, policy):
        if not isinstance(block, dict):
            continue
        for key in ("stockrover_file", "universe_file", "universe_path"):
            val = block.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())

    candidates.extend(_DEFAULT_UNIVERSE_PATHS)

    for path in candidates:
        if not path:
            continue
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            tickers = _load_universe_from_file(expanded)
            if tickers:
                return tickers, f"from file={expanded}"

    # Legacy fallback: inline list inside policy for older configs.
    def _inline_universe(block: Dict) -> List[str]:
        raw = block.get("universe")
        if isinstance(raw, (list, tuple)):
            return [str(t).strip() for t in raw if str(t).strip()]
        return []

    if isinstance(data_block, dict):
        inline = _inline_universe(data_block)
        if inline:
            return inline, "from legacy data.universe list"

    if isinstance(policy, dict):
        inline = _inline_universe(policy)
        if inline:
            return inline, "from legacy policy-level universe list"

    raise RuntimeError(
        "No universe located. Provide a StockRover export via data.universe_file "
        "or inputs.stockrover_file in policy.yaml."
    )
