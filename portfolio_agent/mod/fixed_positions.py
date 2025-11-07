from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

_LOCK_COLS = ("lock", "locked", "fixed", "freeze", "keep")
_WEIGHT_COLS = ("weight", "target_weight", "allocation", "pct", "percentage")


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, (float, np.floating)):
        return not np.isclose(value, 0.0)
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "keep", "locked"}


def _coerce_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _initial_weights_series(policy: Dict) -> pd.Series:
    port = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
    init = port.get("initial_weights") if isinstance(port, dict) else None
    if isinstance(init, dict):
        return pd.Series(init, dtype=float)
    return pd.Series(dtype=float)


def _canonical_ticker_map(universe: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for ticker in universe:
        if ticker is None:
            continue
        key = str(ticker).strip()
        if not key:
            continue
        mapping[key.upper()] = key
    return mapping


def load_fixed_positions(policy: Dict, universe: Iterable[str]) -> Tuple[pd.Series, Dict[str, Dict[str, object]]]:
    """
    Parse fixed (locked) positions from policy configuration.

    Returns
    -------
    tuple
        pd.Series of weights indexed by ticker, plus metadata dictionary keyed by ticker.
        The weights represent the share of total portfolio leverage reserved for locked assets.
    """

    universe = [t for t in universe if t]
    canon = _canonical_ticker_map(universe)
    initial_series = _initial_weights_series(policy)

    port = policy.get("portfolio", {}) if isinstance(policy.get("portfolio"), dict) else policy
    opt_cfg = port.get("optimization") if isinstance(port, dict) else None
    opt_cfg = opt_cfg if isinstance(opt_cfg, dict) else {}

    fixed_file = opt_cfg.get("fixed_positions_file") or port.get("fixed_positions_file") if isinstance(port, dict) else None
    inline_cfg = opt_cfg.get("fixed_positions")

    locked_meta: Dict[str, Dict[str, object]] = {}
    weight_map: Dict[str, float] = {}

    def _register(ticker_raw: str, weight: float | None, meta: Dict[str, object]) -> None:
        ticker_key = str(ticker_raw).strip()
        if not ticker_key:
            return
        canon_key = ticker_key.upper()
        if canon_key not in canon:
            raise ValueError(f"Fixed position ticker '{ticker_raw}' not found in the available universe.")
        ticker = canon[canon_key]
        locked_meta.setdefault(ticker, {}).update(meta)
        if weight is not None:
            weight_map[ticker] = float(weight)

    # File-based configuration
    if isinstance(fixed_file, str) and fixed_file.strip():
        path = os.path.expanduser(fixed_file.strip())
        if not os.path.exists(path):
            raise FileNotFoundError(f"fixed_positions_file not found: {path}")
        df = pd.read_csv(path)
        if df.empty:
            pass
        else:
            df_columns = [str(c).strip().lower() for c in df.columns]
            df.columns = df_columns
            if "ticker" not in df.columns:
                raise ValueError(f"Fixed positions file '{path}' must include a 'ticker' column.")
            for _, row in df.iterrows():
                ticker_raw = row.get("ticker")
                if ticker_raw is None:
                    continue
                is_locked = True
                for lock_col in _LOCK_COLS:
                    if lock_col in row.index:
                        is_locked = _truthy(row[lock_col])
                        break
                if not is_locked:
                    continue
                weight_val = None
                for weight_col in _WEIGHT_COLS:
                    if weight_col in row.index:
                        weight_val = _coerce_float(row[weight_col])
                        if weight_val is not None:
                            break
                meta = {col: row[col] for col in row.index if col != "ticker"}
                _register(str(ticker_raw), weight_val, meta)

    # Inline configuration (list/dict)
    if inline_cfg:
        if isinstance(inline_cfg, dict):
            items = inline_cfg.items()
        elif isinstance(inline_cfg, (list, tuple)):
            items = enumerate(inline_cfg)
        else:
            items = []
        for key, val in items:
            if isinstance(val, str):
                _register(val, None, {"source": "inline"})
            elif isinstance(val, dict):
                ticker = val.get("ticker", key if isinstance(key, str) else None)
                if not ticker:
                    continue
                lock_flag = _truthy(val.get("lock", True))
                if not lock_flag:
                    continue
                weight_val = _coerce_float(val.get("weight"))
                meta = {k: v for k, v in val.items() if k != "weight"}
                meta.setdefault("source", "inline")
                _register(str(ticker), weight_val, meta)
            elif isinstance(key, str):
                weight_val = _coerce_float(val)
                _register(key, weight_val, {"source": "inline"})

    if not locked_meta:
        return pd.Series(dtype=float), {}

    weights = {}
    fallback_candidates = []
    for ticker in locked_meta:
        if ticker in weight_map and weight_map[ticker] is not None:
            weights[ticker] = float(weight_map[ticker])
        else:
            fallback_candidates.append(ticker)

    if fallback_candidates:
        init_weights = initial_series.reindex(fallback_candidates).dropna()
        assigned = set()
        for ticker, value in init_weights.items():
            weights[ticker] = float(value)
            locked_meta[ticker]["auto_weight"] = "initial_weights"
            assigned.add(ticker)

        remaining = [t for t in fallback_candidates if t not in assigned]
        if remaining:
            default_share = 1.0 / max(len(universe), 1) if universe else 0.0
            for ticker in remaining:
                weights[ticker] = float(default_share)
                locked_meta[ticker]["auto_weight"] = "uniform_fallback"

    weights_series = pd.Series(weights, dtype=float)
    weights_series = weights_series.reindex([canon[k] for k in canon if canon[k] in weights_series.index]).dropna()
    return weights_series, locked_meta
