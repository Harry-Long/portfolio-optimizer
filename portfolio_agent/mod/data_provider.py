from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr


def get_prices_synthetic(
    tickers: Iterable[str],
    start: str = "2023-01-03",
    periods: int = 252 * 2,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate simple synthetic price paths for sandbox testing."""
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("Synthetic price generation requires at least one ticker.")
    np.random.seed(seed)
    dates = pd.date_range(start=start, periods=periods, freq="B")
    mu = np.array([0.10, 0.07, 0.04, 0.02])
    vol = np.array([0.22, 0.18, 0.10, 0.05])
    mu = np.resize(mu, len(tickers)) / 252.0
    vol = np.resize(vol, len(tickers)) / np.sqrt(252.0)
    prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    prices.iloc[0] = 100.0
    for idx in range(1, len(dates)):
        shock = np.random.normal(0.0, vol)
        prices.iloc[idx] = prices.iloc[idx - 1] * (1 + mu + shock)
    return prices.ffill()


def get_prices_stooq(tickers: Iterable[str], start: str, end: str | None) -> pd.DataFrame:
    """Fetch daily close prices from Stooq via pandas-datareader."""
    frames: List[pd.Series] = []
    for raw in tickers:
        tk = str(raw).strip()
        if not tk:
            continue
        try:
            df = pdr.DataReader(tk, "stooq", start=start, end=end)
        except Exception as exc:
            print(f"[warn] Failed to fetch {tk} from Stooq: {exc}")
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        series = df.sort_index()["Close"].rename(tk).dropna()
        if not series.empty:
            frames.append(series)
    if not frames:
        raise RuntimeError("No tickers fetched from Stooq. Check symbols and date range.")
    return pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")


def fetch_price_history(policy: Dict, tickers: List[str]) -> pd.DataFrame:
    """Fetch price history according to the policy's data block."""
    if not tickers:
        raise ValueError("Universe must contain at least one ticker to fetch prices.")
    data_cfg = policy.get("data", {}) if isinstance(policy, dict) else {}
    source = str(data_cfg.get("source", data_cfg.get("price_source", "stooq")) or "stooq").lower()
    start = str(data_cfg.get("start") or "2018-01-01")
    end = data_cfg.get("end")
    if isinstance(end, str) and not end.strip():
        end = None
    if end is not None:
        end = str(end)

    if source == "synthetic":
        synth_cfg = data_cfg.get("synthetic", {}) if isinstance(data_cfg.get("synthetic"), dict) else {}
        periods = int(data_cfg.get("synthetic_periods", synth_cfg.get("periods", 504)))
        seed = int(data_cfg.get("synthetic_seed", synth_cfg.get("seed", 42)))
        start_date = str(data_cfg.get("synthetic_start", synth_cfg.get("start", "2023-01-03")))
        return get_prices_synthetic(tickers, start=start_date, periods=periods, seed=seed)

    if source not in {"stooq"}:
        raise ValueError(f"Unsupported data source '{source}'. Expected 'stooq' or 'synthetic'.")

    return get_prices_stooq(tickers, start=start, end=end)
