"""Performance and risk helper metrics shared across optimisation/backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _clean_series(series: pd.Series) -> pd.Series:
    return series.dropna() if isinstance(series, pd.Series) else pd.Series(dtype=float)


def annualize_return(daily_returns: pd.Series, freq: int = TRADING_DAYS) -> float:
    dr = _clean_series(daily_returns)
    if dr.empty:
        return float("nan")
    cumulative = (1 + dr).prod()
    periods = dr.shape[0]
    if periods <= 0 or cumulative <= 0:
        return float("nan")
    return float(cumulative ** (freq / periods) - 1)


def cagr(prices: pd.Series, freq: int = TRADING_DAYS) -> float:
    if prices.empty or len(prices) < 2:
        return float("nan")
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    years = len(prices) / float(freq)
    if years <= 0:
        return float("nan")
    return float((1 + total_return) ** (1 / years) - 1)


def volatility(daily_returns: pd.Series, freq: int = TRADING_DAYS) -> float:
    dr = _clean_series(daily_returns)
    if dr.empty:
        return float("nan")
    return float(dr.std() * np.sqrt(freq))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, freq: int = TRADING_DAYS) -> float:
    dr = _clean_series(daily_returns)
    if dr.empty:
        return float("nan")
    excess = dr - (risk_free_rate / freq)
    ann_ret = excess.mean() * freq
    ann_vol = volatility(dr, freq)
    if np.isnan(ann_vol) or np.isclose(ann_vol, 0.0):
        return float("nan")
    return float(ann_ret / ann_vol)


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0, freq: int = TRADING_DAYS) -> float:
    dr = _clean_series(daily_returns)
    if dr.empty:
        return float("nan")
    excess = dr - (risk_free_rate / freq)
    ann_ret = excess.mean() * freq
    downside = np.minimum(0.0, excess.values)
    downside_var = np.mean(np.square(downside))
    downside_dev = float(np.sqrt(downside_var) * np.sqrt(freq)) if downside_var >= 0 else float("nan")
    if downside_dev is None or np.isnan(downside_dev) or np.isclose(downside_dev, 0.0):
        return float("nan")
    return float(ann_ret / downside_dev)


def equity_curve(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod()


def max_drawdown(cum_returns: pd.Series) -> float:
    if cum_returns.empty:
        return float("nan")
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1.0
    return float(drawdown.min())


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if portfolio_returns.empty or benchmark_returns.empty:
        return float("nan")
    cov = np.cov(portfolio_returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    if np.isclose(var, 0.0):
        return float("nan")
    return float(cov / var)


def summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark: pd.Series | None = None,
    freq: int = TRADING_DAYS,
) -> dict:
    eq = equity_curve(returns)
    result = {
        "CAGR": cagr(eq, freq=freq),
        "Volatility": volatility(returns, freq=freq),
        "Sharpe": sharpe_ratio(returns, risk_free_rate=risk_free_rate, freq=freq),
        "Sortino": sortino_ratio(returns, risk_free_rate=risk_free_rate, freq=freq),
        "MaxDrawdown": max_drawdown(eq),
    }
    if benchmark is not None:
        result["Beta_vs_Bench"] = beta(returns, benchmark)
    clean = {}
    for key, val in result.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            clean[key] = None
        else:
            clean[key] = float(val)
    return clean
