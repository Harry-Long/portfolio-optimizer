from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .optimizer import optimize_portfolio
from .perf_metrics import equity_curve, summary as perf_summary


class Backtester:
    def __init__(self, prices: pd.DataFrame, policy: Dict, returns: Optional[pd.DataFrame] = None):
        self.prices = prices.copy()
        self.returns = returns.copy() if returns is not None else prices.pct_change().dropna()
        self.returns.index = pd.to_datetime(self.returns.index)
        self.policy = policy or {}

        port = self.policy.get("portfolio", {}) if isinstance(self.policy.get("portfolio"), dict) else self.policy
        self.opt_cfg = (port.get("optimization") or {}) if isinstance(port, dict) else {}
        self.engine = (self.opt_cfg.get("engine") or "pyportfolioopt").lower().strip()

        bt_cfg = port.get("backtest") if isinstance(port, dict) else None
        self.bt_cfg = bt_cfg if isinstance(bt_cfg, dict) else {}

        self.enabled = bool(self.bt_cfg.get("enabled", False))
        self.rebalance = (self.bt_cfg.get("rebalance_freq") or "M").upper()
        self.tc_bps = float(self.bt_cfg.get("transaction_cost_bps", 0) or 0.0)
        self.risk_free_rate = float(
            self.bt_cfg.get("risk_free_rate", self.opt_cfg.get("risk_free_rate", 0.0)) or 0.0
        )
        self.train_years = int(self.bt_cfg.get("train_years", 5) or 5)
        self.test_years = int(self.bt_cfg.get("test_years", 1) or 1)
        self.method = (self.bt_cfg.get("method") or "in_sample").lower().strip()

        self.tickers = list(self.returns.columns)

    def _apply_transaction_costs(self, prev_w: np.ndarray, new_w: np.ndarray) -> np.ndarray:
        if self.tc_bps <= 0:
            return new_w
        turnover = np.abs(new_w - prev_w).sum()
        cost = (self.tc_bps / 10_000.0) * turnover
        return new_w * (1 - cost)

    def _weights_to_series(self, weights_dict: Dict[str, float]) -> pd.Series:
        return pd.Series(weights_dict).reindex(self.tickers).fillna(0.0)

    def run(self) -> Dict:
        if not self.enabled:
            raise ValueError("Backtesting is disabled via policy.portfolio.backtest.enabled")
        if self.method == "walk_forward":
            return self.run_walk_forward()
        return self.run_in_sample()

    def run_in_sample(self) -> Dict:
        opt = optimize_portfolio(self.prices, self.policy)
        weights_series = self._weights_to_series(opt["weights"])
        port_returns = self.returns.dot(weights_series.values).dropna()
        metrics = perf_summary(port_returns, risk_free_rate=self.risk_free_rate)
        eq = equity_curve(port_returns)
        period_start = str(eq.index[0].date()) if not eq.empty else None
        period_end = str(eq.index[-1].date()) if not eq.empty else None

        return {
            "mode": "in_sample",
            "engine": self.engine,
            "weights": {t: float(w) for t, w in weights_series.items()},
            "metrics": metrics,
            "period": {"start": period_start, "end": period_end},
        }

    def run_walk_forward(self) -> Dict:
        returns = self.returns.copy()
        prices = self.prices.copy()
        returns.index = pd.to_datetime(returns.index)
        prices.index = pd.to_datetime(prices.index)

        rebalance_dates = returns.resample(self.rebalance).last().index

        all_weights = []
        portfolio_returns = pd.Series(index=returns.index, dtype=float)
        current_weights = np.full(len(self.tickers), 1.0 / len(self.tickers))

        for dt in rebalance_dates:
            train_end = dt
            train_start = dt - pd.DateOffset(years=self.train_years)
            test_end = dt + pd.DateOffset(years=self.test_years)

            train_mask = (returns.index > train_start) & (returns.index <= train_end)
            test_mask = (returns.index > train_end) & (returns.index <= test_end)

            train_returns = returns.loc[train_mask]
            test_returns = returns.loc[test_mask]
            if len(train_returns) < 60 or test_returns.empty:
                continue

            idx = prices.index.intersection(train_returns.index)
            train_prices = prices.loc[idx]
            if train_prices.empty:
                continue

            opt = optimize_portfolio(train_prices, self.policy)
            weights_series = self._weights_to_series(opt["weights"])
            new_weights = weights_series.values
            new_weights = self._apply_transaction_costs(current_weights, new_weights)
            current_weights = new_weights

            all_weights.append(
                {
                    "date": str(train_end.date()),
                    "weights": {ticker: float(weight) for ticker, weight in zip(self.tickers, current_weights)},
                }
            )

            test_port = test_returns.dot(current_weights).dropna()
            portfolio_returns.loc[test_port.index] = test_port.values

        portfolio_returns = portfolio_returns.dropna()
        metrics = perf_summary(portfolio_returns, risk_free_rate=self.risk_free_rate)
        eq = equity_curve(portfolio_returns)

        start = str(eq.index[0].date()) if not eq.empty else None
        end = str(eq.index[-1].date()) if not eq.empty else None

        return {
            "mode": "walk_forward",
            "engine": self.engine,
            "weights_over_time": all_weights,
            "metrics": metrics,
            "period": {"start": start, "end": end},
        }
