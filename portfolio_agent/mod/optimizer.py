from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .perf_metrics import annualize_return, equity_curve, max_drawdown, sharpe_ratio, volatility
from .fixed_positions import load_fixed_positions
from .risk_tools import annualized_cov, risk_contribution

try:  # scipy is required for the Sharpe guardrail solver
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover - surfaced during runtime if scipy is missing
    minimize = None

TRADING_DAYS = 252

_GUARDRAIL_SYNONYMS = {
    "max_volatility": ("target_volatility",),
    "max_drawdown": ("target_max_drawdown",),
    "min_return": ("target_return", "min_ann_return"),
    "min_sharpe": (),
}


class SharpeGuardrailOptimizer:
    """Maximise Sharpe ratio while enforcing optional risk guardrails."""

    def __init__(self, returns: pd.DataFrame, cfg: Dict):
        if minimize is None:
            raise ImportError("scipy is required for SharpeGuardrailOptimizer.")

        if returns is None or returns.empty:
            raise ValueError("Sharpe optimisation requires non-empty returns data.")

        self.returns = returns.dropna(how="all").copy()
        self.tickers = list(self.returns.columns)
        if not self.tickers:
            raise ValueError("No assets available for optimisation.")

        cfg = cfg or {}

        self.freq = int(cfg.get("frequency", TRADING_DAYS))
        leverage = cfg.get("leverage", 1.0)
        self.target_leverage = float(leverage if leverage not in (None, "") else 1.0)

        self.min_weight = float(cfg.get("min_weight", 0.0) or 0.0)
        self.max_weight = float(cfg.get("max_weight", 1.0) or 1.0)
        self.long_only = bool(cfg.get("long_only", True))

        backtest_cfg = cfg.get("backtest", {}) if isinstance(cfg.get("backtest"), dict) else {}
        self.risk_free_rate = float(cfg.get("risk_free_rate", backtest_cfg.get("risk_free_rate", 0.0)) or 0.0)

        self.guardrails: Dict[str, float] = {}
        raw_guardrails: Dict[str, float] = {}
        for key in ("guardrails", "constraints", "targets"):
            block = cfg.get(key)
            if isinstance(block, dict):
                raw_guardrails.update(block)
        for canonical, aliases in _GUARDRAIL_SYNONYMS.items():
            value = raw_guardrails.get(canonical)
            if value is None:
                for alias in aliases:
                    if alias in raw_guardrails:
                        value = raw_guardrails[alias]
                        break
            if value is not None:
                try:
                    self.guardrails[canonical] = float(value)
                except (TypeError, ValueError):
                    continue

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def _bounds(self):
        lower = max(0.0, self.min_weight) if self.long_only else self.min_weight
        return tuple((lower, self.max_weight) for _ in range(self.n_assets))

    @staticmethod
    def _constraint_sum_to_target(weights: np.ndarray, target: float) -> float:
        return float(np.sum(weights) - target)

    def _constraints(self, returns: pd.DataFrame):
        cons = [{"type": "eq", "fun": lambda w: self._constraint_sum_to_target(w, self.target_leverage)}]

        if "max_volatility" in self.guardrails:
            limit = self.guardrails["max_volatility"]
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, cap=limit, data=returns: self._constraint_max_volatility(w, data, cap),
                }
            )

        if "max_drawdown" in self.guardrails:
            limit = self.guardrails["max_drawdown"]
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, cap=limit, data=returns: self._constraint_max_drawdown(w, data, cap),
                }
            )

        if "min_return" in self.guardrails:
            floor = self.guardrails["min_return"]
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, thresh=floor, data=returns: self._constraint_min_return(w, data, thresh),
                }
            )

        if "min_sharpe" in self.guardrails:
            floor = self.guardrails["min_sharpe"]
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda w, thresh=floor, data=returns: self._constraint_min_sharpe(w, data, thresh),
                }
            )

        return tuple(cons)

    def _prepare_returns(self, returns: Optional[pd.DataFrame]) -> pd.DataFrame:
        if returns is None:
            return self.returns.loc[:, self.tickers]
        return returns.reindex(columns=self.tickers).dropna(how="all")

    def _portfolio_statistics(self, weights: np.ndarray, returns: pd.DataFrame) -> Optional[Dict[str, float]]:
        port_rets = returns.dot(weights).dropna()
        if port_rets.empty:
            return None

        eq = equity_curve(port_rets)
        return {
            "ann_return": float(annualize_return(port_rets, freq=self.freq)),
            "ann_vol": float(volatility(port_rets, freq=self.freq)),
            "sharpe": float(sharpe_ratio(port_rets, risk_free_rate=self.risk_free_rate, freq=self.freq)),
            "max_drawdown": float(max_drawdown(eq)),
        }

    def _objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        stats = self._portfolio_statistics(weights, returns)
        sharpe = None if stats is None else stats.get("sharpe")
        if sharpe is None or np.isnan(sharpe):
            return 1e6
        return float(-sharpe)

    def _constraint_max_volatility(self, weights: np.ndarray, returns: pd.DataFrame, limit: float) -> float:
        stats = self._portfolio_statistics(weights, returns)
        vol = None if stats is None else stats.get("ann_vol")
        if vol is None or np.isnan(vol):
            return -1e3
        return float(limit - vol)

    def _constraint_max_drawdown(self, weights: np.ndarray, returns: pd.DataFrame, limit: float) -> float:
        stats = self._portfolio_statistics(weights, returns)
        drawdown = None if stats is None else stats.get("max_drawdown")
        if drawdown is None or np.isnan(drawdown):
            return -1e3
        return float(limit - abs(drawdown))

    def _constraint_min_return(self, weights: np.ndarray, returns: pd.DataFrame, floor: float) -> float:
        stats = self._portfolio_statistics(weights, returns)
        ann_ret = None if stats is None else stats.get("ann_return")
        if ann_ret is None or np.isnan(ann_ret):
            return -1e3
        return float(ann_ret - floor)

    def _constraint_min_sharpe(self, weights: np.ndarray, returns: pd.DataFrame, floor: float) -> float:
        stats = self._portfolio_statistics(weights, returns)
        sharpe = None if stats is None else stats.get("sharpe")
        if sharpe is None or np.isnan(sharpe):
            return -1e3
        return float(sharpe - floor)

    def optimize(self, returns_window: Optional[pd.DataFrame] = None) -> np.ndarray:
        data = self._prepare_returns(returns_window)
        if data.empty:
            raise ValueError("Returns window for optimisation is empty.")

        x0 = np.full(self.n_assets, self.target_leverage / self.n_assets)
        res = minimize(
            fun=lambda w: self._objective(w, data),
            x0=x0,
            bounds=self._bounds(),
            constraints=self._constraints(data),
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-9, "disp": False},
        )

        if not res.success or res.x is None:
            return x0
        return res.x

    def summarize(self, weights: np.ndarray, returns_window: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        stats = self._portfolio_statistics(weights, self._prepare_returns(returns_window))
        if stats is None:
            return {
                "ann_return": float("nan"),
                "ann_vol": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
            }
        return stats


def _parse_weight_bounds(cfg: Dict, long_only: bool) -> Tuple[Optional[float], Optional[float]]:
    bounds = cfg.get("weight_bounds")
    min_weight = cfg.get("min_weight")
    max_weight = cfg.get("max_weight")

    if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        min_weight, max_weight = bounds

    if min_weight is not None:
        try:
            min_weight = float(min_weight)
        except (TypeError, ValueError):
            min_weight = None
    if max_weight is not None:
        try:
            max_weight = float(max_weight)
        except (TypeError, ValueError):
            max_weight = None

    if min_weight is None and long_only:
        min_weight = 0.0
    if max_weight is None and long_only:
        max_weight = 1.0

    return min_weight, max_weight


def _portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = returns.reindex(columns=weights.index).fillna(0.0)
    return aligned.dot(weights.values)


def _summarize_portfolio(returns: pd.Series, freq: int, risk_free_rate: float) -> Dict[str, float]:
    ann_ret = annualize_return(returns, freq=freq)
    ann_vol = volatility(returns, freq=freq)
    sharpe = sharpe_ratio(returns, risk_free_rate=risk_free_rate, freq=freq)
    mdd = max_drawdown(equity_curve(returns))
    return {
        "ann_return": float(ann_ret) if ann_ret is not None else float("nan"),
        "ann_vol": float(ann_vol) if ann_vol is not None else float("nan"),
        "sharpe": float(sharpe) if sharpe is not None else float("nan"),
        "max_drawdown": float(mdd) if mdd is not None else float("nan"),
    }


def _solve_mv_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    delta: float,
    long_only: bool,
    min_weight: Optional[float],
    max_weight: Optional[float],
) -> np.ndarray:
    n = len(mu)
    kkt = np.zeros((n + 1, n + 1))
    kkt[:n, :n] = delta * cov
    kkt[:n, -1] = 1.0
    kkt[-1, :n] = 1.0
    rhs = np.zeros(n + 1)
    rhs[:n] = mu
    rhs[-1] = 1.0

    try:
        sol = np.linalg.solve(kkt, rhs)
        weights = sol[:n]
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(delta * cov)
        weights = pinv @ mu

    weights = np.asarray(weights, dtype=float)

    if long_only:
        lower = 0.0 if min_weight is None else float(min_weight)
        weights = np.clip(weights, lower, None)
        total = weights.sum()
        if total <= 0:
            weights = np.full(n, 1.0 / n)
        else:
            weights /= total
        if max_weight is not None:
            weights = np.clip(weights, None, float(max_weight))
            total = weights.sum()
            if total > 0:
                weights /= total
    else:
        lower = float(min_weight) if min_weight is not None else None
        upper = float(max_weight) if max_weight is not None else None
        if lower is not None or upper is not None:
            lower_val = lower if lower is not None else -np.inf
            upper_val = upper if upper is not None else np.inf
            weights = np.clip(weights, lower_val, upper_val)
        total = weights.sum()
        if not np.isclose(total, 0.0):
            weights /= total

    total = weights.sum()
    if np.isclose(total, 0.0):
        weights = np.full(n, 1.0 / n)
    else:
        weights /= total
    return weights


def _black_litterman_weights(
    returns: pd.DataFrame,
    config: Dict,
    freq: int,
    long_only: bool,
    min_weight: Optional[float],
    max_weight: Optional[float],
    target_leverage: float,
) -> Dict[str, object]:
    tickers = list(returns.columns)
    if not tickers:
        raise ValueError("Black-Litterman optimisation requires at least one asset.")

    cov = returns.cov() * freq
    bl_cfg = config.get("black_litterman", {})
    if not isinstance(bl_cfg, dict):
        bl_cfg = {}

    market_caps_cfg = bl_cfg.get("market_caps")
    market_caps = None
    if isinstance(market_caps_cfg, dict):
        market_caps = pd.Series(market_caps_cfg, dtype=float).reindex(tickers)
    elif isinstance(market_caps_cfg, (list, tuple)) and len(market_caps_cfg) == len(tickers):
        market_caps = pd.Series(market_caps_cfg, index=tickers, dtype=float)

    if market_caps is None:
        market_weights = np.full(len(tickers), 1.0 / len(tickers))
    else:
        market_caps = market_caps.fillna(0.0).clip(lower=0.0)
        total = float(market_caps.sum())
        if total <= 0:
            market_weights = np.full(len(tickers), 1.0 / len(tickers))
        else:
            market_weights = market_caps.values / total

    delta = bl_cfg.get("risk_aversion")
    if delta is None:
        delta = config.get("risk_aversion")
    if delta is None:
        delta = 2.5
    delta = float(delta)

    pi_cfg = bl_cfg.get("pi")
    if isinstance(pi_cfg, dict):
        pi_series = pd.Series(pi_cfg, dtype=float).reindex(tickers).fillna(0.0)
        pi_vec = pi_series.values
    elif isinstance(pi_cfg, (list, tuple)) and len(pi_cfg) == len(tickers):
        pi_vec = np.array(pi_cfg, dtype=float)
    else:
        pi_vec = cov.values.dot(market_weights) * delta

    tau = float(bl_cfg.get("tau", 0.05))
    absolute_views_cfg = bl_cfg.get("absolute_views")
    view_series = pd.Series(dtype=float)
    if isinstance(absolute_views_cfg, dict):
        view_series = pd.Series(absolute_views_cfg, dtype=float)
        view_series = view_series[[t for t in view_series.index if t in tickers]]

    details: Dict[str, object] = {
        "market_weights": {tickers[i]: float(market_weights[i]) for i in range(len(tickers))},
        "risk_aversion_used": float(delta),
        "tau": tau,
    }

    if view_series.empty:
        mu_bl = pi_vec
        cov_bl = cov.values
    else:
        P = np.zeros((len(view_series), len(tickers)))
        Q = view_series.values
        for idx, ticker in enumerate(view_series.index):
            col_idx = tickers.index(ticker)
            P[idx, col_idx] = 1.0
        tau_cov = tau * cov.values
        inv_tau_cov = np.linalg.pinv(tau_cov)
        omega_diag = np.diag(P @ tau_cov @ P.T)
        omega_diag[omega_diag <= 0] = 1e-6
        omega = np.diag(omega_diag)
        inv_omega = np.linalg.pinv(omega)
        middle = inv_tau_cov + P.T @ inv_omega @ P
        rhs = inv_tau_cov.dot(pi_vec) + P.T.dot(inv_omega).dot(Q)
        mu_bl = np.linalg.solve(middle, rhs)
        cov_bl = cov.values + np.linalg.inv(middle)
        details["absolute_views"] = view_series.to_dict()

    weights_np = _solve_mv_weights(
        mu_bl,
        cov_bl,
        delta=delta,
        long_only=long_only,
        min_weight=min_weight,
        max_weight=max_weight,
    )

    weights_series = pd.Series(weights_np, index=tickers, dtype=float)
    current_sum = float(weights_series.sum())
    if not np.isclose(current_sum, 0.0):
        weights_series *= target_leverage / current_sum
    else:
        weights_series[:] = target_leverage / len(weights_series)

    details["posterior_mean"] = {ticker: float(val) for ticker, val in zip(tickers, mu_bl)}
    details["posterior_cov_diag"] = {
        ticker: float(cov_bl[i, i]) for i, ticker in enumerate(tickers)
    }
    return {"weights": weights_series, "details": details}


def _risk_parity_weights(
    returns: pd.DataFrame,
    freq: int,
    long_only: bool,
    min_weight: Optional[float],
    max_weight: Optional[float],
    target_leverage: float,
) -> Dict[str, object]:
    tickers = list(returns.columns)
    if not tickers:
        raise ValueError("Risk parity optimisation requires at least one asset.")

    vol = returns.std(ddof=0)
    inv_vol = 1.0 / vol.replace(0.0, np.nan)
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)

    weights_series = inv_vol / inv_vol.sum()
    weights_series = weights_series.reindex(tickers).fillna(0.0)

    if long_only:
        lower = 0.0 if min_weight is None else float(min_weight)
        weights_series = weights_series.clip(lower=lower)
    if max_weight is not None:
        weights_series = weights_series.clip(upper=float(max_weight))

    total = weights_series.sum()
    if total <= 0:
        weights_series = pd.Series(np.full(len(tickers), 1.0 / len(tickers)), index=tickers)
    else:
        weights_series /= total

    weights_series *= target_leverage

    cov = annualized_cov(returns, periods_per_year=freq)
    rc = risk_contribution(weights_series, cov)

    details: Dict[str, object] = {
        "volatility_estimate": {ticker: float(vol.get(ticker, np.nan)) for ticker in tickers},
        "risk_contribution": {ticker: float(rc.get(ticker, np.nan)) for ticker in rc.index},
    }
    return {"weights": weights_series, "details": details}


def _optimise_free_slice(
    returns: pd.DataFrame,
    config: Dict,
    model: str,
    long_only: bool,
    min_weight: Optional[float],
    max_weight: Optional[float],
    target_leverage: float,
    freq: int,
    risk_free_rate: float,
) -> Tuple[pd.Series, str, Dict[str, object], Dict[str, float], Optional[str]]:
    details: Dict[str, object] = {}
    guardrails: Dict[str, float] = {}
    canonical_model = model
    objective = config.get("objective")

    if target_leverage <= 0 or returns.empty:
        empty = pd.Series(0.0, index=returns.columns, dtype=float)
        details["solver"] = "None"
        return empty, canonical_model, details, guardrails, objective

    if model in ("mean_variance", "mv", "sharpe", "sharpe_guardrail"):
        optimiser_cfg = dict(config)
        if min_weight is not None:
            optimiser_cfg["min_weight"] = min_weight
        if max_weight is not None:
            optimiser_cfg["max_weight"] = max_weight
        optimiser_cfg["leverage"] = target_leverage
        optimiser_cfg["long_only"] = long_only
        optimizer = SharpeGuardrailOptimizer(returns=returns, cfg=optimiser_cfg)
        weights_arr = optimizer.optimize()
        weights_series = pd.Series(weights_arr, index=returns.columns, dtype=float)
        guardrails = optimizer.guardrails
        details["solver"] = "SharpeGuardrailOptimizer"
        details["objective"] = "max_sharpe"
        canonical_model = "mean_variance"
        objective = objective or "max_sharpe"
    elif model in ("black_litterman", "black-litterman", "bl"):
        bl_res = _black_litterman_weights(
            returns=returns,
            config=config,
            freq=freq,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            target_leverage=target_leverage,
        )
        weights_series = bl_res["weights"]
        details.update(bl_res.get("details", {}))
        details["solver"] = "BlackLittermanClosedForm"
        details.setdefault("objective", "max_expected_utility")
        canonical_model = "black_litterman"
        objective = objective or details.get("objective")
    elif model in ("risk_parity", "risk-parity", "rp", "hrp"):
        rp_res = _risk_parity_weights(
            returns=returns,
            freq=freq,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            target_leverage=target_leverage,
        )
        weights_series = rp_res["weights"]
        details.update(rp_res.get("details", {}))
        details["solver"] = "InverseVolatilityRiskParity"
        details.setdefault("objective", "risk_parity")
        canonical_model = "risk_parity"
        objective = objective or details.get("objective")
    elif model in ("equal", "equal_weight", "ew"):
        weights_series = pd.Series(
            np.full(len(returns.columns), target_leverage / max(len(returns.columns), 1)),
            index=returns.columns,
            dtype=float,
        )
        details["solver"] = "EqualWeight"
        details.setdefault("objective", "equal_weight")
        canonical_model = "equal_weight"
        objective = objective or details.get("objective")
    else:
        raise ValueError(f"Unsupported optimisation model '{model}'.")

    return weights_series, canonical_model, details, guardrails, objective


def optimize_portfolio(prices: pd.DataFrame, policy: dict) -> dict:
    """Optimise asset weights given historical prices and a policy definition.

    Supported models:
        - mean_variance (Sharpe maximisation with guardrails)
        - black_litterman (closed-form, absolute views)
        - risk_parity (inverse-volatility heuristic)
        - equal_weight (baseline comparison)
    """

    config: Dict = {}
    if isinstance(policy, dict):
        port_block = policy.get("portfolio", policy)
        if isinstance(port_block, dict):
            config = port_block.get("optimization", port_block) or {}
            if not isinstance(config, dict):
                config = {}

    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Price history is insufficient to compute returns for optimisation.")

    freq = int(config.get("frequency", config.get("freq", TRADING_DAYS)) or TRADING_DAYS)
    risk_free_rate = float(config.get("risk_free_rate", 0.0) or 0.0)
    model_raw = config.get("model") or config.get("engine") or "mean_variance"
    model = str(model_raw).lower()

    long_only = bool(config.get("long_only", True))
    min_weight, max_weight = _parse_weight_bounds(config, long_only=long_only)
    target_leverage = float(config.get("leverage", config.get("target_leverage", 1.0)) or 1.0)

    fixed_weights, fixed_meta = load_fixed_positions(policy, returns.columns)
    fixed_weights = fixed_weights.reindex(returns.columns).fillna(0.0)

    sum_fixed = float(fixed_weights.sum())
    if sum_fixed > target_leverage + 1e-8:
        raise ValueError(
            f"Sum of fixed position weights ({sum_fixed:.4f}) exceeds target leverage ({target_leverage:.4f})."
        )

    residual_budget = max(target_leverage - sum_fixed, 0.0)
    fixed_tickers = set(fixed_weights.index[fixed_weights > 0])
    free_columns = [col for col in returns.columns if col not in fixed_tickers]
    free_returns = returns.loc[:, free_columns] if free_columns else returns.iloc[:, :0]

    weights_free = pd.Series(0.0, index=free_columns, dtype=float)
    details: Dict[str, object] = {}
    guardrails: Dict[str, float] = {}
    canonical_model = "fixed_only" if not free_columns else model
    objective = config.get("objective")

    if fixed_meta:
        details["fixed_positions"] = {
            "count": len(fixed_meta),
            "weights": {ticker: float(fixed_weights.get(ticker, 0.0)) for ticker in fixed_meta},
            "meta": fixed_meta,
        }

    if residual_budget > 1e-8 and free_columns:
        weights_free, canonical_model, model_details, guardrails, objective = _optimise_free_slice(
            returns=free_returns,
            config=config,
            model=model,
            long_only=long_only,
            min_weight=min_weight,
            max_weight=max_weight,
            target_leverage=residual_budget,
            freq=freq,
            risk_free_rate=risk_free_rate,
        )
        details.update(model_details or {})
        sum_free = float(weights_free.sum())
        if sum_free > 0 and not np.isclose(sum_free, residual_budget):
            weights_free *= residual_budget / sum_free
    elif residual_budget > 1e-8 and not free_columns:
        details.setdefault("solver", "FixedPositionsOnly")
        details["note"] = "Residual budget with no free assets; fixed weights scaled proportionally."
        if sum_fixed > 0:
            scale = target_leverage / sum_fixed
            fixed_weights = fixed_weights * scale
            sum_fixed = float(fixed_weights.sum())
    else:
        details.setdefault("solver", "FixedPositionsOnly")

    combined_weights = pd.Series(0.0, index=returns.columns, dtype=float)
    if not fixed_weights.empty:
        combined_weights.loc[fixed_weights.index] = fixed_weights.values
    if not weights_free.empty:
        combined_weights.loc[weights_free.index] = combined_weights.loc[weights_free.index] + weights_free.values

    pf_returns = _portfolio_returns(returns, combined_weights)
    if pf_returns.empty:
        raise ValueError("Optimised portfolio return series is empty.")

    summary = _summarize_portfolio(pf_returns, freq=freq, risk_free_rate=risk_free_rate)
    leverage_actual = float(combined_weights.sum())

    result = {
        "weights": {ticker: float(w) for ticker, w in combined_weights.items()},
        **summary,
        "model": canonical_model,
        "objective": objective,
        "details": details,
        "guardrails": guardrails,
        "leverage": leverage_actual,
        "frequency": freq,
        "risk_free_rate": risk_free_rate,
    }
    return result
