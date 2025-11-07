"""CLI entry point for optimisation/backtesting driven by a YAML policy.

The policy file specifies input paths (price history, universe, fixed positions),
optimisation settings (model, objective, guardrails), and optional backtest
parameters. Command-line options may override selected paths for quick tests.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import pandas as pd
import yaml

from mod.backtest import Backtester
from mod.data_provider import fetch_price_history
from mod.optimizer import optimize_portfolio
from mod.universe import resolve_universe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portfolio optimiser/backtester driven by policy.yaml")
    parser.add_argument("--config", default="policy.yaml", help="Path to YAML policy file (default: policy.yaml).")
    parser.add_argument("--prices", help="Optional override for policy.data.price_file.")
    parser.add_argument("--universe", help="Optional override for policy.data.universe_file.")
    parser.add_argument("--fixed", help="Optional override for policy.portfolio.optimization.fixed_positions_file.")
    parser.add_argument("--model", help="Optional override for policy.portfolio.optimization.model.")
    parser.add_argument("--output-dir", help="Override output directory (policy.reporting.output_dir).")
    parser.add_argument("--weights-file", help="Override weights CSV name (policy.reporting.weights_file).")
    parser.add_argument("--backtest-file", help="Override backtest JSON name (policy.reporting.backtest_file).")
    parser.add_argument("--skip-backtest", action="store_true", help="Disable backtesting regardless of policy.")
    return parser.parse_args()


def _load_policy(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Policy file must decode into a dictionary.")
    return data


def _apply_overrides(policy: Dict[str, Any], args: argparse.Namespace) -> None:
    data_cfg: Dict[str, Any] = policy.setdefault("data", {})
    portfolio_cfg: Dict[str, Any] = policy.setdefault("portfolio", {})
    opt_cfg: Dict[str, Any] = portfolio_cfg.setdefault("optimization", {})

    if args.prices:
        data_cfg["price_file"] = args.prices
    if args.universe:
        data_cfg["universe_file"] = args.universe
        data_cfg.setdefault("stockrover_file", args.universe)
    if args.fixed:
        opt_cfg["fixed_positions_file"] = args.fixed
    if args.model:
        opt_cfg["model"] = args.model
    if "fixed_positions_file" not in opt_cfg and "fixed_positions_file" in data_cfg:
        opt_cfg["fixed_positions_file"] = data_cfg["fixed_positions_file"]
    if "stockrover_file" not in data_cfg and data_cfg.get("universe_file"):
        data_cfg["stockrover_file"] = data_cfg["universe_file"]

    reporting_cfg: Dict[str, Any] = policy.setdefault("reporting", {})
    if args.output_dir:
        reporting_cfg["output_dir"] = args.output_dir
    if args.weights_file:
        reporting_cfg["weights_file"] = args.weights_file
    if args.backtest_file:
        reporting_cfg["backtest_file"] = args.backtest_file


def _ensure_defaults(policy: Dict[str, Any]) -> None:
    data_cfg = policy.setdefault("data", {})
    if "stockrover_file" not in data_cfg and data_cfg.get("universe_file"):
        data_cfg["stockrover_file"] = data_cfg["universe_file"]

    portfolio_cfg = policy.setdefault("portfolio", {})
    opt_cfg = portfolio_cfg.setdefault("optimization", {})
    opt_cfg.setdefault("model", "mean_variance")
    opt_cfg.setdefault("engine", opt_cfg.get("model"))
    opt_cfg.setdefault("long_only", True)
    opt_cfg.setdefault("leverage", 1.0)
    opt_cfg.setdefault("risk_free_rate", 0.0)
    if "fixed_positions_file" not in opt_cfg and data_cfg.get("fixed_positions_file"):
        opt_cfg["fixed_positions_file"] = data_cfg["fixed_positions_file"]

    backtest_cfg = portfolio_cfg.setdefault("backtest", {})
    backtest_cfg.setdefault("enabled", False)
    backtest_cfg.setdefault("method", "in_sample")
    backtest_cfg.setdefault("rebalance_freq", "M")
    backtest_cfg.setdefault("train_years", 3)
    backtest_cfg.setdefault("test_years", 1)
    backtest_cfg.setdefault("transaction_cost_bps", 0.0)
    backtest_cfg.setdefault("risk_free_rate", opt_cfg.get("risk_free_rate", 0.0))

    reporting_cfg = policy.setdefault("reporting", {})
    reporting_cfg.setdefault("output_dir", "output")
    reporting_cfg.setdefault("weights_file", "weights_optimized.csv")
    reporting_cfg.setdefault("backtest_file", "backtest.json")


def _load_prices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Price file not found: {path}")
    df = pd.read_csv(path, index_col=0)
    if df.empty:
        raise ValueError(f"Price file is empty: {path}")
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as exc:  # pragma: no cover - surfaced in CLI usage
        raise ValueError(f"Failed to parse date index from {path}: {exc}") from exc
    df = df.sort_index().dropna(how="all")
    if df.empty:
        raise ValueError("No usable price data after dropping empty rows.")
    return df


def _filter_prices(prices: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    available = set(prices.columns)
    missing = sorted(t for t in tickers if t not in available)
    if missing:
        raise ValueError(f"Price data missing for tickers: {', '.join(missing)}")
    filtered = prices[tickers].dropna(axis=1, how="all")
    if filtered.empty or not filtered.columns.any():
        raise ValueError("No price columns remain after filtering by universe.")
    return filtered


def main() -> None:
    args = _parse_args()
    policy = _load_policy(args.config)
    _apply_overrides(policy, args)
    _ensure_defaults(policy)

    if args.skip_backtest:
        policy["portfolio"]["backtest"]["enabled"] = False

    data_cfg: Dict[str, Any] = policy.get("data", {})

    universe_override = args.universe or data_cfg.get("universe_file") or data_cfg.get("stockrover_file")
    tickers, note = resolve_universe(policy, universe_file=universe_override)
    print(f"Universe resolved ({note}): {len(tickers)} tickers")

    price_path = data_cfg.get("price_file")
    if price_path:
        prices = _load_prices(price_path)
    else:
        prices = fetch_price_history(policy, tickers)

    prices = _filter_prices(prices, tickers)
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Price history is insufficient to compute returns.")

    reporting = policy.get("reporting", {})
    output_dir = reporting.get("output_dir", "output")
    weights_file = reporting.get("weights_file", "weights_optimized.csv")
    backtest_file = reporting.get("backtest_file", "backtest.json")

    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, weights_file)

    optimisation = optimize_portfolio(prices, policy)
    weights = pd.Series(optimisation["weights"], name="weight")
    weights.to_csv(weights_path, header=True)
    print(f"Optimised weights saved to {weights_path}")

    backtest_cfg = policy["portfolio"]["backtest"]
    if backtest_cfg.get("enabled"):
        backtester = Backtester(prices, policy, returns=returns)
        result = backtester.run()
        backtest_path = os.path.join(output_dir, backtest_file)
        with open(backtest_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Backtest ({result['mode']}) saved to {backtest_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
