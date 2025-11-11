"""CLI entry point for optimisation/backtesting driven by a YAML policy.

The policy file specifies input paths (price history, universe, fixed positions),
optimisation settings (model, objective, guardrails), and optional backtest
parameters. Command-line options may override selected paths for quick tests.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from mod.backtest import Backtester
from mod.data_provider import fetch_price_history
from mod.optimizer import optimize_portfolio
from mod.universe import resolve_universe

UPLOAD_DIR_NAME = "to_be_processed"
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"
COUNTER_FILENAME = ".user_counter"
LOG_FILENAME = "run_log.csv"
MANIFEST_FILENAME = "run_manifest.csv"
REQUIRED_UPLOAD_FILES = ("policy.yaml", "universe.txt", "lock.csv")
POLICY_USER_FIELDS = (
    ("user_id",),
    ("userId",),
    ("userID",),
    ("user", "id"),
    ("user", "user_id"),
    ("user", "userId"),
)
FILENAME_TS_FORMAT = "%Y%m%dT%H%M%SZ"
LOG_HEADERS = ["timestamp", "user_id", "status", "message"]
MANIFEST_HEADERS = [
    "user_id",
    "uploaded_at",
    "input_dir",
    "input_policy",
    "input_universe",
    "input_lock",
    "output_dir",
    "completed_at",
    "status",
    "note",
]


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
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip scanning the /to_be_processed folder and run directly against --config.",
    )
    return parser.parse_args()


def _ensure_workspace(base_dir: str) -> Dict[str, str]:
    upload_dir = os.path.join(base_dir, UPLOAD_DIR_NAME)
    input_dir = os.path.join(base_dir, INPUT_DIR_NAME)
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    for path in (upload_dir, input_dir, output_dir):
        os.makedirs(path, exist_ok=True)
    return {
        "upload_dir": upload_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "counter_path": os.path.join(output_dir, COUNTER_FILENAME),
        "log_path": os.path.join(output_dir, LOG_FILENAME),
        "manifest_path": os.path.join(output_dir, MANIFEST_FILENAME),
    }


def _ensure_csv_headers(path: str, headers: List[str]) -> None:
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)


def _append_csv_row(path: str, headers: List[str], row: Dict[str, str]) -> None:
    _ensure_csv_headers(path, headers)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writerow({key: row.get(key, "") for key in headers})


def _log_event(log_path: str, user_id: str, status: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    _append_csv_row(
        log_path,
        LOG_HEADERS,
        {"timestamp": timestamp, "user_id": user_id, "status": status, "message": message},
    )


def _clear_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        target = os.path.join(directory, name)
        if os.path.isdir(target):
            shutil.rmtree(target, ignore_errors=True)
        else:
            try:
                os.remove(target)
            except FileNotFoundError:
                continue


def _sanitize_user_id(raw_id: object) -> Optional[str]:
    if raw_id is None:
        return None
    text = str(raw_id).strip()
    if not text:
        return None
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    cleaned = cleaned.strip("_- ")
    return cleaned or None


def _next_user_id(counter_path: str) -> str:
    counter = 0
    if os.path.exists(counter_path):
        with open(counter_path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
            if value.isdigit():
                counter = int(value)
    counter += 1
    with open(counter_path, "w", encoding="utf-8") as handle:
        handle.write(str(counter))
    return f"user_{counter:04d}"


def _policy_user_id_policy(data: Any) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for path in POLICY_USER_FIELDS:
        cursor: Any = data
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor.get(key)
        user_id = _sanitize_user_id(cursor)
        if user_id:
            return user_id
    return None


def _policy_user_id_from_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return None
    return _policy_user_id_policy(data)


def _ingest_upload_set(workspace: Dict[str, str]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    upload_dir = workspace["upload_dir"]
    log_path = workspace["log_path"]
    required_paths = {name: os.path.join(upload_dir, name) for name in REQUIRED_UPLOAD_FILES}
    present = {name: path for name, path in required_paths.items() if os.path.isfile(path)}

    if not present:
        return None, "No upload set detected in /to_be_processed."

    missing = [name for name in REQUIRED_UPLOAD_FILES if name not in present]
    if missing:
        message = f"Partial upload detected; missing files: {', '.join(missing)}"
        _log_event(log_path, "N/A", "warning", message)
        return None, message

    policy_user_id = _policy_user_id_from_file(required_paths["policy.yaml"])
    user_id = policy_user_id or _next_user_id(workspace["counter_path"])
    uploaded_at = datetime.now(timezone.utc)
    uploaded_stamp = uploaded_at.strftime(FILENAME_TS_FORMAT)
    user_input_dir = os.path.join(workspace["input_dir"], user_id)
    user_output_dir = os.path.join(workspace["output_dir"], user_id)
    os.makedirs(user_input_dir, exist_ok=True)
    os.makedirs(user_output_dir, exist_ok=True)

    relocated: Dict[str, str] = {}
    for name, src in required_paths.items():
        dest_name = f"{user_id}_{uploaded_stamp}_{name}"
        dest_path = os.path.join(user_input_dir, dest_name)
        shutil.move(src, dest_path)
        relocated[name] = dest_path

    _clear_directory(upload_dir)
    _log_event(log_path, user_id, "queued", f"Upload moved to {user_input_dir}")

    job = {
        "user_id": user_id,
        "uploaded_at": uploaded_at.isoformat(),
        "input_dir": user_input_dir,
        "output_dir": user_output_dir,
        "policy_path": relocated["policy.yaml"],
        "universe_path": relocated["universe.txt"],
        "lock_path": relocated["lock.csv"],
    }
    return job, None


def _write_manifest_entry(manifest_path: str, job: Dict[str, str], status: str, note: str = "") -> None:
    row = {
        "user_id": job["user_id"],
        "uploaded_at": job["uploaded_at"],
        "input_dir": job["input_dir"],
        "input_policy": job["policy_path"],
        "input_universe": job["universe_path"],
        "input_lock": job["lock_path"],
        "output_dir": job["output_dir"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "note": note,
    }
    _append_csv_row(manifest_path, MANIFEST_HEADERS, row)


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


def _force_job_paths(policy: Dict[str, Any], job: Dict[str, str]) -> None:
    data_cfg = policy.setdefault("data", {})
    data_cfg["universe_file"] = job["universe_path"]
    data_cfg["stockrover_file"] = job["universe_path"]
    data_cfg["fixed_positions_file"] = job["lock_path"]

    portfolio_cfg = policy.setdefault("portfolio", {})
    opt_cfg = portfolio_cfg.setdefault("optimization", {})
    opt_cfg["fixed_positions_file"] = job["lock_path"]

    reporting_cfg = policy.setdefault("reporting", {})
    reporting_cfg["output_dir"] = job["output_dir"]


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


def _run_pipeline(policy: Dict[str, Any], args: argparse.Namespace) -> None:
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


def _process_job(job: Dict[str, str], args: argparse.Namespace, workspace: Dict[str, str]) -> None:
    user_id = job["user_id"]
    log_path = workspace["log_path"]
    manifest_path = workspace["manifest_path"]

    _log_event(log_path, user_id, "started", f"Processing upload for {user_id}")
    try:
        policy = _load_policy(job["policy_path"])
        _apply_overrides(policy, args)
        _force_job_paths(policy, job)
        _ensure_defaults(policy)
        if args.skip_backtest:
            policy["portfolio"]["backtest"]["enabled"] = False
        _run_pipeline(policy, args)
    except Exception as exc:
        note = str(exc)
        _log_event(log_path, user_id, "failed", note)
        _write_manifest_entry(manifest_path, job, "failed", note)
        raise
    else:
        _log_event(log_path, user_id, "succeeded", f"Outputs saved to {job['output_dir']}")
        _write_manifest_entry(manifest_path, job, "succeeded")


def main() -> None:
    args = _parse_args()
    workspace = _ensure_workspace(os.path.abspath(os.getcwd()))

    if not args.skip_upload:
        job, reason = _ingest_upload_set(workspace)
        if job:
            _process_job(job, args, workspace)
            return
        if reason:
            print(reason)
        else:
            print("No upload jobs ready.")
        return

    policy = _load_policy(args.config)
    _apply_overrides(policy, args)
    _ensure_defaults(policy)

    if args.skip_backtest:
        policy["portfolio"]["backtest"]["enabled"] = False

    _run_pipeline(policy, args)


if __name__ == "__main__":  # pragma: no cover
    main()
