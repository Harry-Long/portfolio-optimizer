# Portfolio Optimisation Core

A compact toolkit for running portfolio optimisation with optional backtesting.
The CLI expects pre-collected price data, a universe file that lists admissible
tickers, and (optionally) a CSV describing fixed positions that must be locked
during optimisation.

## Prerequisites
- Python environment described in `environment.yml` (NumPy, pandas, SciPy,
  PyPortfolioOpt, PyYAML, pandas-datareader).
- Update `policy.yaml`:
  - `data.universe_file`: txt/csv listing admissible tickers (comments with `#`
    are ignored). The same path is used by the optimiser and the data fetcher.
  - `data.source`: set to `stooq` (default) to download price history on the
    fly or `synthetic` for randomised paths. Supplying `data.price_file`
    overrides fetching if you prefer to point at a pre-built CSV.
  - `data.fixed_positions_file`: optional CSV describing locked holdings (see
    loader in `mod/fixed_positions.py` for accepted columns).
  - `portfolio.optimization`: choose the model/objective, weight bounds, and
    guardrails.
  - `portfolio.backtest`: configure in-sample or walk-forward evaluation.

## Running the optimiser
```bash
python portfolio_agent/run.py            # uses policy.yaml in the current directory
```
Optional overrides (e.g. alternate config or scratch CSV) can be supplied:
```bash
python portfolio_agent/run.py \
  --config my_policy.yaml \
  --prices data/prices_latest.csv \
  --universe data/universe/new_universe.txt
```

Outputs are written to the directory defined in `reporting.output_dir`:
- Optimised weights CSV (`reporting.weights_file`).
- Backtest summary JSON when `portfolio.backtest.enabled` is true.

## Modules of interest
- `portfolio_agent/run.py`: CLI entry point.
- `portfolio_agent/mod/optimizer.py`: optimisation engines and summary helper.
- `portfolio_agent/mod/backtest.py`: in-sample and walk-forward backtesting.
- `portfolio_agent/mod/data_provider.py`: price fetcher (Stooq or synthetic).
- `portfolio_agent/mod/fixed_positions.py`: parser for locked holdings.
- `portfolio_agent/mod/universe.py`: universe loader for txt/CSV exports.
- `portfolio_agent/mod/perf_metrics.py`: performance metric utilities.
- `portfolio_agent/mod/risk_tools.py`: return/covariance helpers used by the optimiser.

These modules form the minimal surface needed to run allocation experiments
once price data is already available.
