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
The CLI now manages a simple drop-folder workflow and will, by default, look
for uploaded input files before launching a run.

```bash
python portfolio_agent/run.py            # scans /to_be_processed for a full input set
```
Optional overrides (e.g. alternate config or scratch CSV) can still be
supplied, but any run triggered from `/to_be_processed` is forced to read from `/input`
and write into `/output/<user_id>`:

```bash
python portfolio_agent/run.py \
  --config my_policy.yaml \
  --prices data/prices_latest.csv \
  --universe data/universe/new_universe.txt
```

Outputs are written to the directory defined in `reporting.output_dir` (or the
per-user subfolder when using the upload workflow):
- Optimised weights CSV (`reporting.weights_file`).
- Backtest summary JSON when `portfolio.backtest.enabled` is true.

Pass `--skip-upload` to bypass the upload workflow entirely and run directly
against `--config`, which preserves the original single-run behaviour.

## Optimisation models & objectives
- **mean_variance** (default): Sharpe Guardrail optimiser that maximises
  `max_sharpe` subject to optional guardrails. Defaults — `frequency: 252`,
  `risk_free_rate: 0.0`, `leverage: 1.0`, `long_only: true`, `weight_bounds:
  [0, 1]` (or `[0, leverage]` when leverage ≠ 1). Guardrails accept
  `max_volatility`, `max_drawdown`, `min_return`, `min_sharpe` (aliases such as
  `target_volatility` are recognised in the CLI).
- **black_litterman**: closed-form solution using absolute views. Defaults —
  `black_litterman.risk_aversion: 2.5`, `black_litterman.tau: 0.05`, market
  weights derived from `black_litterman.market_caps` (or equal-weighted if
  absent), and prior returns (`pi`) inferred from equilibrium unless
  `black_litterman.pi` is supplied. Objective reported as
  `max_expected_utility`.
- **risk_parity**: inverse-volatility heuristic targeting uniform risk
  contribution with optional `weight_bounds`/`long_only`/`leverage` taken from
  the main optimisation block. Objective reported as `risk_parity`.
- **equal_weight**: baseline allocation that distributes leverage evenly across
  the investable universe, obeying `leverage` and any weight bounds.

Set `portfolio.optimization.model` to one of the above (synonyms such as `mv`,
`bl`, `rp`, `ew` are accepted). The `objective` field is optional and serves as
metadata; recognised values are `max_sharpe`, `max_expected_utility`,
`risk_parity`, and `equal_weight`. Regardless of the label, the behaviour is
controlled by the `model` and guardrail settings described above.

### Parameter reference & rationale
- `frequency`: trading days per year used to annualise returns/volatility (252
  for US equities). Changing it lets you align the optimiser with other markets
  (e.g., 365 for crypto, 260 for futures).
- `risk_free_rate`: baseline yield (per annum). Setting this closer to current
  cash yields keeps the Sharpe calculation grounded, especially during
  high-rate regimes.
- `leverage`: target sum of weights. Keep it at 1.0 for fully invested cash
  portfolios; increase when modelling gross exposure (e.g., 1.2 for 20% gross
  leverage).
- `long_only`: toggles whether short positions are allowed. Leave `true` for
  long-only mandates; set `false` if you can short or run market-neutral
  strategies.
- `weight_bounds`: min/max allocation per asset. Tight bounds reduce
  concentration risk and keep the outputs practical for large universes.
- `guardrails.*`: risk constraints for the Sharpe optimiser. Use
  `max_volatility` and `max_drawdown` to cap total risk, `min_return` to demand
  a minimum annualised return, and `min_sharpe` to avoid low-quality solutions.
  These are enforced via nonlinear constraints inside the solver.
- `black_litterman.tau`: scales the uncertainty placed on the prior
  covariance—higher values mean you trust your views more relative to the
  market equilibrium.
- `black_litterman.risk_aversion`: controls how aggressively the posterior
  tilts toward higher-return views. Use 2.5 as a neutral baseline; raise it for
  more conservative allocations.
- `black_litterman.market_caps`: maps tickers to market caps to derive
  equilibrium weights. Equal weighting is used when omitted, but supplying real
  market caps ensures the prior matches actual dominance in the index.
- `black_litterman.absolute_views`: dictionary of return expectations (e.g.,
  `{AAPL: 0.12}` = “AAPL should earn 12% annualised”). Views are optional but
  provide the main edge of the model.
- `risk_parity.method`: currently `inverse_volatility`, producing weights
  inversely proportional to realised volatility. This is a fast heuristic for
  users who want diversified risk without modelling means/covariances.
- `equal_weight`: parameter-less baseline—useful as a control group when
  benchmarking the other models.

### When to pick each model
- **mean_variance**: best when you trust historical returns/covariances and
  need fine-grained control via guardrails. It can absorb transaction-cost
  considerations through the backtest block and handles fixed positions.
- **black_litterman**: appropriate when you have a reliable market-cap prior
  and analyst/quant views that should gently tilt the portfolio. It produces
  smooth allocations and avoids overreacting to noisy mean estimates.
- **risk_parity**: favourable when relative risk targeting is more important
  than alpha forecasting (e.g., diversified multi-asset sleeves). It requires
  minimal configuration and remains stable during volatile periods.
- **equal_weight**: acts as a sanity check or fallback when users want a quick
  allocation without modelling choices. Use it to benchmark more complex runs.

## Upload workflow
- Required directories live in the project root and are created on demand:
  - `/to_be_processed` — drop `policy.yaml`, `universe.txt`, and `lock.csv` here.
  - `/input` — populated automatically with per-user subfolders once a full
    upload is detected.
  - `/output` — contains one subfolder per user plus shared CSV logs.
- Each CLI invocation (without `--skip-upload`) will:
  1. Check `/to_be_processed` for the required trio of files. Partial uploads are left
     untouched, logged to `/output/run_log.csv`, and the run exits early.
  2. When all three files are present, assign the next incremental ID
     (`user_0001`, `user_0002`, ...) or reuse the `user_id` provided inside
    `policy.yaml` (top-level `user_id` or `user.id`). Each file is renamed to
    `<user_id>_<ISO8601>_<original>` before being moved into `/input/<user_id>`
    so you can trace when/how it was ingested. The `/to_be_processed` folder is
    cleared afterwards.
  3. Run the optimiser/backtester using the relocated files. All artefacts are
     written to `/output/<user_id>/`, regardless of what the policy specified.
- Logs and manifests:
  - `/output/run_log.csv` captures every event (queued, started, succeeded,
    failed, warnings) with timestamps and user IDs.
  - `/output/run_manifest.csv` records a per-run summary (input/output paths,
    uploaded timestamp, completion timestamp, status, notes).
- Manual runs do not touch `/to_be_processed` / `/input`; invoke `python
  portfolio_agent/run.py --skip-upload ...` if you simply want to operate on a
  policy file already stored elsewhere.

## Local setup
Use Python 3.11+ and install the dependencies with pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python portfolio_agent/run.py
```

## Docker usage
Build the image from the project root:

```bash
docker build -t portfolio-optimizer .
```

Create the working directories on the host and mount them so uploads/outputs
persist outside the container:

```bash
mkdir -p to_be_processed input output
docker run --rm \
  -v $(pwd)/to_be_processed:/app/to_be_processed \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  portfolio-optimizer
```

The container’s default command scans `/to_be_processed` once and processes any
complete upload. To bypass the upload workflow for a one-off policy, append the
usual CLI flags:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/to_be_processed:/app/to_be_processed \  # optional but ensures the folder exists
  portfolio-optimizer \
  python portfolio_agent/run.py --skip-upload --config policy.yaml --skip-backtest
```

Example policy snippet with a reusable user identifier:

```yaml
user:
  id: client_acme
data:
  source: stooq
  universe_file: data/universe.txt
  fixed_positions_file: data/lock.csv
```
If `user.id`/`user_id` is omitted, the CLI falls back to sequential IDs
(`user_0001`, `user_0002`, ...).

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
