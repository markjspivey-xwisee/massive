"""Command-line interface for the Massive Strategy Manager."""

from __future__ import annotations

import argparse
import sys

from massive_sm.backtester import Backtester
from massive_sm.data.client import MassiveClient
from massive_sm.optimizer import Optimizer
from massive_sm.scanner import Scanner
from massive_sm.strategy.examples import BUILTIN_STRATEGIES
from massive_sm.trader import Trader


def parse_params(param_strings: list[str] | None) -> dict:
    """Parse 'key=value' strings into a dict with auto type conversion."""
    if not param_strings:
        return {}
    params = {}
    for p in param_strings:
        k, v = p.split("=", 1)
        # Auto-convert types
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
        params[k] = v
    return params


def cmd_backtest(args):
    strategy_cls = BUILTIN_STRATEGIES[args.strategy]
    params = parse_params(args.param)
    strategy = strategy_cls(**params)

    client = MassiveClient()
    data = client.get_bars(args.symbol, start=args.start, end=args.end)

    bt = Backtester(
        strategy,
        initial_cash=args.cash,
        commission_pct=args.commission / 100 if args.commission else None,
    )
    result = bt.run(data, args.symbol)
    print(result.summary())
    print()

    if args.trades:
        print(f"\n{'Date':<12} {'Side':<6} {'Entry':>10} {'Exit':>10} {'P&L':>10} {'P&L%':>8}")
        print("─" * 60)
        for t in result.trades:
            print(
                f"{str(t.exit_date)[:10]:<12} {t.side:<6} "
                f"${t.entry_price:>9.2f} ${t.exit_price:>9.2f} "
                f"${t.pnl:>9.2f} {t.pnl_pct:>+7.1f}%"
            )


def cmd_optimize(args):
    strategy_cls = BUILTIN_STRATEGIES[args.strategy]

    # Parse param grid: fast_period=5,10,20
    param_grid = {}
    for p in args.grid:
        k, v = p.split("=", 1)
        values = []
        for x in v.split(","):
            try:
                values.append(int(x))
            except ValueError:
                try:
                    values.append(float(x))
                except ValueError:
                    values.append(x)
        param_grid[k] = values

    client = MassiveClient()
    data = client.get_bars(args.symbol, start=args.start, end=args.end)

    opt = Optimizer(
        strategy_cls, data, args.symbol,
        metric=args.metric, initial_cash=args.cash,
    )
    result = opt.grid_search(param_grid)
    print(result.summary())


def cmd_scan(args):
    strategy_cls = BUILTIN_STRATEGIES[args.strategy]
    params = parse_params(args.param)

    client = MassiveClient()
    if args.universe == "sp500":
        symbols = client.get_sp500()
    else:
        symbols = args.universe.split(",")

    scanner = Scanner(client)
    results = scanner.scan(
        symbols, strategy_cls, params,
        run_backtest=not args.no_backtest,
    )
    print(scanner.format_results(results, top_n=args.top))


def cmd_trade(args):
    strategy_cls = BUILTIN_STRATEGIES[args.strategy]
    params = parse_params(args.param)
    strategy = strategy_cls(**params)
    symbols = [s.strip() for s in args.symbols.split(",")]

    trader = Trader(
        strategy,
        symbols,
        initial_cash=args.cash,
        max_positions=args.max_positions,
        position_size_pct=args.size_pct / 100,
        stop_loss_pct=args.stop_loss / 100 if args.stop_loss else None,
        take_profit_pct=args.take_profit / 100 if args.take_profit else None,
        portfolio_path=args.portfolio_file,
    )
    trader.run_loop(
        interval_seconds=args.interval,
        max_ticks=args.max_ticks,
    )


def cmd_list(args):
    print("Available strategies:")
    for name, cls in BUILTIN_STRATEGIES.items():
        s = cls()
        print(f"  {name:<25} {s.name}")
        print(f"    params: {s.params}")
        print()


def main():
    parser = argparse.ArgumentParser(
        prog="massive",
        description="Massive Stock Strategy Manager",
    )
    sub = parser.add_subparsers(dest="command")

    # ── backtest ─────────────────────────────────────────────────
    bt = sub.add_parser("backtest", help="Backtest a strategy")
    bt.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    bt.add_argument("symbol", help="Ticker symbol (e.g. AAPL)")
    bt.add_argument("--start", help="Start date (YYYY-MM-DD)")
    bt.add_argument("--end", help="End date (YYYY-MM-DD)")
    bt.add_argument("--cash", type=float, default=100_000)
    bt.add_argument("--commission", type=float, help="Commission in %%")
    bt.add_argument("-p", "--param", action="append", help="key=value param override")
    bt.add_argument("--trades", action="store_true", help="Print trade log")
    bt.set_defaults(func=cmd_backtest)

    # ── optimize ─────────────────────────────────────────────────
    op = sub.add_parser("optimize", help="Optimize strategy parameters")
    op.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    op.add_argument("symbol", help="Ticker symbol")
    op.add_argument("--grid", action="append", required=True, help="param=val1,val2,val3")
    op.add_argument("--metric", default="sharpe_ratio", help="Metric to maximize")
    op.add_argument("--start", help="Start date")
    op.add_argument("--end", help="End date")
    op.add_argument("--cash", type=float, default=100_000)
    op.set_defaults(func=cmd_optimize)

    # ── scan ─────────────────────────────────────────────────────
    sc = sub.add_parser("scan", help="Scan market for signals")
    sc.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    sc.add_argument("--universe", default="sp500", help="sp500 or comma-separated symbols")
    sc.add_argument("--top", type=int, default=20, help="Show top N results")
    sc.add_argument("--no-backtest", action="store_true")
    sc.add_argument("-p", "--param", action="append")
    sc.set_defaults(func=cmd_scan)

    # ── trade ────────────────────────────────────────────────────
    tr = sub.add_parser("trade", help="Paper trade with a strategy")
    tr.add_argument("strategy", choices=list(BUILTIN_STRATEGIES))
    tr.add_argument("--symbols", required=True, help="Comma-separated symbols")
    tr.add_argument("--cash", type=float, default=100_000)
    tr.add_argument("--interval", type=int, default=60, help="Seconds between ticks")
    tr.add_argument("--max-positions", type=int, default=10)
    tr.add_argument("--size-pct", type=float, default=10, help="Position size as %% of portfolio")
    tr.add_argument("--stop-loss", type=float, help="Stop-loss %%")
    tr.add_argument("--take-profit", type=float, help="Take-profit %%")
    tr.add_argument("--max-ticks", type=int, help="Max ticks before stopping")
    tr.add_argument("--portfolio-file", default="portfolio.json")
    tr.add_argument("-p", "--param", action="append")
    tr.set_defaults(func=cmd_trade)

    # ── list ─────────────────────────────────────────────────────
    sub.add_parser("list", help="List available strategies").set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
