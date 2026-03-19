"""Market scanner — screen the universe and rank by strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Type

import pandas as pd

from massive_sm.backtester import Backtester
from massive_sm.data.client import MassiveClient
from massive_sm.strategy.base import Strategy, TradeAction


@dataclass
class ScanResult:
    symbol: str
    last_signal: str
    signal_strength: float
    last_close: float
    rsi: float | None = None
    atr_pct: float | None = None
    backtest_sharpe: float | None = None
    backtest_return_pct: float | None = None
    fundamentals: dict | None = None


class Scanner:
    """Scan a universe of symbols using one or more strategies."""

    def __init__(
        self,
        client: MassiveClient | None = None,
        lookback_days: int = 365,
    ):
        self.client = client or MassiveClient()
        self.lookback_days = lookback_days

    def scan(
        self,
        symbols: list[str],
        strategy_class: Type[Strategy],
        strategy_params: dict | None = None,
        run_backtest: bool = True,
    ) -> list[ScanResult]:
        """Scan symbols and return ranked results.

        Args:
            symbols: List of ticker symbols to scan.
            strategy_class: Strategy class to evaluate.
            strategy_params: Optional parameter overrides.
            run_backtest: If True, also run a quick backtest per symbol.
        """
        params = strategy_params or {}
        results: list[ScanResult] = []

        for symbol in symbols:
            try:
                data = self.client.get_bars(symbol, interval="1d")
                if data.empty or len(data) < 30:
                    continue

                strategy = strategy_class(**params)
                data = strategy.setup(data)
                signals = strategy.generate_signals(data)
                last_signal = signals[-1] if signals else None

                result = ScanResult(
                    symbol=symbol,
                    last_signal=last_signal.action.value if last_signal else "NONE",
                    signal_strength=last_signal.strength if last_signal else 0,
                    last_close=data["close"].iloc[-1],
                )

                # Attach RSI if computed
                if "rsi" in data.columns:
                    result.rsi = data["rsi"].iloc[-1]

                if run_backtest:
                    strategy_bt = strategy_class(**params)
                    bt = Backtester(strategy_bt)
                    bt_result = bt.run(data, symbol)
                    result.backtest_sharpe = bt_result.sharpe_ratio
                    result.backtest_return_pct = bt_result.total_return_pct

                results.append(result)

            except Exception:
                continue

        # Rank by signal strength, then backtest sharpe
        results.sort(
            key=lambda r: (
                1 if r.last_signal == "BUY" else 0,
                r.signal_strength,
                r.backtest_sharpe or 0,
            ),
            reverse=True,
        )
        return results

    def scan_with_fundamentals(
        self,
        symbols: list[str],
        strategy_class: Type[Strategy],
        strategy_params: dict | None = None,
    ) -> list[ScanResult]:
        """Scan with both technical and fundamental data."""
        results = self.scan(symbols, strategy_class, strategy_params)
        for r in results:
            try:
                r.fundamentals = self.client.get_fundamentals(r.symbol)
            except Exception:
                pass
        return results

    def format_results(self, results: list[ScanResult], top_n: int = 20) -> str:
        lines = [
            f"{'Symbol':<8} {'Signal':<6} {'Strength':>8} {'Close':>10} "
            f"{'RSI':>6} {'Sharpe':>8} {'Return%':>9}",
            "─" * 65,
        ]
        for r in results[:top_n]:
            rsi_str = f"{r.rsi:.1f}" if r.rsi is not None else "—"
            sharpe_str = f"{r.backtest_sharpe:.2f}" if r.backtest_sharpe is not None else "—"
            ret_str = f"{r.backtest_return_pct:+.1f}" if r.backtest_return_pct is not None else "—"
            lines.append(
                f"{r.symbol:<8} {r.last_signal:<6} {r.signal_strength:>8.2f} "
                f"${r.last_close:>9.2f} {rsi_str:>6} {sharpe_str:>8} {ret_str:>9}"
            )
        return "\n".join(lines)
