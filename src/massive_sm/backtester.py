"""Backtesting engine — event-driven, single-symbol or portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from massive_sm.config import config
from massive_sm.strategy.base import Position, Signal, Strategy, TradeAction


@dataclass
class Trade:
    symbol: str
    side: str  # "long" or "short"
    entry_date: Any
    entry_price: float
    exit_date: Any
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    reason_entry: str = ""
    reason_exit: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    symbol: str
    start_date: Any
    end_date: Any
    initial_cash: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    total_trades: int
    avg_trade_pnl_pct: float
    profit_factor: float
    trades: list[Trade]
    equity_curve: pd.Series

    def summary(self) -> str:
        lines = [
            f"═══ {self.strategy_name} on {self.symbol} ═══",
            f"Period:          {self.start_date} → {self.end_date}",
            f"Initial capital: ${self.initial_cash:,.2f}",
            f"Final equity:    ${self.final_equity:,.2f}",
            f"Total return:    {self.total_return_pct:+.2f}%",
            f"CAGR:            {self.cagr_pct:+.2f}%",
            f"Max drawdown:    {self.max_drawdown_pct:.2f}%",
            f"Sharpe ratio:    {self.sharpe_ratio:.2f}",
            f"Sortino ratio:   {self.sortino_ratio:.2f}",
            f"Win rate:        {self.win_rate:.1f}%",
            f"Total trades:    {self.total_trades}",
            f"Avg trade P&L:   {self.avg_trade_pnl_pct:+.2f}%",
            f"Profit factor:   {self.profit_factor:.2f}",
        ]
        return "\n".join(lines)


class Backtester:
    """Run a Strategy against historical OHLCV data."""

    def __init__(
        self,
        strategy: Strategy,
        initial_cash: float | None = None,
        commission_pct: float | None = None,
        slippage_pct: float | None = None,
    ):
        self.strategy = strategy
        self.initial_cash = initial_cash or config.default_cash
        self.commission_pct = commission_pct if commission_pct is not None else config.commission_pct
        self.slippage_pct = slippage_pct if slippage_pct is not None else config.slippage_pct

    def run(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        data = data.copy()
        data = self.strategy.setup(data)
        signals = self.strategy.generate_signals(data)

        cash = self.initial_cash
        position: Position | None = None
        trades: list[Trade] = []
        equity_values: list[float] = []
        equity_dates: list = []

        for i, (ts, bar) in enumerate(data.iterrows()):
            signal = signals[i] if i < len(signals) else Signal(action=TradeAction.HOLD, symbol=symbol)
            signal.symbol = symbol
            price = bar["close"]

            # Check custom exit rules first
            if position is not None:
                exit_signal = self.strategy.should_exit(position, bar)
                if exit_signal is not None and exit_signal.action in (
                    TradeAction.SELL,
                    TradeAction.COVER,
                ):
                    signal = exit_signal
                    signal.symbol = symbol

            # Execute signal
            if signal.action == TradeAction.BUY and position is None:
                fill_price = price * (1 + self.slippage_pct)
                shares = (cash * signal.size_pct) / fill_price
                commission = shares * fill_price * self.commission_pct
                cash -= shares * fill_price + commission
                position = Position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=fill_price,
                    entry_date=ts,
                    side="long",
                )

            elif signal.action == TradeAction.SELL and position is not None and position.side == "long":
                fill_price = price * (1 - self.slippage_pct)
                proceeds = position.shares * fill_price
                commission = proceeds * self.commission_pct
                cash += proceeds - commission
                pnl = (fill_price - position.entry_price) * position.shares - commission
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="long",
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=ts,
                        exit_price=fill_price,
                        shares=position.shares,
                        pnl=pnl,
                        pnl_pct=(fill_price / position.entry_price - 1) * 100,
                        reason_entry=signal.reason,
                        reason_exit=signal.reason,
                    )
                )
                position = None

            elif signal.action == TradeAction.SHORT and position is None:
                fill_price = price * (1 - self.slippage_pct)
                shares = (cash * signal.size_pct) / fill_price
                commission = shares * fill_price * self.commission_pct
                cash += shares * fill_price - commission  # short proceeds
                position = Position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=fill_price,
                    entry_date=ts,
                    side="short",
                )

            elif signal.action == TradeAction.COVER and position is not None and position.side == "short":
                fill_price = price * (1 + self.slippage_pct)
                cost = position.shares * fill_price
                commission = cost * self.commission_pct
                cash -= cost + commission
                pnl = (position.entry_price - fill_price) * position.shares - commission
                trades.append(
                    Trade(
                        symbol=symbol,
                        side="short",
                        entry_date=position.entry_date,
                        entry_price=position.entry_price,
                        exit_date=ts,
                        exit_price=fill_price,
                        shares=position.shares,
                        pnl=pnl,
                        pnl_pct=(position.entry_price / fill_price - 1) * 100,
                    )
                )
                position = None

            # Track equity
            equity = cash
            if position is not None:
                if position.side == "long":
                    equity += position.shares * price
                else:
                    equity += position.shares * (2 * position.entry_price - price)
            equity_values.append(equity)
            equity_dates.append(ts)

        equity_curve = pd.Series(equity_values, index=equity_dates, name="equity")
        metrics = self._compute_metrics(equity_curve, trades, data)

        return BacktestResult(
            strategy_name=self.strategy.name,
            params=dict(self.strategy.params),
            symbol=symbol,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_cash=self.initial_cash,
            final_equity=equity_values[-1] if equity_values else self.initial_cash,
            trades=trades,
            equity_curve=equity_curve,
            **metrics,
        )

    def _compute_metrics(
        self, equity: pd.Series, trades: list[Trade], data: pd.DataFrame
    ) -> dict:
        total_return = (equity.iloc[-1] / self.initial_cash - 1) * 100
        days = (data.index[-1] - data.index[0]).days or 1
        years = days / 365.25
        cagr = ((equity.iloc[-1] / self.initial_cash) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Drawdown
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = dd.min() * 100

        # Returns
        daily_returns = equity.pct_change().dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            downside = daily_returns[daily_returns < 0].std()
            sortino = (daily_returns.mean() / downside) * np.sqrt(252) if downside > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # Trade stats
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = (len(winning) / len(trades) * 100) if trades else 0
        avg_pnl = np.mean([t.pnl_pct for t in trades]) if trades else 0
        gross_profit = sum(t.pnl for t in winning) or 0
        gross_loss = abs(sum(t.pnl for t in losing)) or 1
        profit_factor = gross_profit / gross_loss if gross_loss else float("inf")

        return {
            "total_return_pct": total_return,
            "cagr_pct": cagr,
            "max_drawdown_pct": abs(max_dd),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "avg_trade_pnl_pct": avg_pnl,
            "profit_factor": profit_factor,
        }
