"""Paper & live trading engine with portfolio management."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd

from massive_sm.config import config
from massive_sm.data.client import MassiveClient
from massive_sm.strategy.base import Position, Signal, Strategy, TradeAction


@dataclass
class OrderFill:
    symbol: str
    side: str
    shares: float
    fill_price: float
    timestamp: str
    reason: str = ""


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    order_history: list[OrderFill] = field(default_factory=list)
    pnl_history: list[dict] = field(default_factory=list)

    @property
    def position_symbols(self) -> list[str]:
        return list(self.positions.keys())

    def total_equity(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            if pos.side == "long":
                equity += pos.shares * price
            else:
                equity += pos.shares * (2 * pos.entry_price - price)
        return equity

    def summary(self, prices: dict[str, float]) -> str:
        equity = self.total_equity(prices)
        lines = [
            f"═══ Portfolio Summary ═══",
            f"Cash:     ${self.cash:,.2f}",
            f"Equity:   ${equity:,.2f}",
            f"Positions: {len(self.positions)}",
        ]
        for sym, pos in self.positions.items():
            price = prices.get(sym, pos.entry_price)
            pnl = pos.unrealized_pnl(price)
            pnl_pct = pos.unrealized_pnl_pct(price) * 100
            lines.append(
                f"  {sym}: {pos.shares:.2f} shares @ ${pos.entry_price:.2f} "
                f"({pos.side}) → P&L: ${pnl:,.2f} ({pnl_pct:+.1f}%)"
            )
        lines.append(f"Total trades: {len(self.order_history)}")
        return "\n".join(lines)

    def save(self, path: str | Path):
        """Persist portfolio state to JSON."""
        state = {
            "cash": self.cash,
            "positions": {
                sym: {
                    "shares": p.shares,
                    "entry_price": p.entry_price,
                    "entry_date": str(p.entry_date),
                    "side": p.side,
                }
                for sym, p in self.positions.items()
            },
            "order_history": [
                {
                    "symbol": o.symbol,
                    "side": o.side,
                    "shares": o.shares,
                    "fill_price": o.fill_price,
                    "timestamp": o.timestamp,
                    "reason": o.reason,
                }
                for o in self.order_history
            ],
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Portfolio:
        """Load portfolio state from JSON."""
        state = json.loads(Path(path).read_text())
        portfolio = cls(cash=state["cash"])
        for sym, p in state["positions"].items():
            portfolio.positions[sym] = Position(
                symbol=sym,
                shares=p["shares"],
                entry_price=p["entry_price"],
                entry_date=p["entry_date"],
                side=p["side"],
            )
        for o in state.get("order_history", []):
            portfolio.order_history.append(OrderFill(**o))
        return portfolio


class Trader:
    """Paper trading engine that runs strategies in real-time or on a schedule.

    Supports:
    - Paper trading (simulated fills at market price)
    - Multi-symbol portfolio management
    - Risk management (position sizing, max positions, stop-loss)
    - State persistence (save/load portfolio)
    """

    def __init__(
        self,
        strategy: Strategy,
        symbols: list[str],
        client: MassiveClient | None = None,
        initial_cash: float | None = None,
        max_positions: int = 10,
        position_size_pct: float = 0.1,
        stop_loss_pct: float | None = 0.05,
        take_profit_pct: float | None = 0.15,
        portfolio_path: str | None = None,
    ):
        self.strategy = strategy
        self.symbols = [s.upper() for s in symbols]
        self.client = client or MassiveClient()
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.portfolio_path = portfolio_path

        if portfolio_path and Path(portfolio_path).exists():
            self.portfolio = Portfolio.load(portfolio_path)
        else:
            self.portfolio = Portfolio(cash=initial_cash or config.default_cash)

    def tick(self) -> list[OrderFill]:
        """Run one cycle: fetch data, generate signals, execute trades.

        Call this on a schedule (e.g., every minute, every market open).
        """
        fills: list[OrderFill] = []

        # Get current quotes
        try:
            quotes = self.client.get_quotes(self.symbols)
            prices = {q["symbol"]: q["price"] for q in quotes}
        except Exception:
            return fills

        # Check stop-loss / take-profit on existing positions
        for sym in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[sym]
            price = prices.get(sym)
            if price is None:
                continue

            pnl_pct = pos.unrealized_pnl_pct(price)
            should_exit = False
            reason = ""

            if self.stop_loss_pct and pnl_pct < -self.stop_loss_pct:
                should_exit = True
                reason = f"Stop-loss triggered ({pnl_pct*100:.1f}%)"
            elif self.take_profit_pct and pnl_pct > self.take_profit_pct:
                should_exit = True
                reason = f"Take-profit triggered ({pnl_pct*100:.1f}%)"

            if should_exit:
                fill = self._close_position(sym, price, reason)
                if fill:
                    fills.append(fill)

        # Generate signals for each symbol
        for symbol in self.symbols:
            try:
                data = self.client.get_bars(symbol, interval="1d")
                if data.empty or len(data) < 30:
                    continue

                data = self.strategy.setup(data)
                signals = self.strategy.generate_signals(data)
                if not signals:
                    continue

                signal = signals[-1]
                signal.symbol = symbol
                price = prices.get(symbol, data["close"].iloc[-1])

                fill = self._execute_signal(signal, price)
                if fill:
                    fills.append(fill)

            except Exception:
                continue

        # Record equity snapshot
        self.portfolio.pnl_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "equity": self.portfolio.total_equity(prices),
                "cash": self.portfolio.cash,
                "n_positions": len(self.portfolio.positions),
            }
        )

        # Persist state
        if self.portfolio_path:
            self.portfolio.save(self.portfolio_path)

        return fills

    def run_loop(self, interval_seconds: int = 60, max_ticks: int | None = None):
        """Run tick() in a loop. Use for paper trading."""
        tick_count = 0
        print(f"Starting trader loop (interval={interval_seconds}s)")
        print(f"Symbols: {self.symbols}")
        print(f"Strategy: {self.strategy}")
        print()

        while True:
            tick_count += 1
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Tick #{tick_count}")

            fills = self.tick()
            for fill in fills:
                print(
                    f"  → {fill.side.upper()} {fill.shares:.2f} {fill.symbol} "
                    f"@ ${fill.fill_price:.2f} ({fill.reason})"
                )

            if not fills:
                print("  No trades this tick")

            if max_ticks and tick_count >= max_ticks:
                print("Max ticks reached, stopping.")
                break

            time.sleep(interval_seconds)

    def _execute_signal(self, signal: Signal, price: float) -> OrderFill | None:
        sym = signal.symbol

        if signal.action == TradeAction.BUY and sym not in self.portfolio.positions:
            if len(self.portfolio.positions) >= self.max_positions:
                return None
            alloc = self.portfolio.cash * self.position_size_pct
            if alloc < price:
                return None
            shares = alloc / price
            cost = shares * price
            self.portfolio.cash -= cost
            self.portfolio.positions[sym] = Position(
                symbol=sym,
                shares=shares,
                entry_price=price,
                entry_date=datetime.now().isoformat(),
                side="long",
            )
            fill = OrderFill(
                symbol=sym, side="buy", shares=shares,
                fill_price=price, timestamp=datetime.now().isoformat(),
                reason=signal.reason,
            )
            self.portfolio.order_history.append(fill)
            return fill

        elif signal.action == TradeAction.SELL and sym in self.portfolio.positions:
            return self._close_position(sym, price, signal.reason)

        return None

    def _close_position(self, symbol: str, price: float, reason: str) -> OrderFill | None:
        pos = self.portfolio.positions.pop(symbol, None)
        if pos is None:
            return None
        proceeds = pos.shares * price
        self.portfolio.cash += proceeds
        fill = OrderFill(
            symbol=symbol, side="sell", shares=pos.shares,
            fill_price=price, timestamp=datetime.now().isoformat(),
            reason=reason,
        )
        self.portfolio.order_history.append(fill)
        return fill
