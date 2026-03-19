"""Base strategy framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    HOLD = "HOLD"


@dataclass
class Signal:
    action: TradeAction
    symbol: str
    strength: float = 1.0  # 0.0 – 1.0
    size_pct: float = 1.0  # fraction of available capital
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    shares: float
    entry_price: float
    entry_date: Any = None
    side: str = "long"  # "long" or "short"

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    def unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.shares
        return (self.entry_price - current_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        return self.unrealized_pnl(current_price) / self.cost_basis


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
        - name: human-readable strategy name
        - params: dict of tunable parameters with defaults
        - setup(): compute indicators, called once per backtest run
        - generate_signals(): produce buy/sell signals for each bar
    """

    name: str = "BaseStrategy"
    params: dict = {}

    def __init__(self, **param_overrides):
        self.params = {**self.__class__.params, **param_overrides}

    @abstractmethod
    def setup(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to the dataframe. Return the enriched df."""
        ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Return a list of Signals, one per bar row."""
        ...

    def should_exit(
        self, position: Position, current_bar: pd.Series
    ) -> Signal | None:
        """Optional per-bar exit logic (stop-loss, take-profit, etc.).

        Override in subclass for custom exit rules.
        """
        return None

    def __repr__(self):
        return f"{self.name}({self.params})"
