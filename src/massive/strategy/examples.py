"""Built-in example strategies."""

from __future__ import annotations

import pandas as pd

from massive import indicators as ind
from massive.strategy.base import Signal, Strategy, TradeAction


class SMACrossover(Strategy):
    """Simple / Exponential Moving Average crossover strategy."""

    name = "SMA Crossover"
    params = {"fast_period": 10, "slow_period": 30, "use_ema": False}

    def setup(self, data: pd.DataFrame) -> pd.DataFrame:
        fn = ind.ema if self.params["use_ema"] else ind.sma
        data["fast_ma"] = fn(data["close"], self.params["fast_period"])
        data["slow_ma"] = fn(data["close"], self.params["slow_period"])
        return data

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        prev_fast = None
        prev_slow = None
        for ts, row in data.iterrows():
            fast, slow = row["fast_ma"], row["slow_ma"]
            action = TradeAction.HOLD
            reason = ""
            if prev_fast is not None and not pd.isna(fast) and not pd.isna(slow):
                if prev_fast <= prev_slow and fast > slow:
                    action = TradeAction.BUY
                    reason = "Golden cross"
                elif prev_fast >= prev_slow and fast < slow:
                    action = TradeAction.SELL
                    reason = "Death cross"
            signals.append(Signal(action=action, symbol="", reason=reason))
            prev_fast, prev_slow = fast, slow
        return signals


class RSIMeanReversion(Strategy):
    """Buy oversold, sell overbought based on RSI."""

    name = "RSI Mean Reversion"
    params = {"rsi_period": 14, "oversold": 30, "overbought": 70}

    def setup(self, data: pd.DataFrame) -> pd.DataFrame:
        data["rsi"] = ind.rsi(data["close"], self.params["rsi_period"])
        return data

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        for ts, row in data.iterrows():
            r = row["rsi"]
            if pd.isna(r):
                signals.append(Signal(action=TradeAction.HOLD, symbol=""))
            elif r < self.params["oversold"]:
                signals.append(
                    Signal(
                        action=TradeAction.BUY,
                        symbol="",
                        strength=(self.params["oversold"] - r) / self.params["oversold"],
                        reason=f"RSI oversold ({r:.1f})",
                    )
                )
            elif r > self.params["overbought"]:
                signals.append(
                    Signal(
                        action=TradeAction.SELL,
                        symbol="",
                        strength=(r - self.params["overbought"])
                        / (100 - self.params["overbought"]),
                        reason=f"RSI overbought ({r:.1f})",
                    )
                )
            else:
                signals.append(Signal(action=TradeAction.HOLD, symbol=""))
        return signals


class MACDStrategy(Strategy):
    """Trade on MACD signal-line crossovers."""

    name = "MACD Crossover"
    params = {"fast": 12, "slow": 26, "signal": 9}

    def setup(self, data: pd.DataFrame) -> pd.DataFrame:
        macd_df = ind.macd(
            data["close"],
            fast=self.params["fast"],
            slow=self.params["slow"],
            signal=self.params["signal"],
        )
        data["macd"] = macd_df["macd"]
        data["macd_signal"] = macd_df["signal"]
        data["macd_hist"] = macd_df["histogram"]
        return data

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        prev_hist = None
        for ts, row in data.iterrows():
            h = row["macd_hist"]
            action = TradeAction.HOLD
            reason = ""
            if prev_hist is not None and not pd.isna(h) and not pd.isna(prev_hist):
                if prev_hist <= 0 and h > 0:
                    action = TradeAction.BUY
                    reason = "MACD bullish crossover"
                elif prev_hist >= 0 and h < 0:
                    action = TradeAction.SELL
                    reason = "MACD bearish crossover"
            signals.append(Signal(action=action, symbol="", reason=reason))
            prev_hist = h
        return signals


class BollingerBandStrategy(Strategy):
    """Mean-reversion using Bollinger Bands."""

    name = "Bollinger Band Reversion"
    params = {"bb_period": 20, "bb_std": 2.0}

    def setup(self, data: pd.DataFrame) -> pd.DataFrame:
        bb = ind.bollinger_bands(
            data["close"], self.params["bb_period"], self.params["bb_std"]
        )
        data["bb_upper"] = bb["upper"]
        data["bb_middle"] = bb["middle"]
        data["bb_lower"] = bb["lower"]
        return data

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        signals: list[Signal] = []
        for ts, row in data.iterrows():
            c, u, l = row["close"], row["bb_upper"], row["bb_lower"]
            if pd.isna(u):
                signals.append(Signal(action=TradeAction.HOLD, symbol=""))
            elif c <= l:
                signals.append(
                    Signal(action=TradeAction.BUY, symbol="", reason="Price at lower BB")
                )
            elif c >= u:
                signals.append(
                    Signal(action=TradeAction.SELL, symbol="", reason="Price at upper BB")
                )
            else:
                signals.append(Signal(action=TradeAction.HOLD, symbol=""))
        return signals


# Registry for CLI/scanner use
BUILTIN_STRATEGIES = {
    "sma_crossover": SMACrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "macd": MACDStrategy,
    "bollinger": BollingerBandStrategy,
}
