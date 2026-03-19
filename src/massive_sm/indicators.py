"""Technical indicators computed on pandas DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend ────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


# ── Momentum ─────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator (%K and %D)."""
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"%K": k, "%D": d})


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (percentage)."""
    return series.pct_change(periods=period) * 100


# ── Volatility ───────────────────────────────────────────────────

def bollinger_bands(
    series: pd.Series, period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """Bollinger Bands (upper, middle, lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    return pd.DataFrame(
        {
            "upper": middle + std_dev * std,
            "middle": middle,
            "lower": middle - std_dev * std,
        }
    )


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ── Volume ───────────────────────────────────────────────────────

def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Volume-Weighted Average Price (cumulative intraday)."""
    typical = (high + low + close) / 3
    return (typical * volume).cumsum() / volume.cumsum()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    return (direction * volume).fillna(0).cumsum()
