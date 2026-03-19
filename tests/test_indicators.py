"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from massive_sm import indicators as ind


@pytest.fixture
def price_series():
    """Generate a simple price series for testing."""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    return pd.Series(prices, name="close")


@pytest.fixture
def ohlcv_data(price_series):
    n = len(price_series)
    np.random.seed(42)
    return pd.DataFrame({
        "open": price_series + np.random.randn(n) * 0.2,
        "high": price_series + abs(np.random.randn(n) * 0.5),
        "low": price_series - abs(np.random.randn(n) * 0.5),
        "close": price_series,
        "volume": np.random.randint(1000, 10000, n),
    })


class TestSMA:
    def test_length(self, price_series):
        result = ind.sma(price_series, 20)
        assert len(result) == len(price_series)

    def test_first_values_nan(self, price_series):
        result = ind.sma(price_series, 20)
        assert result.iloc[:19].isna().all()
        assert not pd.isna(result.iloc[19])

    def test_correct_value(self, price_series):
        result = ind.sma(price_series, 5)
        expected = price_series.iloc[:5].mean()
        assert abs(result.iloc[4] - expected) < 1e-10


class TestEMA:
    def test_length(self, price_series):
        result = ind.ema(price_series, 20)
        assert len(result) == len(price_series)


class TestRSI:
    def test_range(self, price_series):
        result = ind.rsi(price_series, 14)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestMACD:
    def test_columns(self, price_series):
        result = ind.macd(price_series)
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_is_diff(self, price_series):
        result = ind.macd(price_series)
        diff = result["macd"] - result["signal"]
        np.testing.assert_array_almost_equal(result["histogram"], diff)


class TestBollingerBands:
    def test_columns(self, price_series):
        result = ind.bollinger_bands(price_series, 20)
        assert list(result.columns) == ["upper", "middle", "lower"]

    def test_ordering(self, price_series):
        result = ind.bollinger_bands(price_series, 20).dropna()
        assert (result["upper"] >= result["middle"]).all()
        assert (result["middle"] >= result["lower"]).all()


class TestATR:
    def test_positive(self, ohlcv_data):
        result = ind.atr(ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"])
        valid = result.dropna()
        assert (valid > 0).all()


class TestOBV:
    def test_length(self, ohlcv_data):
        result = ind.obv(ohlcv_data["close"], ohlcv_data["volume"])
        assert len(result) == len(ohlcv_data)
