"""Tests for strategy base and built-in strategies."""

import numpy as np
import pandas as pd
import pytest

from massive.strategy.base import Strategy, Signal, TradeAction
from massive.strategy.examples import (
    SMACrossover, RSIMeanReversion, MACDStrategy, BollingerBandStrategy,
    BUILTIN_STRATEGIES,
)


@pytest.fixture
def sample_data():
    np.random.seed(99)
    n = 200
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.2,
            "high": close + abs(np.random.randn(n) * 0.5),
            "low": close - abs(np.random.randn(n) * 0.5),
            "close": close,
            "volume": np.random.randint(10000, 100000, n),
        },
        index=dates,
    )


class TestBuiltinStrategies:
    @pytest.mark.parametrize("name,cls", list(BUILTIN_STRATEGIES.items()))
    def test_setup_returns_dataframe(self, name, cls, sample_data):
        strategy = cls()
        result = strategy.setup(sample_data.copy())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

    @pytest.mark.parametrize("name,cls", list(BUILTIN_STRATEGIES.items()))
    def test_signals_length_matches_data(self, name, cls, sample_data):
        strategy = cls()
        data = strategy.setup(sample_data.copy())
        signals = strategy.generate_signals(data)
        assert len(signals) == len(sample_data)

    @pytest.mark.parametrize("name,cls", list(BUILTIN_STRATEGIES.items()))
    def test_signals_are_valid(self, name, cls, sample_data):
        strategy = cls()
        data = strategy.setup(sample_data.copy())
        signals = strategy.generate_signals(data)
        for s in signals:
            assert isinstance(s, Signal)
            assert isinstance(s.action, TradeAction)
            assert 0 <= s.strength <= 1

    def test_sma_crossover_generates_buys_and_sells(self, sample_data):
        strategy = SMACrossover(fast_period=5, slow_period=20)
        data = strategy.setup(sample_data.copy())
        signals = strategy.generate_signals(data)
        actions = {s.action for s in signals}
        assert TradeAction.BUY in actions or TradeAction.SELL in actions

    def test_param_override(self):
        s = SMACrossover(fast_period=7, slow_period=50)
        assert s.params["fast_period"] == 7
        assert s.params["slow_period"] == 50

    def test_registry_has_all(self):
        assert len(BUILTIN_STRATEGIES) == 4
        assert "sma_crossover" in BUILTIN_STRATEGIES
        assert "rsi_mean_reversion" in BUILTIN_STRATEGIES
        assert "macd" in BUILTIN_STRATEGIES
        assert "bollinger" in BUILTIN_STRATEGIES
