"""Tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from massive_sm.backtester import Backtester
from massive_sm.strategy.examples import SMACrossover, RSIMeanReversion


@pytest.fixture
def sample_data():
    """Generate synthetic OHLCV data with a trend change for testing."""
    np.random.seed(123)
    n = 300
    # Oscillating trend to ensure crossovers happen
    t = np.arange(n)
    trend = 100 + 20 * np.sin(2 * np.pi * t / 60)  # cycle every 60 bars
    noise = np.random.randn(n) * 2
    close = trend + noise
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + abs(np.random.randn(n) * 1.0),
            "low": close - abs(np.random.randn(n) * 1.0),
            "close": close,
            "volume": np.random.randint(10000, 100000, n),
        },
        index=dates,
    )


class TestBacktester:
    def test_basic_run(self, sample_data):
        strategy = SMACrossover(fast_period=10, slow_period=30)
        bt = Backtester(strategy, initial_cash=100_000)
        result = bt.run(sample_data, "TEST")
        assert result.total_trades > 0
        assert result.final_equity > 0
        assert result.symbol == "TEST"

    def test_no_negative_equity(self, sample_data):
        strategy = SMACrossover(fast_period=5, slow_period=20)
        bt = Backtester(strategy, initial_cash=100_000)
        result = bt.run(sample_data, "TEST")
        assert (result.equity_curve > 0).all()

    def test_equity_curve_length(self, sample_data):
        strategy = RSIMeanReversion()
        bt = Backtester(strategy)
        result = bt.run(sample_data, "TEST")
        assert len(result.equity_curve) == len(sample_data)

    def test_summary_string(self, sample_data):
        strategy = SMACrossover()
        bt = Backtester(strategy)
        result = bt.run(sample_data, "TEST")
        summary = result.summary()
        assert "TEST" in summary
        assert "Total return" in summary

    def test_commission_impact(self, sample_data):
        strategy = SMACrossover(fast_period=5, slow_period=20)
        bt_low = Backtester(strategy, commission_pct=0.0)
        bt_high = Backtester(strategy, commission_pct=0.01)
        r_low = bt_low.run(sample_data, "TEST")
        r_high = bt_high.run(sample_data, "TEST")
        # Higher commission should result in lower or equal equity
        assert r_low.final_equity >= r_high.final_equity

    def test_sharpe_ratio_computed(self, sample_data):
        strategy = SMACrossover()
        bt = Backtester(strategy)
        result = bt.run(sample_data, "TEST")
        assert isinstance(result.sharpe_ratio, float)

    def test_max_drawdown_non_negative(self, sample_data):
        strategy = SMACrossover()
        bt = Backtester(strategy)
        result = bt.run(sample_data, "TEST")
        assert result.max_drawdown_pct >= 0
