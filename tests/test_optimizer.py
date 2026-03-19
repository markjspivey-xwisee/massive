"""Tests for the optimizer."""

import numpy as np
import pandas as pd
import pytest

from massive.optimizer import Optimizer
from massive.strategy.examples import SMACrossover


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 250
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


class TestOptimizer:
    def test_grid_search(self, sample_data):
        opt = Optimizer(SMACrossover, sample_data, "TEST")
        result = opt.grid_search({"fast_period": [5, 10], "slow_period": [20, 30]})
        assert len(result.all_results) == 4
        assert result.best_params is not None

    def test_random_search(self, sample_data):
        opt = Optimizer(SMACrossover, sample_data, "TEST")
        result = opt.random_search(
            {"fast_period": (5, 20), "slow_period": (20, 50)}, n_iter=10
        )
        assert len(result.all_results) == 10

    def test_top_n(self, sample_data):
        opt = Optimizer(SMACrossover, sample_data, "TEST")
        result = opt.grid_search({"fast_period": [5, 10, 15], "slow_period": [20, 30, 40]})
        top = result.top_n(3)
        assert len(top) == 3
        assert top[0]["metric_value"] >= top[1]["metric_value"]

    def test_walk_forward(self, sample_data):
        opt = Optimizer(SMACrossover, sample_data, "TEST")
        results = opt.walk_forward(
            {"fast_period": [5, 10], "slow_period": [20, 30]}, n_splits=3
        )
        assert len(results) > 0

    def test_summary(self, sample_data):
        opt = Optimizer(SMACrossover, sample_data, "TEST")
        result = opt.grid_search({"fast_period": [5, 10], "slow_period": [20, 30]})
        summary = result.summary()
        assert "Best" in summary
