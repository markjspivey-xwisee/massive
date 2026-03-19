"""Parameter optimizer — grid search and random search over strategy params."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Type

import pandas as pd

from massive.backtester import BacktestResult, Backtester
from massive.strategy.base import Strategy


@dataclass
class OptimizationResult:
    best_params: dict
    best_metric_value: float
    metric: str
    all_results: list[dict]

    def top_n(self, n: int = 10) -> list[dict]:
        return sorted(
            self.all_results, key=lambda r: r["metric_value"], reverse=True
        )[:n]

    def summary(self) -> str:
        lines = [
            f"═══ Optimization Results ═══",
            f"Metric:       {self.metric}",
            f"Best value:   {self.best_metric_value:.4f}",
            f"Best params:  {self.best_params}",
            f"Combos tried: {len(self.all_results)}",
            "",
            "Top 5:",
        ]
        for i, r in enumerate(self.top_n(5), 1):
            lines.append(f"  {i}. {r['params']}  →  {r['metric_value']:.4f}")
        return "\n".join(lines)


class Optimizer:
    """Optimize strategy parameters via grid or random search."""

    def __init__(
        self,
        strategy_class: Type[Strategy],
        data: pd.DataFrame,
        symbol: str = "UNKNOWN",
        metric: str = "sharpe_ratio",
        initial_cash: float | None = None,
    ):
        self.strategy_class = strategy_class
        self.data = data
        self.symbol = symbol
        self.metric = metric
        self.initial_cash = initial_cash

    def grid_search(self, param_grid: dict[str, list]) -> OptimizationResult:
        """Exhaustive grid search over all parameter combinations.

        Args:
            param_grid: e.g. {"fast_period": [5, 10, 20], "slow_period": [20, 30, 50]}
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        return self._search([dict(zip(keys, c)) for c in combos])

    def random_search(
        self, param_ranges: dict[str, tuple], n_iter: int = 100
    ) -> OptimizationResult:
        """Random search over parameter ranges.

        Args:
            param_ranges: e.g. {"fast_period": (5, 50), "slow_period": (20, 100)}
                          Values are (min, max) for int sampling.
            n_iter: Number of random samples.
        """
        combos = []
        for _ in range(n_iter):
            params = {}
            for key, (lo, hi) in param_ranges.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[key] = random.randint(lo, hi)
                else:
                    params[key] = round(random.uniform(lo, hi), 4)
            combos.append(params)
        return self._search(combos)

    def walk_forward(
        self,
        param_grid: dict[str, list],
        n_splits: int = 5,
        train_pct: float = 0.7,
    ) -> list[BacktestResult]:
        """Walk-forward optimization: optimize on in-sample, test on out-of-sample.

        Splits data into n_splits sequential windows. For each window, optimizes
        on the first train_pct of data and tests on the remaining.
        """
        total_len = len(self.data)
        window_size = total_len // n_splits
        results: list[BacktestResult] = []

        for i in range(n_splits):
            start = i * window_size
            end = min(start + window_size, total_len)
            window = self.data.iloc[start:end]

            split = int(len(window) * train_pct)
            train_data = window.iloc[:split]
            test_data = window.iloc[split:]

            if len(train_data) < 20 or len(test_data) < 5:
                continue

            # Optimize on train
            train_opt = Optimizer(
                self.strategy_class, train_data, self.symbol,
                self.metric, self.initial_cash,
            )
            opt_result = train_opt.grid_search(param_grid)

            # Test on out-of-sample
            strategy = self.strategy_class(**opt_result.best_params)
            bt = Backtester(strategy, initial_cash=self.initial_cash)
            result = bt.run(test_data, self.symbol)
            results.append(result)

        return results

    def _search(self, param_combos: list[dict]) -> OptimizationResult:
        all_results: list[dict] = []
        best_value = float("-inf")
        best_params: dict = {}

        for params in param_combos:
            try:
                strategy = self.strategy_class(**params)
                bt = Backtester(strategy, initial_cash=self.initial_cash)
                result = bt.run(self.data, self.symbol)
                value = getattr(result, self.metric)
                all_results.append({"params": params, "metric_value": value, "result": result})
                if value > best_value:
                    best_value = value
                    best_params = params
            except Exception:
                continue

        return OptimizationResult(
            best_params=best_params,
            best_metric_value=best_value,
            metric=self.metric,
            all_results=all_results,
        )
