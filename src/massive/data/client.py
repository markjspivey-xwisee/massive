"""Massive.com Stock Data API client."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from massive.config import config


class MassiveClient:
    """Client for the Massive.com stock data API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or config.api_key
        self.base_url = (base_url or config.api_base_url).rstrip("/")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            }
        )

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Market Data ──────────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start date (default: 1 year ago).
            end: End date (default: today).
            interval: Bar interval — "1m", "5m", "15m", "1h", "1d", "1wk".

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        data = self._get(
            "/market/bars",
            params={
                "symbol": symbol.upper(),
                "start": str(start),
                "end": str(end),
                "interval": interval,
            },
        )

        df = pd.DataFrame(data["bars"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_quote(self, symbol: str) -> dict:
        """Get a real-time quote for a symbol."""
        return self._get("/market/quote", params={"symbol": symbol.upper()})

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Batch real-time quotes."""
        return self._get(
            "/market/quotes",
            params={"symbols": ",".join(s.upper() for s in symbols)},
        )["quotes"]

    # ── Screening / Fundamentals ─────────────────────────────────

    def screen(self, filters: dict) -> list[dict]:
        """Run a stock screener with the given filters.

        Example filters:
            {"market_cap_min": 1e9, "sector": "Technology", "pe_max": 30}
        """
        return self._get("/screener", params=filters)["results"]

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch fundamental data (PE, EPS, market cap, etc.)."""
        return self._get(
            "/fundamentals", params={"symbol": symbol.upper()}
        )

    # ── Universe helpers ─────────────────────────────────────────

    def get_sp500(self) -> list[str]:
        """Return current S&P 500 constituent tickers."""
        return self._get("/universe/sp500")["symbols"]

    def search(self, query: str) -> list[dict]:
        """Search for symbols by name or ticker."""
        return self._get("/search", params={"q": query})["results"]
