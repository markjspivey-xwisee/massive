"""Wrapper around the official Massive.com Python SDK (massive>=2.0)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
from massive import RESTClient

from massive_sm.config import config

# Map user-friendly intervals to (multiplier, timespan) for the Massive API
_INTERVAL_MAP = {
    "1m": (1, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
    "1wk": (1, "week"),
}


class MassiveClient:
    """Thin wrapper over the official Massive RESTClient.

    Provides pandas-friendly helpers used by the backtester, scanner, and trader.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or config.api_key
        config.validate()
        self._client = RESTClient(api_key=key)

    # ── Market Data ──────────────────────────────────────────────

    def get_bars(
        self,
        symbol: str,
        start: str | date | None = None,
        end: str | date | None = None,
        interval: str = "1d",
        limit: int = 50_000,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            start: Start date (default: 1 year ago).
            end: End date (default: today).
            interval: "1m", "5m", "15m", "1h", "1d", "1wk".
            limit: Max bars per API page.

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        multiplier, timespan = _INTERVAL_MAP.get(interval, (1, "day"))

        aggs = list(
            self._client.list_aggs(
                ticker=symbol.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=str(start),
                to=str(end),
                limit=limit,
            )
        )

        if not aggs:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        records = []
        for a in aggs:
            records.append(
                {
                    "timestamp": pd.to_datetime(a.timestamp, unit="ms"),
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                }
            )

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        return df

    def get_quote(self, symbol: str) -> dict:
        """Get the most recent quote for a symbol."""
        q = self._client.get_last_quote(ticker=symbol.upper())
        return {
            "symbol": symbol.upper(),
            "bid": q.bid_price,
            "ask": q.ask_price,
            "price": (q.bid_price + q.ask_price) / 2,
            "bid_size": q.bid_size,
            "ask_size": q.ask_size,
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Get the most recent quote for each symbol."""
        return [self.get_quote(s) for s in symbols]

    def get_snapshot(self, symbol: str) -> dict:
        """Get a full market snapshot for a symbol."""
        snap = self._client.get_snapshot_ticker(
            market_type="stocks", ticker=symbol.upper()
        )
        t = snap.ticker
        return {
            "symbol": t.ticker,
            "day_open": t.day.open if t.day else None,
            "day_high": t.day.high if t.day else None,
            "day_low": t.day.low if t.day else None,
            "day_close": t.day.close if t.day else None,
            "day_volume": t.day.volume if t.day else None,
            "prev_close": t.prev_day.close if t.prev_day else None,
            "last_price": t.last_trade.price if t.last_trade else None,
            "change_pct": t.todays_change_percent,
        }

    def get_previous_close(self, symbol: str) -> dict:
        """Get previous day's OHLCV."""
        agg = self._client.get_previous_close_agg(ticker=symbol.upper())
        a = agg[0] if agg else None
        if a is None:
            return {}
        return {
            "symbol": symbol.upper(),
            "open": a.open,
            "high": a.high,
            "low": a.low,
            "close": a.close,
            "volume": a.volume,
        }

    # ── Technical Indicators (server-side) ───────────────────────

    def get_sma(
        self, symbol: str, window: int = 50, timespan: str = "day"
    ) -> pd.DataFrame:
        """Fetch server-side SMA from the API."""
        results = list(
            self._client.get_sma(
                ticker=symbol.upper(),
                timespan=timespan,
                window=window,
            )
        )
        if not results:
            return pd.DataFrame(columns=["timestamp", "sma"])
        records = [{"timestamp": r.timestamp, "sma": r.value} for r in results]
        return pd.DataFrame(records).set_index("timestamp").sort_index()

    def get_rsi(
        self, symbol: str, window: int = 14, timespan: str = "day"
    ) -> pd.DataFrame:
        """Fetch server-side RSI from the API."""
        results = list(
            self._client.get_rsi(
                ticker=symbol.upper(),
                timespan=timespan,
                window=window,
            )
        )
        if not results:
            return pd.DataFrame(columns=["timestamp", "rsi"])
        records = [{"timestamp": r.timestamp, "rsi": r.value} for r in results]
        return pd.DataFrame(records).set_index("timestamp").sort_index()

    def get_macd(
        self,
        symbol: str,
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9,
        timespan: str = "day",
    ) -> pd.DataFrame:
        """Fetch server-side MACD from the API."""
        results = list(
            self._client.get_macd(
                ticker=symbol.upper(),
                timespan=timespan,
                short_window=short_window,
                long_window=long_window,
                signal_window=signal_window,
            )
        )
        if not results:
            return pd.DataFrame(columns=["timestamp", "macd", "signal", "histogram"])
        records = [
            {
                "timestamp": r.timestamp,
                "macd": r.value,
                "signal": r.signal,
                "histogram": r.histogram,
            }
            for r in results
        ]
        return pd.DataFrame(records).set_index("timestamp").sort_index()

    def get_ema(
        self, symbol: str, window: int = 50, timespan: str = "day"
    ) -> pd.DataFrame:
        """Fetch server-side EMA from the API."""
        results = list(
            self._client.get_ema(
                ticker=symbol.upper(),
                timespan=timespan,
                window=window,
            )
        )
        if not results:
            return pd.DataFrame(columns=["timestamp", "ema"])
        records = [{"timestamp": r.timestamp, "ema": r.value} for r in results]
        return pd.DataFrame(records).set_index("timestamp").sort_index()

    # ── Fundamentals / Reference ─────────────────────────────────

    def get_fundamentals(self, symbol: str) -> dict:
        """Fetch company financials (most recent income statement ratios)."""
        ratios = list(
            self._client.list_financials_ratios(ticker=symbol.upper(), limit=1)
        )
        if ratios:
            r = ratios[0]
            return {
                "symbol": symbol.upper(),
                "pe_ratio": getattr(r, "price_to_earnings_ratio", None),
                "pb_ratio": getattr(r, "price_to_book_ratio", None),
                "roe": getattr(r, "return_on_equity", None),
                "debt_to_equity": getattr(r, "debt_to_equity_ratio", None),
            }
        return {"symbol": symbol.upper()}

    def get_ticker_details(self, symbol: str) -> dict:
        """Fetch ticker details (name, market cap, sector, etc.)."""
        d = self._client.get_ticker_details(ticker=symbol.upper())
        return {
            "symbol": d.ticker,
            "name": d.name,
            "market_cap": d.market_cap,
            "sector": getattr(d, "sic_description", None),
            "locale": d.locale,
            "primary_exchange": d.primary_exchange,
        }

    def get_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """Fetch recent news for a ticker."""
        articles = list(
            self._client.list_ticker_news(ticker=symbol.upper(), limit=limit)
        )
        return [
            {
                "title": a.title,
                "url": a.article_url,
                "published": a.published_utc,
                "source": getattr(a, "publisher", {}).get("name", "")
                if isinstance(getattr(a, "publisher", None), dict)
                else str(getattr(a, "publisher", "")),
            }
            for a in articles
        ]

    # ── Universe helpers ─────────────────────────────────────────

    def list_tickers(
        self, market: str = "stocks", active: bool = True, limit: int = 1000
    ) -> list[str]:
        """List all active tickers in a market."""
        tickers = list(
            self._client.list_tickers(market=market, active=active, limit=limit)
        )
        return [t.ticker for t in tickers]

    def search(self, query: str) -> list[dict]:
        """Search for tickers by name or symbol."""
        tickers = list(
            self._client.list_tickers(search=query, active=True, limit=20)
        )
        return [
            {"symbol": t.ticker, "name": t.name, "market": t.market}
            for t in tickers
        ]
