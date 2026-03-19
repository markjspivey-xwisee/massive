"""Configuration loader."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_key: str = field(
        default_factory=lambda: os.getenv("MASSIVE_API_KEY", "") or os.getenv("MASSIVE", "")
    )
    api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MASSIVE_API_BASE_URL", "https://api.massive.com/v1"
        )
    )
    default_cash: float = 100_000.0
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage

    def validate(self):
        if not self.api_key:
            raise ValueError(
                "MASSIVE_API_KEY not set. Add it to .env or set the environment variable."
            )


config = Config()
