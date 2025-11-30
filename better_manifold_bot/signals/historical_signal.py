"""Historical pattern signal based on creator's track record and market analysis."""

import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from better_manifold_bot.manifold.api import ManifoldAPI
from .base_signal import BaseSignal, SignalResult, clamp_prob, clamp_conf

logger = logging.getLogger(__name__)


class HistoricalSignal(BaseSignal):
    """
    Signal based on historical market patterns and creator track record.

    Improvements applied:
    - Laplace smoothing for low sample sizes
    - Bayesian shrinkage toward global yes-rate
    - Smarter confidence using sqrt(sample)
    - Keyword scoring instead of flat offsets
    """

    def __init__(
        self,
        name: str = "historical",
        weight: float = 0.30,
        enabled: bool = True,
        api_key: Optional[str] = None,
        cache_size: int = 200,
        target_creator: Optional[str] = "MikhailTal"
    ):
        super().__init__(name, weight, enabled)
        self.creator_stats = {}  # creator -> {"yes_rate": float, "count": int, "outcomes": list}
        self.loaded = False
        self.api_key = api_key or os.getenv("MANIFOLD_API_KEY")
        self.cache_size = cache_size
        self.target_creator = target_creator

        if self.api_key:
            self._load_historical_data()

    # --------------------------------------------------------
    # Load historical resolved markets
    # --------------------------------------------------------
    def _load_historical_data(self):
        if self.loaded:
            return

        try:
            logger.info(f"Loading historical data for {self.target_creator}...")
            api = ManifoldAPI(self.api_key)

            # Fetch markets by creator if specified
            # Reduced multiplier to avoid loading too much data upfront
            if self.target_creator:
                try:
                    all_markets = api.get_all_markets_by_creator(
                        username=self.target_creator,
                        limit=self.cache_size * 2  # Reduced from 10x to 2x
                    )
                except Exception as e:
                    logger.warning(f"Failed to fetch creator markets: {e}")
                    all_markets = api.get_markets(limit=self.cache_size)
            else:
                all_markets = api.get_markets(limit=self.cache_size)

            # Filter for resolved binary markets only
            resolved = [
                m for m in all_markets
                if m.get("isResolved")
                and m.get("outcomeType") == "BINARY"
                and m.get("resolution") in ["YES", "NO"]
            ]

            creator_outcomes = defaultdict(list)
            for m in resolved:
                creator = m.get("creatorUsername", "unknown")
                res = 1.0 if m.get("resolution") == "YES" else 0.0
                creator_outcomes[creator].append(res)

            # Build stats
            for creator, outcomes in creator_outcomes.items():
                if len(outcomes) >= 5:
                    self.creator_stats[creator] = {
                        "yes_rate": sum(outcomes) / len(outcomes),
                        "count": len(outcomes),
                        "outcomes": outcomes[-20:]
                    }

            self.loaded = True
            logger.info(f"Historical stats loaded for {len(self.creator_stats)} creators.")

        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")
            self.loaded = True

    # --------------------------------------------------------
    # Main analysis logic
    # --------------------------------------------------------
    def analyze(self, market: Dict[str, Any]) -> Optional[SignalResult]:

        # Only load historical data once, on first use
        # Removed redundant check that was causing reloads
        creator = market.get("creatorUsername", "")
        current_prob = float(market.get("probability", 0.5))
        question = market.get("question", "").lower()

        # -------- GLOBAL YES RATE (fallback baseline) --------
        global_yes_rate = 0.5
        if self.creator_stats:
            counts = [s["count"] for s in self.creator_stats.values()]
            yes_rates = [s["yes_rate"] for s in self.creator_stats.values()]
            total = sum(counts) or 1
            global_yes_rate = sum(r * c for r, c in zip(yes_rates, counts)) / total

        # Start values
        adjusted_prob = current_prob
        confidence = 0.50
        reasoning_parts = []

        # -------- CREATOR-BASED ADJUSTMENTS --------
        if creator in self.creator_stats:
            stats = self.creator_stats[creator]
            sample = stats["count"]
            raw_rate = stats["yes_rate"]

            # Laplace smoothing
            alpha = 2
            smoothed = (raw_rate * sample + global_yes_rate * alpha) / (sample + alpha)

            # Bayesian shrinkage using sqrt(sample)
            shrink_weight = min(0.6, (sample ** 0.5) / 10.0)

            adjusted_prob = (
                (1 - shrink_weight) * current_prob +
                shrink_weight * smoothed
            )

            # Confidence using sqrt scaling
            confidence = 0.45 + min(0.4, (sample ** 0.5) / 10.0)

            reasoning_parts.append(
                f"{creator}: raw_rate={raw_rate:.2f}, smoothed={smoothed:.2f}, n={sample}, w={shrink_weight:.2f}"
            )

        # -------- KEYWORD SCORING --------
        keyword_effect = 0.0
        keywords = {
            "will": 0.01, "likely": 0.02, "expected": 0.02, "predicted": 0.02,
            "won't": -0.03, "unlikely": -0.02, "never": -0.02, "impossible": -0.03
        }

        for k, v in keywords.items():
            if k in question:
                keyword_effect += v

        if keyword_effect != 0.0:
            adjusted_prob = clamp_prob(adjusted_prob + keyword_effect)
            reasoning_parts.append(f"keyword adj {keyword_effect:+.2f}")

        # -------- CLAMPING --------
        adjusted_prob = clamp_prob(adjusted_prob)
        confidence = clamp_conf(confidence)

        # -------- REASONING --------
        if reasoning_parts:
            reasoning = "Historical: " + "; ".join(reasoning_parts)
        else:
            reasoning = "Historical: no creator data, used market probability"

        return SignalResult(
            probability=adjusted_prob,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "creator": creator,
                "has_creator_data": creator in self.creator_stats,
                "adjustment": adjusted_prob - current_prob
            }
        )
