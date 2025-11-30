"""Market microstructure signal analyzing liquidity and activity."""

import logging
from typing import Dict, Any, Optional
from .base_signal import BaseSignal, SignalResult, clamp_prob, clamp_conf

logger = logging.getLogger(__name__)

class MicrostructureSignal(BaseSignal):
    """Signal based on market microstructure."""
    
    def __init__(self, name: str = "microstructure", weight: float = 0.30, enabled: bool = True):
        super().__init__(name, weight, enabled)
    
    def analyze(self, market: Dict[str, Any]) -> Optional[SignalResult]:
        current_prob = float(market.get("probability", 0.5))
        liquidity = float(market.get("totalLiquidity", 0) or 0)
        volume = float(market.get("volume", 0) or 0)
        recent_trades = market.get("recentTrades", [])  # if API returns

        # continuous mapping for confidence (smooth)
        conf = 0.30 + (1.0 - (1.0 / (1.0 + (liquidity / 200.0)))) * 0.6
        # conf roughly 0.4 @ 100, 0.6 @ 500, 0.75 @ 1000
        if volume > 1000:
            conf = min(0.95, conf + 0.08)

        # directional hint: small momentum estimator
        momentum = 0.0
        # if API has recent trades we can estimate direction: average signed trade flow
        try:
            buy_minus_sell = sum((1 if t["side"] == "buy" else -1) * float(t["size"]) for t in recent_trades)
            total_size = sum(float(t["size"]) for t in recent_trades) or 1.0
            momentum = (buy_minus_sell / total_size) * 0.02  # tiny tilt
        except Exception:
            # fallback: use volume vs liquidity heuristic
            if liquidity > 0:
                usage = min(3.0, volume / max(1.0, liquidity))
                momentum = (usage - 1.0) * 0.01  # small tilt if active trading

        # apply small tilt to probability but keep it conservative
        adjusted_prob = clamp_prob(current_prob + momentum)
        return SignalResult(
            probability=adjusted_prob,
            confidence=clamp_conf(conf),
            reasoning=f"liquidity={liquidity:.0f}, volume={volume:.0f}, momentum={momentum:+.3f}",
            metadata={"liquidity": liquidity, "volume": volume, "momentum": momentum}
        )
