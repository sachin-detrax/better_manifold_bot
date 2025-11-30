"""
Kelly Criterion bet sizing for optimal capital allocation.

The Kelly Criterion maximizes long-term growth rate by determining the optimal
fraction of bankroll to wager based on edge and odds.

This module can be contributed to manifoldbot as it provides general-purpose
bet sizing functionality.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KellyCalculator:
    """
    Calculate optimal bet sizes using the Kelly Criterion.

    The Kelly Criterion formula for binary outcomes:
        f* = (bp - q) / b

    Where:
        f* = fraction of bankroll to wager
        b = net odds received (payout/stake - 1)
        p = probability of winning
        q = probability of losing (1-p)

    Example:
        >>> calc = KellyCalculator(kelly_fraction=0.25)  # Use 25% Kelly
        >>> bet = calc.calculate_bet_size(
        ...     bankroll=1000,
        ...     predicted_prob=0.70,
        ...     market_prob=0.50,
        ...     bet_on_yes=True
        ... )
        >>> print(f"Optimal bet: ${bet:.2f}")
    """

    def __init__(self, kelly_fraction: float = 0.25, max_bet_fraction: float = 0.10):
        """
        Initialize Kelly calculator.

        Args:
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
                           Fractional Kelly is more conservative and recommended
            max_bet_fraction: Maximum fraction of bankroll to bet (safety cap)
        """
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction

    def calculate_bet_size(
        self,
        bankroll: float,
        predicted_prob: float,
        market_prob: float,
        bet_on_yes: bool,
        min_bet: float = 1.0,
        max_bet: Optional[float] = None
    ) -> float:
        """
        Calculate optimal bet size using Kelly Criterion.

        Args:
            bankroll: Current bankroll
            predicted_prob: Our estimated probability of YES outcome (0.0 to 1.0)
            market_prob: Market's current probability (0.0 to 1.0)
            bet_on_yes: True if betting YES, False if betting NO
            min_bet: Minimum bet size
            max_bet: Maximum bet size (overrides kelly if set)

        Returns:
            Optimal bet size in dollars
        """
        # Validate inputs
        if not (0 < predicted_prob < 1):
            logger.warning(f"Invalid predicted_prob: {predicted_prob}")
            return min_bet

        if not (0 < market_prob < 1):
            logger.warning(f"Invalid market_prob: {market_prob}")
            return min_bet

        # Calculate our edge
        edge = self._calculate_edge(predicted_prob, market_prob, bet_on_yes)

        if edge <= 0:
            logger.debug(f"No edge detected: {edge:.4f}")
            return 0.0  # No bet if no edge

        # Calculate odds received
        odds = self._calculate_odds(market_prob, bet_on_yes)

        # Kelly formula: f* = (edge / odds)
        # More precisely: f* = (odds * p - q) / odds = ((odds + 1) * p - 1) / odds
        if bet_on_yes:
            p_win = predicted_prob
        else:
            p_win = 1 - predicted_prob

        # Full Kelly percentage
        kelly_pct = ((odds + 1) * p_win - 1) / odds

        # Apply fractional Kelly (more conservative)
        kelly_pct *= self.kelly_fraction

        # Cap at max bet fraction
        kelly_pct = min(kelly_pct, self.max_bet_fraction)

        # Convert to dollar amount
        bet_size = kelly_pct * bankroll

        # Apply min/max constraints
        bet_size = max(bet_size, min_bet)
        if max_bet is not None:
            bet_size = min(bet_size, max_bet)

        logger.debug(
            f"Kelly calculation: edge={edge:.1%}, odds={odds:.2f}, "
            f"kelly_pct={kelly_pct:.1%}, bet=${bet_size:.2f}"
        )

        return round(bet_size, 2)

    def _calculate_edge(
        self,
        predicted_prob: float,
        market_prob: float,
        bet_on_yes: bool
    ) -> float:
        """
        Calculate our edge in the bet.

        Edge is the difference between our estimated probability and the implied
        probability from market odds.

        Args:
            predicted_prob: Our probability of YES
            market_prob: Market probability of YES
            bet_on_yes: Direction of bet

        Returns:
            Edge as a decimal (0.1 = 10% edge)
        """
        if bet_on_yes:
            # Betting YES: we have edge if our prob > market prob
            edge = predicted_prob - market_prob
        else:
            # Betting NO: we have edge if market thinks YES more than we do
            # If we think YES prob is 0.4, we think NO prob is 0.6
            # If market thinks YES prob is 0.5, it thinks NO prob is 0.5
            # Our edge on NO is: (1 - predicted_prob) - (1 - market_prob)
            edge = market_prob - predicted_prob

        return edge

    def _calculate_odds(self, market_prob: float, bet_on_yes: bool) -> float:
        """
        Calculate net odds received on a bet in a LMSR market.

        In Manifold's LMSR market:
        - Buying YES at price p costs p per share, pays 1 if wins
        - Net odds = (payout - cost) / cost = (1 - p) / p

        Args:
            market_prob: Market probability (price)
            bet_on_yes: Betting YES or NO

        Returns:
            Net odds (payout/stake - 1)
        """
        if bet_on_yes:
            # Buying YES at price p: net odds = (1-p)/p
            odds = (1 - market_prob) / market_prob
        else:
            # Buying NO at price p: net odds = p/(1-p)
            odds = market_prob / (1 - market_prob)

        return odds

    def calculate_recommended_thresholds(
        self,
        min_kelly_pct: float = 0.02
    ) -> tuple[float, float]:
        """
        Calculate recommended min_confidence and min_edge thresholds.

        These ensure we only bet when Kelly suggests at least min_kelly_pct of bankroll.

        Args:
            min_kelly_pct: Minimum Kelly percentage to trigger a bet

        Returns:
            Tuple of (recommended_min_confidence, recommended_min_edge)
        """
        # For a bet to be worth min_kelly_pct of bankroll:
        # kelly_pct * kelly_fraction >= min_kelly_pct
        # So we need: (edge / odds) >= min_kelly_pct / kelly_fraction

        # Conservative recommendation:
        # At minimum edge, we want significant confidence
        recommended_min_edge = min_kelly_pct / self.kelly_fraction
        recommended_min_confidence = 0.65  # Always require at least 65% confidence

        logger.info(
            f"Recommended thresholds for min_kelly={min_kelly_pct:.1%}: "
            f"min_edge={recommended_min_edge:.1%}, min_confidence={recommended_min_confidence:.1%}"
        )

        return recommended_min_confidence, recommended_min_edge


def calculate_position_size(
    bankroll: float,
    predicted_prob: float,
    market_prob: float,
    bet_on_yes: bool,
    kelly_fraction: float = 0.25,
    max_bet: Optional[float] = None
) -> float:
    """
    Convenience function to calculate position size.

    Args:
        bankroll: Current bankroll
        predicted_prob: Estimated probability of YES
        market_prob: Market probability
        bet_on_yes: Betting YES or NO
        kelly_fraction: Fraction of Kelly to use
        max_bet: Maximum bet size

    Returns:
        Recommended bet size
    """
    calc = KellyCalculator(kelly_fraction=kelly_fraction)
    return calc.calculate_bet_size(
        bankroll=bankroll,
        predicted_prob=predicted_prob,
        market_prob=market_prob,
        bet_on_yes=bet_on_yes,
        max_bet=max_bet
    )
