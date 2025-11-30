"""
Enhanced bot with Kelly Criterion bet sizing.

Extends ManifoldBot to use optimal position sizing based on edge and confidence.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from manifoldbot.manifold.bot import ManifoldBot, MarketDecision
from .kelly_criterion import KellyCalculator

logger = logging.getLogger(__name__)


class KellyBot(ManifoldBot):
    """
    Bot with Kelly Criterion bet sizing.

    Instead of fixed bet amounts, this bot calculates optimal position sizes
    based on edge, confidence, and current bankroll.
    """

    def __init__(
        self,
        manifold_api_key: str,
        decision_maker,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.10,
        min_bet: float = 1.0,
        max_bet: float = 100.0,
        dry_run: bool = False
    ):
        """
        Initialize Kelly bot.

        Args:
            manifold_api_key: Manifold API key
            decision_maker: Decision maker strategy
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, recommended)
            max_bet_fraction: Maximum fraction of bankroll per bet
            min_bet: Minimum bet size
            max_bet: Maximum bet size
            dry_run: If True, simulate bets without placing them
        """
        super().__init__(manifold_api_key, decision_maker)
        self.kelly_calculator = KellyCalculator(
            kelly_fraction=kelly_fraction,
            max_bet_fraction=max_bet_fraction
        )
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.dry_run = dry_run

    def calculate_bet_amount(self, decision: MarketDecision) -> float:
        """
        Calculate optimal bet amount using Kelly Criterion.

        Args:
            decision: Market decision with probability estimates

        Returns:
            Optimal bet amount in dollars
        """
        if decision.decision == "SKIP":
            return 0.0

        # Get current bankroll
        bankroll = self.writer.get_balance()

        # Get probability estimates
        predicted_prob = decision.metadata.get("ensemble_probability", decision.current_probability)
        market_prob = decision.current_probability
        bet_on_yes = (decision.decision == "YES")

        # Calculate Kelly bet size
        bet_size = self.kelly_calculator.calculate_bet_size(
            bankroll=bankroll,
            predicted_prob=predicted_prob,
            market_prob=market_prob,
            bet_on_yes=bet_on_yes,
            min_bet=self.min_bet,
            max_bet=self.max_bet
        )

        logger.info(
            f"Kelly sizing: Bankroll=${bankroll:.2f}, "
            f"Edge={abs(predicted_prob-market_prob):.1%}, "
            f"Bet=${bet_size:.2f}"
        )

        return bet_size

    def place_bet_if_decision(
        self,
        decision: MarketDecision,
        bet_amount: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Place bet with Kelly-sized amount.

        Args:
            decision: Market decision
            bet_amount: Override amount (if None, uses Kelly calculation)

        Returns:
            Tuple of (success: bool, actual_bet_amount: float)
        """
        if decision.decision == "SKIP":
            return False, 0.0

        # Use Kelly sizing if no amount specified
        if bet_amount is None:
            bet_amount = self.calculate_bet_amount(decision)

        # Don't bet if Kelly says bet is too small
        if bet_amount < self.min_bet:
            logger.info(f"Kelly bet size ${bet_amount:.2f} below minimum ${self.min_bet}, skipping")
            return False, 0.0

        # Prepare data for tracking
        market_dict = {
            'id': decision.market_id,
            'question': decision.question,
            'probability': decision.current_probability
        }
        
        # Get current balance
        current_balance = self.writer.get_balance()

        if self.dry_run:
            logger.info(f"[DRY RUN] Would place bet: ${bet_amount:.2f} on {decision.decision}")
            
            # Track the simulated bet
            if hasattr(self, 'tracker') and self.tracker:
                self.tracker.log_bet(
                    market=market_dict,
                    decision=decision,
                    bet_amount=bet_amount,
                    current_balance=current_balance,
                    signal_results=getattr(self, 'last_signal_results', [])
                )
                
                # Update live dashboard if available
                if hasattr(self, 'dashboard') and self.dashboard:
                    stats = self.tracker.get_session_stats()
                    self.dashboard.show_live_session_stats(stats, current_balance)
            
            return True, bet_amount

        # Place the bet
        success = super().place_bet_if_decision(decision, bet_amount)
        
        if success:
            # Track the real bet
            if hasattr(self, 'tracker') and self.tracker:
                # Update balance after bet
                new_balance = self.writer.get_balance()
                self.tracker.log_bet(
                    market=market_dict,
                    decision=decision,
                    bet_amount=bet_amount,
                    current_balance=new_balance,
                    signal_results=getattr(self, 'last_signal_results', [])
                )
                
                # Update live dashboard if available
                if hasattr(self, 'dashboard') and self.dashboard:
                    stats = self.tracker.get_session_stats()
                    self.dashboard.show_live_session_stats(stats, new_balance)

        return success, bet_amount if success else 0.0
