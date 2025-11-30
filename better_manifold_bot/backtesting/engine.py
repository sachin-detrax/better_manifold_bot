"""
Backtesting engine for evaluating trading strategies on historical markets.

This module can be contributed back to manifoldbot as it provides general-purpose
backtesting functionality that any trading bot would benefit from.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import statistics

from manifoldbot.manifold.bot import DecisionMaker, MarketDecision
from manifoldbot.manifold.api import ManifoldAPI

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a single trade in the backtest."""
    market_id: str
    question: str
    decision: str  # YES, NO, or SKIP
    bet_amount: float
    entry_probability: float
    predicted_probability: float
    confidence: float
    actual_outcome: Optional[bool]  # True=YES, False=NO, None=unresolved/canceled
    profit_loss: float = 0.0
    timestamp: Optional[str] = None
    reasoning: str = ""


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""

    # Basic metrics
    total_markets: int = 0
    markets_traded: int = 0
    markets_skipped: int = 0

    # Win/loss stats
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0

    # Financial metrics
    total_profit_loss: float = 0.0
    roi: float = 0.0  # Return on investment
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Calibration metrics
    calibration_score: float = 0.0  # Brier score
    average_confidence: float = 0.0

    # Detailed data
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Pretty print backtest results."""
        return f"""
╔══════════════════════════════════════════════════════════╗
║              BACKTEST RESULTS                            ║
╚══════════════════════════════════════════════════════════╝

📊 Trading Activity:
   Markets Analyzed:  {self.total_markets}
   Markets Traded:    {self.markets_traded}
   Markets Skipped:   {self.markets_skipped}
   Trade Rate:        {self.markets_traded/max(1, self.total_markets):.1%}

🎯 Win/Loss Stats:
   Wins:              {self.wins}
   Losses:            {self.losses}
   Win Rate:          {self.win_rate:.1%}

💰 Financial Performance:
   Total P&L:         ${self.total_profit_loss:+.2f}
   ROI:               {self.roi:+.1%}
   Sharpe Ratio:      {self.sharpe_ratio:.2f}
   Max Drawdown:      {self.max_drawdown:.1%}

📈 Calibration:
   Brier Score:       {self.calibration_score:.4f} (lower is better)
   Avg Confidence:    {self.average_confidence:.1%}

⚙️  Configuration:
   {self._format_config()}
"""

    def _format_config(self) -> str:
        """Format configuration dict for display."""
        return "\n   ".join(f"{k}: {v}" for k, v in self.config.items())


class BacktestEngine:
    """
    Engine for backtesting trading strategies on historical Manifold markets.

    This can be used to:
    1. Evaluate strategy performance on resolved markets
    2. Optimize parameters (min_confidence, min_edge, etc.)
    3. Compare different signal combinations
    4. Validate strategy improvements

    Example:
        >>> engine = BacktestEngine(api_key="your_key")
        >>> markets = engine.fetch_resolved_markets(creator="MikhailTal", limit=100)
        >>> result = engine.run_backtest(decision_maker, markets, bet_amount=10)
        >>> print(result)
    """

    def __init__(self, api_key: str):
        """
        Initialize backtest engine.

        Args:
            api_key: Manifold API key
        """
        self.api = ManifoldAPI(api_key)

    def fetch_resolved_markets(
        self,
        creator: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch resolved markets for backtesting.

        Args:
            creator: Filter by market creator username
            limit: Maximum number of markets to fetch
            filters: Additional filters for get_markets

        Returns:
            List of resolved market dictionaries
        """
        logger.info(f"Fetching resolved markets (creator={creator}, limit={limit})...")

        # Build filters
        market_filters = filters or {}

        # Add creator filter if specified
        if creator:
            try:
                user = self.api.get_user(creator)
                user_id = user.get("id")
                if user_id:
                    market_filters["userId"] = user_id
            except Exception as e:
                logger.warning(f"Could not resolve creator {creator}: {e}")

        # Fetch all markets (we'll filter for resolved ones)
        all_markets = self.api.get_markets(limit=limit * 2, filters=market_filters)

        # Filter for resolved binary markets only
        resolved_markets = [
            m for m in all_markets
            if m.get("isResolved") and m.get("outcomeType") == "BINARY"
        ][:limit]

        logger.info(f"Found {len(resolved_markets)} resolved binary markets")
        return resolved_markets

    def run_backtest(
        self,
        decision_maker: DecisionMaker,
        markets: List[Dict[str, Any]],
        bet_amount: float = 10.0,
        initial_balance: float = 1000.0,
        config: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest on a set of markets.

        Args:
            decision_maker: Strategy to test
            markets: List of resolved markets
            bet_amount: Amount to bet per trade
            initial_balance: Starting bankroll
            config: Configuration dict to store with results

        Returns:
            BacktestResult with complete performance metrics
        """
        logger.info(f"Running backtest on {len(markets)} markets...")

        result = BacktestResult(
            total_markets=len(markets),
            config=config or {}
        )

        balance = initial_balance
        equity_curve = [initial_balance]

        for market in markets:
            try:
                # Get the decision from the strategy
                decision = decision_maker.analyze_market(market)

                # Determine actual outcome
                resolution = market.get("resolution")
                if resolution == "YES":
                    actual_outcome = True
                elif resolution == "NO":
                    actual_outcome = False
                else:
                    # Skip markets that resolved to other outcomes (CANCEL, MKT, etc.)
                    continue

                # Create trade record
                trade = BacktestTrade(
                    market_id=market.get("id", ""),
                    question=market.get("question", ""),
                    decision=decision.decision,
                    bet_amount=bet_amount if decision.decision != "SKIP" else 0,
                    entry_probability=decision.current_probability,
                    predicted_probability=decision.metadata.get("ensemble_probability", decision.current_probability) if hasattr(decision, 'metadata') and decision.metadata else decision.current_probability,
                    confidence=decision.confidence,
                    actual_outcome=actual_outcome,
                    timestamp=market.get("createdTime"),
                    reasoning=decision.reasoning
                )

                # Calculate P&L for this trade
                if decision.decision == "SKIP":
                    result.markets_skipped += 1
                    trade.profit_loss = 0
                else:
                    result.markets_traded += 1
                    pnl = self._calculate_pnl(
                        decision.decision,
                        bet_amount,
                        decision.current_probability,
                        actual_outcome
                    )
                    trade.profit_loss = pnl
                    balance += pnl

                    # Track wins/losses
                    if pnl > 0:
                        result.wins += 1
                    else:
                        result.losses += 1

                result.trades.append(trade)
                equity_curve.append(balance)

            except Exception as e:
                logger.error(f"Error backtesting market {market.get('id')}: {e}")
                continue

        # Calculate aggregate metrics
        result.equity_curve = equity_curve
        result.total_profit_loss = balance - initial_balance
        result.win_rate = result.wins / max(1, result.markets_traded)
        result.roi = result.total_profit_loss / initial_balance

        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
            if returns and statistics.stdev(returns) > 0:
                result.sharpe_ratio = statistics.mean(returns) / statistics.stdev(returns) * (len(returns) ** 0.5)

        # Calculate max drawdown
        peak = initial_balance
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown = max_dd

        # Calculate calibration (Brier score)
        traded_predictions = [
            (t.predicted_probability, 1.0 if t.actual_outcome else 0.0)
            for t in result.trades
            if t.decision != "SKIP" and t.actual_outcome is not None
        ]

        if traded_predictions:
            brier_scores = [(pred - actual) ** 2 for pred, actual in traded_predictions]
            result.calibration_score = statistics.mean(brier_scores)
            result.average_confidence = statistics.mean([t.confidence for t in result.trades if t.decision != "SKIP"])

        logger.info(f"Backtest complete: {result.win_rate:.1%} win rate, ${result.total_profit_loss:+.2f} P&L")
        return result

    def _calculate_pnl(
        self,
        decision: str,
        bet_amount: float,
        market_probability: float,
        actual_outcome: bool
    ) -> float:
        """
        Calculate profit/loss for a trade using Manifold's LMSR rules.

        Simplified calculation:
        - If correct: Win proportional to odds
        - If wrong: Lose bet amount

        Args:
            decision: YES or NO
            bet_amount: Amount bet
            market_probability: Market price at time of bet
            actual_outcome: True if resolved YES, False if NO

        Returns:
            Profit or loss amount
        """
        bet_yes = (decision == "YES")

        # Determine if bet won
        won = (bet_yes and actual_outcome) or (not bet_yes and not actual_outcome)

        if won:
            # Winner gets payout based on odds
            if bet_yes:
                # Betting YES at probability p costs p per share, pays 1 if wins
                payout_per_share = 1.0
                shares = bet_amount / market_probability
                return shares * payout_per_share - bet_amount
            else:
                # Betting NO at probability p
                payout_per_share = 1.0
                shares = bet_amount / (1 - market_probability)
                return shares * payout_per_share - bet_amount
        else:
            # Lost the bet
            return -bet_amount

    def optimize_parameters(
        self,
        decision_maker_factory,
        markets: List[Dict[str, Any]],
        param_grid: Dict[str, List[Any]],
        bet_amount: float = 10.0,
        metric: str = "sharpe_ratio"
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Optimize parameters using grid search.

        Args:
            decision_maker_factory: Function that takes **params and returns DecisionMaker
            markets: Markets to backtest on
            param_grid: Dict mapping param names to lists of values to try
            bet_amount: Bet amount for each backtest
            metric: Metric to optimize ("sharpe_ratio", "win_rate", "roi", etc.)

        Returns:
            Tuple of (best_params, best_result)
        """
        logger.info("Starting parameter optimization...")

        import itertools

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        best_score = float('-inf')
        best_params = None
        best_result = None

        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            logger.info(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Create decision maker with these params
            dm = decision_maker_factory(**params)

            # Run backtest
            result = self.run_backtest(
                dm,
                markets,
                bet_amount=bet_amount,
                config=params
            )

            # Get metric value
            score = getattr(result, metric)

            logger.info(f"  {metric}: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params
                best_result = result
                logger.info(f"  ⭐ New best!")

        logger.info(f"\n🏆 Optimization complete!")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {metric}: {best_score:.4f}")

        return best_params, best_result
