"""
Main script for ensemble trading bot.

Uses manifoldbot framework with our ensemble decision maker.
"""

import os
import argparse
import logging
import time
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from manifoldbot.manifold.bot import ManifoldBot, TradingSession
from better_manifold_bot.ensemble_decision_maker import EnsembleDecisionMaker
from better_manifold_bot.signals.llm_signal import LLMSignal
from better_manifold_bot.signals.historical_signal import HistoricalSignal
from better_manifold_bot.signals.microstructure_signal import MicrostructureSignal
from better_manifold_bot.kelly_bot import KellyBot
from better_manifold_bot.performance_tracker import PerformanceTracker
from better_manifold_bot.dashboard import Dashboard, create_performance_report
from better_manifold_bot.visualizations import PerformanceVisualizer

console = Console()
logger = logging.getLogger(__name__)

class BetterManifoldBot(ManifoldBot):
    """Subclass of ManifoldBot with optimized market fetching and better logging."""

    def run_on_markets(
        self,
        markets: List[Dict[str, Any]],
        bet_amount: Optional[int] = None,
        max_bets: int = 5,
        delay_between_bets: float = 1.0
    ) -> TradingSession:
        """
        Run the bot on a list of markets.
        Overridden to show FULL question without truncation.

        Args:
            bet_amount: If None, uses Kelly sizing. If set, overrides Kelly (not recommended).
        """
        decisions = []
        bets_placed = 0
        errors = []
        initial_balance = self.writer.get_balance()

        self.logger.info(f"Analyzing {len(markets)} markets...")

        for i, market in enumerate(markets):
            if bets_placed >= max_bets:
                self.logger.info(f"Reached maximum bets limit ({max_bets})")
                break

            # SHOW FULL QUESTION HERE - No truncation
            question = market.get("question", "")
            self.logger.info(f"Analyzing market {i+1}/{len(markets)}: {question}")

            try:
                # Analyze market
                decision = self.analyze_market(market)
                decisions.append(decision)

                # Log decision with more details
                self.logger.info(
                    f"Decision: {decision.decision} | "
                    f"Type: {decision.outcome_type} | "
                    f"Current: {decision.current_probability:.1%} | "
                    f"Confidence: {decision.confidence:.1%}"
                )
                self.logger.info(f"  Reasoning: {decision.reasoning}")

                if decision.decision != "SKIP":
                    # Place bet and get actual amount
                    success, actual_bet_amount = self.place_bet_if_decision(decision, bet_amount)
                    
                    if success:
                        bets_placed += 1

                        # Always log bet to tracker when placed
                        if self.tracker:
                            current_balance = self.writer.get_balance()
                            self.tracker.log_bet(
                                market=market,
                                decision=decision,
                                bet_amount=actual_bet_amount,
                                current_balance=current_balance,
                                signal_results=getattr(self, 'last_signal_results', [])
                            )

                        # Show live stats
                        if self.dashboard and self.tracker:
                            stats = self.tracker.get_session_stats()
                            current_balance = self.writer.get_balance()
                            self.dashboard.show_live_session_stats(stats, current_balance)

                        time.sleep(delay_between_bets)  # Rate limiting
                        
            except Exception as e:
                error_msg = f"Error analyzing market {market.get('id', 'unknown')}: {e}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        final_balance = self.writer.get_balance()
        
        return TradingSession(
            markets_analyzed=len(decisions),
            bets_placed=bets_placed,
            initial_balance=initial_balance,
            final_balance=final_balance,
            decisions=decisions,
            errors=errors
        )

    def run_on_user_markets(
        self,
        username: str = "MikhailTal",
        limit: int = 20,
        bet_amount: Optional[int] = None,
        max_bets: int = 5,
        delay_between_bets: float = 1.0
    ) -> TradingSession:
        """
        Run the bot on markets created by a specific user.
        Optimized to use get_markets with userId filter.
        """
        self.logger.info(f"Getting markets created by user: {username}")
        
        try:
            # First resolve username to ID
            try:
                user = self.reader.get_user(username)
                user_id = user.get("id")
                if not user_id:
                    raise ValueError(f"Could not find user ID for {username}")
                self.logger.info(f"Resolved {username} to ID: {user_id}")
            except Exception as e:
                self.logger.warning(f"Failed to resolve username {username}: {e}")
                # Fallback to using username directly if get_user fails (maybe it is an ID?)
                user_id = username

            # Use get_markets with userId filter which is reliable
            self.logger.info(f"Fetching {limit} markets from {username} (ID: {user_id})...")
            markets = self.reader.get_markets(limit=limit, filters={"userId": user_id})
            
            self.logger.info(f"Found {len(markets)} markets created by {username}")
            
            return self.run_on_markets(markets, bet_amount, max_bets, delay_between_bets)
            
        except Exception as e:
            error_msg = f"Error getting markets for user {username}: {e}"
            self.logger.error(error_msg)
            return TradingSession(
                markets_analyzed=0,
                bets_placed=0,
                initial_balance=self.writer.get_balance(),
                final_balance=self.writer.get_balance(),
                decisions=[],
                errors=[error_msg]
            )

def main():
    # Fix encoding issues on Windows
    import sys
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Ensemble Manifold Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--limit", type=int, default=10, help="Number of markets to analyze")
    parser.add_argument("--bet-amount", type=int, default=10, help="Fixed bet amount (overrides Kelly)")
    parser.add_argument("--show-report", action="store_true", help="Show performance report")
    parser.add_argument("--generate-graphs", action="store_true", help="Generate performance graphs")
    args = parser.parse_args()

    load_dotenv()

    console.print(Panel.fit(
        "[bold cyan] Ensemble Super-Forecaster Bot[/bold cyan]\n"
        "[dim]Multi-signal prediction market trading[/dim]",
        border_style="cyan"
    ))

    # Initialize performance tracking
    tracker = PerformanceTracker()
    dashboard = Dashboard()
    visualizer = PerformanceVisualizer()

    # Show historical performance if requested
    if args.show_report:
        create_performance_report(tracker, dashboard)
        if args.generate_graphs:
            console.print("\n[cyan]Generating performance graphs...[/cyan]")
            graphs = visualizer.generate_all_graphs(tracker)
            for name, path in graphs.items():
                if path:
                    console.print(f"  [green]?[/green] {name}: {path}")
        return

    # Get API key
    api_key = os.getenv("MANIFOLD_API_KEY")

    # Create signals (Historical signal now learns from real data!)
    # Using smaller cache_size for faster initialization
    signals = [
        LLMSignal(weight=0.60),  # Primary signal - LLM has best judgment
        HistoricalSignal(weight=0.25, api_key=api_key, cache_size=50),  # Reduced cache for speed
        MicrostructureSignal(weight=0.15)  # Tertiary - confirmation only
    ]

    console.print(f"[green] Loaded {len(signals)} signals[/green]")

    # Create ensemble decision maker
    # Kelly will size bets based on edge, so we can be more aggressive on thresholds
    # Lower thresholds = more opportunities for Kelly to find +EV bets
    decision_maker = EnsembleDecisionMaker(
        signals=signals,
        min_confidence=0.60,  # Lower threshold - Kelly sizes appropriately anyway
        min_edge=0.05         # 3% minimum edge - Kelly handles the sizing
    )

    # Create bot with Kelly Criterion bet sizing
    # Using quarter-Kelly for good risk/reward balance
    bot = KellyBot(
        manifold_api_key=api_key,
        decision_maker=decision_maker,
        kelly_fraction=0.25,     # Quarter Kelly - safe but still grows well
        max_bet_fraction=0.15,   # Cap at 15% of bankroll per bet
        min_bet=2.0,             # Minimum $2 to make bet worthwhile
        max_bet=50.0,            # Cap at $50 to diversify risk
        dry_run=args.dry_run
    )

    # Add tracking to bot instance
    bot.tracker = tracker
    bot.dashboard = dashboard
    bot.last_signal_results = []
    original_analyze = decision_maker.analyze_market
    def patched_analyze(market):
        result = original_analyze(market)
        bot.last_signal_results = getattr(decision_maker, 'last_signal_results', [])
        return result
    decision_maker.analyze_market = patched_analyze

    console.print("[green] Ensemble decision maker initialized[/green]")
    console.print("[green] Performance tracking enabled[/green]\n")

    # Run on MikhailTal's markets
    # NOTE: Not passing bet_amount so Kelly sizing is used!
    session = bot.run_on_user_markets(
        username="MikhailTal",
        limit=args.limit,
        bet_amount=None,  # Use Kelly sizing
        max_bets=20  # Allow more bets since Kelly sizes them optimally
    )

    # End tracking session
    summary = tracker.end_session(
        markets_analyzed=session.markets_analyzed,
        initial_balance=session.initial_balance,
        final_balance=session.final_balance
    )

    console.print("\n" + "="*80)
    # Show session summary
    dashboard.show_session_summary(summary.__dict__ if hasattr(summary, '__dict__') else summary)

    # Show recent bets
    if tracker.session_bets:
        console.print()
        dashboard.show_live_bets_table([
            bet.__dict__ if hasattr(bet, '__dict__') else bet
            for bet in tracker.session_bets
        ])

    # Generate graphs if requested
    if args.generate_graphs:
        console.print("\n[cyan]Generating performance graphs...[/cyan]")
        graphs = visualizer.generate_all_graphs(tracker)
        for name, path in graphs.items():
            if path:
                console.print(f"  [green]?[/green] {name}: {path}")

    console.print("\n[dim]Data saved to: performance_data/[/dim]")
    console.print("[dim]Run with --show-report to see full historical performance[/dim]\n")

if __name__ == "__main__":
    main()




