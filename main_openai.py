"""
Main script for OpenAI-enhanced ensemble trading bot.

Uses the new OpenAISignal with:
- Structured forecasting framework
- Self-consistency ensemble
- Cross-signal disagreement penalty
- Mathematical decomposition validation
"""

import os
import argparse
import logging
import sys
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from manifoldbot.manifold.bot import TradingSession
from better_manifold_bot.ensemble_decision_maker import EnhancedEnsembleDecisionMaker
from better_manifold_bot.signals.openai_signal import OpenAISignal
from better_manifold_bot.signals.historical_signal import HistoricalSignal
from better_manifold_bot.signals.microstructure_signal import MicrostructureSignal
from better_manifold_bot.kelly_bot import KellyBot
from better_manifold_bot.performance_tracker import PerformanceTracker
from better_manifold_bot.dashboard import Dashboard, create_performance_report
from better_manifold_bot.visualizations import PerformanceVisualizer
from main_ensemble import BetterManifoldBot

console = Console()
logger = logging.getLogger(__name__)


def main():
    # Fix encoding issues on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="OpenAI-Enhanced Manifold Trading Bot")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--limit", type=int, default=10, help="Number of markets to analyze")
    parser.add_argument("--bet-amount", type=int, default=None, help="Fixed bet amount (overrides Kelly)")
    parser.add_argument("--show-report", action="store_true", help="Show performance report")
    parser.add_argument("--generate-graphs", action="store_true", help="Generate performance graphs")

    # OpenAI-specific parameters
    parser.add_argument("--n-runs", type=int, default=3, help="Number of OpenAI runs for self-consistency")
    parser.add_argument("--variance-penalty", type=float, default=0.5, help="Variance penalty coefficient")
    parser.add_argument("--disagreement-low", type=float, default=0.03, help="Low disagreement threshold")
    parser.add_argument("--disagreement-high", type=float, default=0.10, help="High disagreement threshold")
    parser.add_argument("--disable-openai", action="store_true", help="Disable OpenAI signal (for testing)")

    args = parser.parse_args()

    load_dotenv()

    console.print(Panel.fit(
        "[bold cyan] OpenAI-Enhanced Super-Forecaster Bot[/bold cyan]\n"
        "[dim]Structured forecasting with self-consistency ensemble[/dim]",
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
                    console.print(f"  [green]✓[/green] {name}: {path}")
        return

    # Get API keys
    manifold_api_key = os.getenv("MANIFOLD_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    if not manifold_api_key:
        console.print("[red]Error: MANIFOLD_API_KEY not found in environment[/red]")
        return

    # Create signals
    signals = []

    # Historical signal (runs first)
    signals.append(
        HistoricalSignal(
            weight=0.25,
            api_key=manifold_api_key,
            cache_size=50  # Reduced for speed
        )
    )
    console.print("[green]✓[/green] Historical signal loaded")

    # Microstructure signal (runs second)
    signals.append(
        MicrostructureSignal(weight=0.15)
    )
    console.print("[green]✓[/green] Microstructure signal loaded")

    # OpenAI signal (runs last, uses other signals for cross-validation)
    if not args.disable_openai:
        if not OPENAI_API_KEY:
            console.print("[yellow]⚠ OPENAI_API_KEY not found, OpenAI signal disabled[/yellow]")
        else:
            signals.append(
                OpenAISignal(
                    weight=0.60,
                    n_runs=args.n_runs,
                    variance_penalty_k=args.variance_penalty,
                    disagreement_threshold_low=args.disagreement_low,
                    disagreement_threshold_high=args.disagreement_high
                )
            )
            console.print(
                f"[green]✓[/green] OpenAI signal loaded "
                f"(n_runs={args.n_runs}, variance_penalty={args.variance_penalty})"
            )
    else:
        console.print("[yellow]⚠ OpenAI signal disabled by --disable-openai flag[/yellow]")

    console.print(f"\n[cyan]Total signals loaded: {len(signals)}[/cyan]\n")

    # Create enhanced ensemble decision maker
    decision_maker = EnhancedEnsembleDecisionMaker(
        signals=signals,
        min_confidence=0.60,  # Lower threshold - Kelly sizes appropriately
        min_edge=0.05         # 5% minimum edge
    )

    # Create bot with Kelly Criterion bet sizing
    bot = KellyBot(
        manifold_api_key=manifold_api_key,
        decision_maker=decision_maker,
        kelly_fraction=0.25,     # Quarter Kelly
        max_bet_fraction=0.15,   # Cap at 15% of bankroll
        min_bet=2.0,
        max_bet=50.0
    )

    # Add tracking to bot instance
    bot.tracker = tracker
    bot.dashboard = dashboard
    bot.last_signal_results = []

    # Monkey patch to capture signal results
    original_analyze = decision_maker.analyze_market
    def patched_analyze(market):
        result = original_analyze(market)
        bot.last_signal_results = getattr(decision_maker, 'last_signal_results', [])
        return result
    decision_maker.analyze_market = patched_analyze

    console.print("[green]✓[/green] Enhanced ensemble decision maker initialized")
    console.print("[green]✓[/green] Performance tracking enabled\n")

    # Show signal configuration
    console.print("[cyan]Signal Configuration:[/cyan]")
    for signal in signals:
        console.print(f"  • {signal.name}: weight={signal.weight:.2f}")
    console.print()

    # Run on MikhailTal's markets
    session = bot.run_on_user_markets(
        username="MikhailTal",
        limit=args.limit,
        bet_amount=args.bet_amount,  # Use None for Kelly sizing
        max_bets=20
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
                console.print(f"  [green]✓[/green] {name}: {path}")

    console.print("\n[dim]Data saved to: performance_data/[/dim]")
    console.print("[dim]Run with --show-report to see full historical performance[/dim]")

    # Show configuration used
    console.print(f"\n[dim]Configuration: n_runs={args.n_runs}, "
                  f"variance_penalty={args.variance_penalty}, "
                  f"disagreement_thresholds=[{args.disagreement_low}, {args.disagreement_high}][/dim]\n")


if __name__ == "__main__":
    main()
