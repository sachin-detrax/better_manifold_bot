"""
Backtest and optimize the ensemble strategy.

Usage:
    python backtest_optimize.py --mode backtest  # Run backtest with current params
    python backtest_optimize.py --mode optimize  # Optimize parameters
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from better_manifold_bot.ensemble_decision_maker import EnsembleDecisionMaker
from better_manifold_bot.signals.llm_signal import LLMSignal
from better_manifold_bot.signals.historical_signal import HistoricalSignal
from better_manifold_bot.signals.microstructure_signal import MicrostructureSignal
from better_manifold_bot.backtesting import BacktestEngine
from better_manifold_bot.backtesting.metrics import PerformanceMetrics, CalibrationMetrics

console = Console()
logging.basicConfig(level=logging.INFO)


def create_decision_maker(min_confidence: float = 0.60, min_edge: float = 0.05):
    """Factory function to create decision maker with given parameters."""
    signals = [
        LLMSignal(weight=0.40),
        HistoricalSignal(weight=0.30),
        MicrostructureSignal(weight=0.30)
    ]

    return EnsembleDecisionMaker(
        signals=signals,
        min_confidence=min_confidence,
        min_edge=min_edge
    )


def run_backtest(api_key: str, limit: int = 100):
    """Run backtest with current parameters."""
    console.print(Panel.fit(
        "[bold cyan] Running Backtest[/bold cyan]\n"
        "[dim]Testing strategy on resolved markets[/dim]",
        border_style="cyan"
    ))

    # Create engine
    engine = BacktestEngine(api_key)

    # Fetch resolved markets
    console.print("\n[yellow]Fetching resolved markets from MikhailTal...[/yellow]")
    markets = engine.fetch_resolved_markets(creator="MikhailTal", limit=limit)
    console.print(f"[green]✓ Found {len(markets)} resolved markets[/green]")

    # Create decision maker
    decision_maker = create_decision_maker(
        min_confidence=0.60,
        min_edge=0.05
    )

    # Run backtest
    console.print("\n[yellow]Running backtest...[/yellow]")
    result = engine.run_backtest(
        decision_maker,
        markets,
        bet_amount=10.0,
        initial_balance=1000.0,
        config={
            "min_confidence": 0.60,
            "min_edge": 0.05,
            "bet_amount": 10.0
        }
    )

    # Print results
    console.print(result)

    # Calculate detailed metrics
    perf_metrics = PerformanceMetrics.from_trades(result.trades)
    cal_metrics = CalibrationMetrics.from_trades(result.trades)

    # Print detailed metrics table
    table = Table(title=" Detailed Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Sharpe Ratio", f"{perf_metrics.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{perf_metrics.sortino_ratio:.2f}")
    table.add_row("Calmar Ratio", f"{perf_metrics.calmar_ratio:.2f}")
    table.add_row("Max Drawdown", f"{perf_metrics.max_drawdown:.1%}")
    table.add_row("Avg Win", f"${perf_metrics.avg_win:.2f}")
    table.add_row("Avg Loss", f"${perf_metrics.avg_loss:.2f}")
    table.add_row("Win/Loss Ratio", f"{perf_metrics.win_loss_ratio:.2f}")
    table.add_row("Expectancy", f"${perf_metrics.expectancy:.2f}")
    table.add_row("Profit Factor", f"{perf_metrics.profit_factor:.2f}")

    console.print("\n")
    console.print(table)

    # Print calibration
    cal_metrics.print_calibration_plot()

    # Analyze trades by confidence level
    print_trade_analysis(result.trades)

    return result


def run_optimization(api_key: str, limit: int = 100):
    """Optimize parameters using grid search."""
    console.print(Panel.fit(
        "[bold cyan] Parameter Optimization[/bold cyan]\n"
        "[dim]Finding optimal min_confidence and min_edge[/dim]",
        border_style="cyan"
    ))

    # Create engine
    engine = BacktestEngine(api_key)

    # Fetch resolved markets
    console.print("\n[yellow]Fetching resolved markets from MikhailTal...[/yellow]")
    markets = engine.fetch_resolved_markets(creator="MikhailTal", limit=limit)
    console.print(f"[green]✓ Found {len(markets)} resolved markets[/green]")

    # Define parameter grid
    param_grid = {
        "min_confidence": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "min_edge": [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    }

    console.print(f"\n[yellow]Testing {len(param_grid['min_confidence']) * len(param_grid['min_edge'])} parameter combinations...[/yellow]")

    # Run optimization
    best_params, best_result = engine.optimize_parameters(
        decision_maker_factory=create_decision_maker,
        markets=markets,
        param_grid=param_grid,
        bet_amount=10.0,
        metric="sharpe_ratio"  # Optimize for risk-adjusted returns
    )

    # Print best results
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]🏆 Best Parameters Found[/bold green]\n\n"
        f"min_confidence: {best_params['min_confidence']:.2f}\n"
        f"min_edge: {best_params['min_edge']:.2f}\n\n"
        f"Win Rate: {best_result.win_rate:.1%}\n"
        f"Sharpe Ratio: {best_result.sharpe_ratio:.2f}\n"
        f"Total P&L: ${best_result.total_profit_loss:+.2f}",
        border_style="green"
    ))

    console.print(best_result)

    return best_params, best_result


def print_trade_analysis(trades):
    """Analyze trades by confidence bucket."""
    print("\n Performance by Confidence Level")
    print("=" * 60)

    # Group by confidence buckets
    buckets = {
        "50-60%": [],
        "60-70%": [],
        "70-80%": [],
        "80-90%": [],
        "90-100%": []
    }

    for trade in trades:
        if trade.decision == "SKIP":
            continue

        conf = trade.confidence
        if conf < 0.6:
            buckets["50-60%"].append(trade)
        elif conf < 0.7:
            buckets["60-70%"].append(trade)
        elif conf < 0.8:
            buckets["70-80%"].append(trade)
        elif conf < 0.9:
            buckets["80-90%"].append(trade)
        else:
            buckets["90-100%"].append(trade)

    for bucket_name, bucket_trades in buckets.items():
        if not bucket_trades:
            continue

        wins = sum(1 for t in bucket_trades if t.profit_loss > 0)
        win_rate = wins / len(bucket_trades)
        total_pnl = sum(t.profit_loss for t in bucket_trades)

        print(f"{bucket_name}: n={len(bucket_trades):3d} | Win Rate: {win_rate:.1%} | P&L: ${total_pnl:+.2f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backtest and optimize trading strategy")
    parser.add_argument(
        "--mode",
        choices=["backtest", "optimize"],
        default="backtest",
        help="Run mode: backtest with current params or optimize parameters"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of markets to fetch for backtesting"
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("MANIFOLD_API_KEY")

    if not api_key:
        console.print("[red]Error: MANIFOLD_API_KEY not found in .env file[/red]")
        return

    if args.mode == "backtest":
        run_backtest(api_key, args.limit)
    else:
        run_optimization(api_key, args.limit)


if __name__ == "__main__":
    main()
