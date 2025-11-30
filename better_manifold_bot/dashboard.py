"""
Rich dashboard and visualization for Better Manifold Bot performance.

Displays:
- Real-time P&L
- Bet history table
- Performance metrics
- Beautiful formatted reports
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich import box
from rich.columns import Columns
from rich.tree import Tree

logger = logging.getLogger(__name__)


class Dashboard:
    """Rich dashboard for displaying bot performance."""

    def __init__(self):
        self.console = Console()

    def show_session_summary(self, summary: Dict[str, Any]):
        """
        Display comprehensive session summary.

        Args:
            summary: SessionSummary data as dict
        """
        # Header
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan] Session Summary[/bold cyan]",
            border_style="cyan"
        ))

        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="top"),
            Layout(name="middle"),
            Layout(name="bottom")
        )

        # Top: Overview
        overview_table = Table(show_header=False, box=box.ROUNDED, expand=True)
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", style="bold white")

        session_duration = timedelta(seconds=summary['duration_seconds'])
        pnl = summary['gross_pnl']
        pnl_color = "green" if pnl >= 0 else "red"

        overview_table.add_row("Session ID", summary['session_id'])
        overview_table.add_row("Duration", str(session_duration))
        overview_table.add_row("Markets Analyzed", str(summary['markets_analyzed']))
        overview_table.add_row("Bets Placed", str(summary['bets_placed']))
        overview_table.add_row("", "")
        overview_table.add_row("Initial Balance", f"${summary['initial_balance']:.2f}")
        overview_table.add_row("Final Balance", f"${summary['final_balance']:.2f}")
        overview_table.add_row("P&L", f"[{pnl_color}]{pnl:+.2f} M$[/{pnl_color}]")

        # Middle: Bet Statistics
        bet_stats_table = Table(show_header=False, box=box.ROUNDED, expand=True)
        bet_stats_table.add_column("Metric", style="yellow")
        bet_stats_table.add_column("Value", style="bold white")

        if summary['bets_placed'] > 0:
            bet_stats_table.add_row("Total Bet Amount", f"${summary['total_bet_amount']:.2f}")
            bet_stats_table.add_row("Average Bet Size", f"${summary['avg_bet_size']:.2f}")
            bet_stats_table.add_row("Min Bet Size", f"${summary['min_bet_size']:.2f}")
            bet_stats_table.add_row("Max Bet Size", f"${summary['max_bet_size']:.2f}")
            bet_stats_table.add_row("", "")
            bet_stats_table.add_row("Average Confidence", f"{summary['avg_confidence']:.1%}")
            bet_stats_table.add_row("Average Edge", f"{summary['avg_edge']:.1%}")
            bet_stats_table.add_row("", "")
            bet_stats_table.add_row("YES Bets", str(summary['yes_bets']))
            bet_stats_table.add_row("NO Bets", str(summary['no_bets']))
        else:
            bet_stats_table.add_row("No bets placed", "-")

        # Bottom: Distributions
        dist_panels = []

        # Confidence distribution
        if summary.get('confidence_distribution'):
            conf_table = Table(title="Confidence Distribution", box=box.SIMPLE)
            conf_table.add_column("Range", style="cyan")
            conf_table.add_column("Count", style="bold")
            for range_label, count in sorted(summary['confidence_distribution'].items()):
                conf_table.add_row(range_label, str(count))
            dist_panels.append(Panel(conf_table, border_style="blue"))

        # Edge distribution
        if summary.get('edge_distribution'):
            edge_table = Table(title="Edge Distribution", box=box.SIMPLE)
            edge_table.add_column("Range", style="yellow")
            edge_table.add_column("Count", style="bold")
            for range_label, count in sorted(summary['edge_distribution'].items()):
                edge_table.add_row(range_label, str(count))
            dist_panels.append(Panel(edge_table, border_style="yellow"))

        # Display
        self.console.print(Panel(overview_table, title="[bold]Overview[/bold]", border_style="cyan"))
        self.console.print(Panel(bet_stats_table, title="[bold]Betting Statistics[/bold]", border_style="yellow"))

        if dist_panels:
            self.console.print(Columns(dist_panels))

        self.console.print()

    def show_live_bets_table(self, bets: List[Dict[str, Any]], max_rows: int = 20):
        """
        Display table of recent bets.

        Args:
            bets: List of bet records
            max_rows: Maximum rows to display
        """
        if not bets:
            self.console.print("[yellow]No bets to display[/yellow]")
            return

        table = Table(title="Recent Bets", box=box.ROUNDED)

        table.add_column("Time", style="dim", width=8)
        table.add_column("Direction", style="bold", width=5)
        table.add_column("Amount", style="cyan", justify="right", width=8)
        table.add_column("Market", style="white", width=50)
        table.add_column("Market P", style="yellow", justify="right", width=7)
        table.add_column("Ensemble P", style="green", justify="right", width=7)
        table.add_column("Edge", style="magenta", justify="right", width=6)
        table.add_column("Conf", style="blue", justify="right", width=6)

        # Show most recent bets
        recent_bets = bets[-max_rows:] if len(bets) > max_rows else bets

        for bet in recent_bets:
            # Parse timestamp
            timestamp = datetime.fromisoformat(bet['timestamp'])
            time_str = timestamp.strftime("%H:%M:%S")

            # Direction with color
            direction = bet['direction']
            dir_style = "green" if direction == "YES" else "red"
            dir_text = f"[{dir_style}]{direction}[/{dir_style}]"

            # Format market question (truncate)
            question = bet['question']
            if len(question) > 47:
                question = question[:47] + "..."

            # Edge with color
            edge = bet['edge']
            edge_color = "green" if edge > 0.05 else "yellow"

            table.add_row(
                time_str,
                dir_text,
                f"${bet['bet_amount']:.2f}",
                question,
                f"{bet['market_prob']:.1%}",
                f"{bet['ensemble_prob']:.1%}",
                f"[{edge_color}]{edge:.1%}[/{edge_color}]",
                f"{bet['confidence']:.1%}"
            )

        self.console.print(table)

    def show_pnl_chart(self, sessions: List[Dict[str, Any]]):
        """
        Display P&L progression chart using ASCII art.

        Args:
            sessions: List of session summaries
        """
        if not sessions:
            self.console.print("[yellow]No session data available[/yellow]")
            return

        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan] P&L History[/bold cyan]",
            border_style="cyan"
        ))

        # Create table
        table = Table(box=box.ROUNDED)
        table.add_column("Date", style="cyan")
        table.add_column("Session", style="dim")
        table.add_column("Bets", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Balance", justify="right", style="bold")
        table.add_column("Chart", width=30)

        max_pnl = max(abs(s['gross_pnl']) for s in sessions)

        for session in sessions[-20:]:  # Last 20 sessions
            # Parse date
            date = datetime.fromisoformat(session['start_time'])
            date_str = date.strftime("%m/%d")

            # Session ID (shortened)
            session_id = session['session_id'][-8:]

            # P&L with color
            pnl = session['gross_pnl']
            pnl_color = "green" if pnl >= 0 else "red"
            pnl_str = f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]"

            # Balance
            balance_str = f"${session['final_balance']:.2f}"

            # ASCII bar chart
            if max_pnl > 0:
                bar_length = int(abs(pnl) / max_pnl * 20)
                if pnl >= 0:
                    bar = "[green]" + "█" * bar_length + "[/green]"
                else:
                    bar = "[red]" + "█" * bar_length + "[/red]"
            else:
                bar = ""

            table.add_row(
                date_str,
                session_id,
                str(session['bets_placed']),
                pnl_str,
                balance_str,
                bar
            )

        self.console.print(table)
        self.console.print()

    def show_signal_breakdown(self, bet: Dict[str, Any]):
        """
        Show detailed signal breakdown for a bet.

        Args:
            bet: Bet record with signal data
        """
        tree = Tree(f"[bold cyan]Signal Analysis: {bet['direction']}[/bold cyan]")

        # LLM Signal
        llm_branch = tree.add(f"[green]LLM Signal[/green]")
        llm_branch.add(f"Probability: {bet['llm_prob']:.1%}")
        llm_branch.add(f"Confidence: {bet['llm_conf']:.1%}")

        # Historical Signal
        hist_branch = tree.add(f"[yellow]Historical Signal[/yellow]")
        hist_branch.add(f"Probability: {bet['historical_prob']:.1%}")
        hist_branch.add(f"Confidence: {bet['historical_conf']:.1%}")

        # Microstructure Signal
        micro_branch = tree.add(f"[blue]Microstructure Signal[/blue]")
        micro_branch.add(f"Probability: {bet['microstructure_prob']:.1%}")
        micro_branch.add(f"Confidence: {bet['microstructure_conf']:.1%}")

        # Ensemble
        ensemble_branch = tree.add(f"[bold magenta]Ensemble Result[/bold magenta]")
        ensemble_branch.add(f"Final Probability: {bet['ensemble_prob']:.1%}")
        ensemble_branch.add(f"Market Probability: {bet['market_prob']:.1%}")
        ensemble_branch.add(f"Edge: {bet['edge']:.1%}")
        ensemble_branch.add(f"Confidence: {bet['confidence']:.1%}")

        self.console.print(tree)

    def show_daily_summary(self, tracker):
        """
        Show daily performance summary.

        Args:
            tracker: PerformanceTracker instance
        """
        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan]Daily Summary[/bold cyan]",
            border_style="cyan"
        ))

        # Get today's data
        perf = tracker.get_historical_performance(days=1)

        if perf['total_bets'] == 0:
            self.console.print("[yellow]No bets placed today[/yellow]")
            return

        # Create summary table
        table = Table(show_header=False, box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="bold white", justify="right")

        pnl = perf['total_pnl']
        pnl_color = "green" if pnl >= 0 else "red"

        table.add_row("Total Sessions", str(perf['total_sessions']))
        table.add_row("Total Bets", str(perf['total_bets']))
        table.add_row("Average Bets/Session", f"{perf['avg_bets_per_session']:.1f}")
        table.add_row("", "")
        table.add_row("Total Bet Amount", f"${perf['total_bet_amount']:.2f}")
        table.add_row("Total P&L", f"[{pnl_color}]{pnl:+.2f} M$[/{pnl_color}]")
        table.add_row("Average Session P&L", f"[{pnl_color}]{perf['avg_session_pnl']:+.2f} M$[/{pnl_color}]")

        self.console.print(table)
        self.console.print()

    def show_live_session_stats(self, stats: Dict[str, Any], balance: float):
        """
        Show live session statistics during bot run.

        Args:
            stats: Current session stats
            balance: Current balance
        """
        # Create compact stats display
        if stats['bets_placed'] == 0:
            self.console.print("[dim]No bets placed yet in this session[/dim]")
            return

        panel_content = (
            f"[cyan]Bets:[/cyan] {stats['bets_placed']} "
            f"([green]{stats['yes_bets']}Y[/green]/[red]{stats['no_bets']}N[/red]) | "
            f"[yellow]Avg Bet:[/yellow] ${stats['avg_bet']:.2f} "
            f"([dim]${stats['min_bet']:.2f}-${stats['max_bet']:.2f}[/dim]) | "
            f"[magenta]Avg Edge:[/magenta] {stats['avg_edge']:.1%} | "
            f"[blue]Avg Conf:[/blue] {stats['avg_confidence']:.1%} | "
            f"[bold white]Balance:[/bold white] ${balance:.2f}"
        )

        self.console.print(Panel(panel_content, border_style="dim", expand=False))

    def show_roi_metrics(self, sessions: List[Dict[str, Any]]):
        """
        Display ROI and performance metrics.

        Args:
            sessions: List of session summaries
        """
        if not sessions:
            self.console.print("[yellow]No historical data available[/yellow]")
            return

        self.console.print()
        self.console.print(Panel.fit(
            "[bold cyan] ROI Metrics[/bold cyan]",
            border_style="cyan"
        ))

        # Calculate metrics
        total_pnl = sum(s['gross_pnl'] for s in sessions)
        total_bet_amount = sum(s['total_bet_amount'] for s in sessions)
        total_bets = sum(s['bets_placed'] for s in sessions)

        roi = (total_pnl / total_bet_amount * 100) if total_bet_amount > 0 else 0
        avg_pnl_per_bet = total_pnl / total_bets if total_bets > 0 else 0

        # Winning vs losing sessions
        winning_sessions = sum(1 for s in sessions if s['gross_pnl'] > 0)
        losing_sessions = sum(1 for s in sessions if s['gross_pnl'] < 0)

        table = Table(show_header=False, box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="bold white", justify="right")

        roi_color = "green" if roi >= 0 else "red"

        table.add_row("Total Sessions", str(len(sessions)))
        table.add_row("Winning Sessions", f"[green]{winning_sessions}[/green]")
        table.add_row("Losing Sessions", f"[red]{losing_sessions}[/red]")
        table.add_row("", "")
        table.add_row("Total Bets", str(total_bets))
        table.add_row("Total Bet Amount", f"${total_bet_amount:.2f}")
        table.add_row("Total P&L", f"[{roi_color}]{total_pnl:+.2f} M$[/{roi_color}]")
        table.add_row("", "")
        table.add_row("ROI", f"[{roi_color}]{roi:+.2f}%[/{roi_color}]")
        table.add_row("Avg P&L per Bet", f"[{roi_color}]{avg_pnl_per_bet:+.2f} M$[/{roi_color}]")

        self.console.print(table)
        self.console.print()


def create_performance_report(tracker, dashboard: Dashboard):
    """
    Create comprehensive performance report.

    Args:
        tracker: PerformanceTracker instance
        dashboard: Dashboard instance
    """
    console = Console()

    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan] COMPREHENSIVE PERFORMANCE REPORT[/bold cyan]",
        border_style="cyan"
    ))
    console.print("="*80 + "\n")

    # 1. Daily Summary
    dashboard.show_daily_summary(tracker)

    # 2. Historical Performance
    all_sessions = tracker.load_all_sessions()
    if all_sessions:
        # Last 7 days
        perf_7d = tracker.get_historical_performance(days=7)
        console.print(Panel.fit(
            f"[bold]Last 7 Days:[/bold] "
            f"{perf_7d['total_bets']} bets, "
            f"P&L: {perf_7d['total_pnl']:+.2f} M$",
            border_style="blue"
        ))

        # Last 30 days
        perf_30d = tracker.get_historical_performance(days=30)
        console.print(Panel.fit(
            f"[bold]Last 30 Days:[/bold] "
            f"{perf_30d['total_bets']} bets, "
            f"P&L: {perf_30d['total_pnl']:+.2f} M$",
            border_style="blue"
        ))

        # 3. ROI Metrics
        dashboard.show_roi_metrics(all_sessions)

        # 4. P&L Chart
        dashboard.show_pnl_chart(all_sessions)

    # 5. Recent Bets
    all_bets = tracker.load_all_bets()
    if all_bets:
        console.print(Panel.fit(
            "[bold cyan] Recent Bets[/bold cyan]",
            border_style="cyan"
        ))
        dashboard.show_live_bets_table(all_bets, max_rows=15)

    console.print("\n" + "="*80 + "\n")
