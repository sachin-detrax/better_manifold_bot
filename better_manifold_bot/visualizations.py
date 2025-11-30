"""
Data visualization and graph generation for Better Manifold Bot.

Creates:
- ROI charts
- P&L progression
- Confidence vs. Outcome scatter plots
- Signal performance comparison
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - graph generation disabled")


class PerformanceVisualizer:
    """Generate performance visualization graphs."""

    def __init__(self, output_dir: str = "performance_data/graphs"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save generated graphs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Visualization features disabled - matplotlib not installed")

    def plot_pnl_progression(
        self,
        sessions: List[Dict[str, Any]],
        filename: str = "pnl_progression.png"
    ) -> Optional[str]:
        """
        Plot P&L progression over time.

        Args:
            sessions: List of session summaries
            filename: Output filename

        Returns:
            Path to saved graph or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not sessions:
            return None

        # Extract data
        dates = [datetime.fromisoformat(s['start_time']) for s in sessions]
        pnls = [s['gross_pnl'] for s in sessions]
        balances = [s['final_balance'] for s in sessions]

        # Create cumulative P&L
        cumulative_pnl = []
        total = 0
        for pnl in pnls:
            total += pnl
            cumulative_pnl.append(total)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Cumulative P&L
        ax1.plot(dates, cumulative_pnl, 'b-', linewidth=2, label='Cumulative P&L')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(dates, cumulative_pnl, 0, alpha=0.3,
                         where=[p >= 0 for p in cumulative_pnl], color='green',
                         interpolate=True, label='Profit')
        ax1.fill_between(dates, cumulative_pnl, 0, alpha=0.3,
                         where=[p < 0 for p in cumulative_pnl], color='red',
                         interpolate=True, label='Loss')
        ax1.set_ylabel('Cumulative P&L (M$)', fontsize=12)
        ax1.set_title('P&L Progression Over Time', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Balance
        ax2.plot(dates, balances, 'g-', linewidth=2, label='Balance')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Balance (M$)', fontsize=12)
        ax2.set_title('Account Balance', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved P&L progression graph to {output_path}")
        return str(output_path)

    def plot_roi_by_edge(
        self,
        bets: List[Dict[str, Any]],
        filename: str = "roi_by_edge.png"
    ) -> Optional[str]:
        """
        Plot ROI vs Edge scatter plot.

        Args:
            bets: List of bet records
            filename: Output filename

        Returns:
            Path to saved graph or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not bets:
            return None

        # Filter resolved bets only
        resolved_bets = [b for b in bets if b.get('resolved') and b.get('roi') is not None]

        if not resolved_bets:
            logger.warning("No resolved bets to plot ROI")
            return None

        # Extract data
        edges = [b['edge'] * 100 for b in resolved_bets]  # Convert to percentage
        rois = [b['roi'] for b in resolved_bets]
        colors = ['green' if roi > 0 else 'red' for roi in rois]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        ax.scatter(edges, rois, c=colors, alpha=0.6, s=50)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Edge (%)', fontsize=12)
        ax.set_ylabel('ROI (%)', fontsize=12)
        ax.set_title('ROI vs. Edge', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROI vs Edge graph to {output_path}")
        return str(output_path)

    def plot_confidence_calibration(
        self,
        bets: List[Dict[str, Any]],
        filename: str = "confidence_calibration.png"
    ) -> Optional[str]:
        """
        Plot confidence calibration chart.

        Shows whether the bot's confidence levels are well-calibrated.
        E.g., do 70% confidence bets win 70% of the time?

        Args:
            bets: List of bet records
            filename: Output filename

        Returns:
            Path to saved graph or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not bets:
            return None

        # Filter resolved bets
        resolved_bets = [b for b in bets if b.get('resolved') and b.get('roi') is not None]

        if not resolved_bets:
            logger.warning("No resolved bets for calibration plot")
            return None

        # Group by confidence bins
        bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        bin_labels = []
        actual_win_rates = []
        bet_counts = []

        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            bin_label = f"{bin_start:.0%}-{bin_end:.0%}"

            # Get bets in this bin
            bin_bets = [
                b for b in resolved_bets
                if bin_start <= b['confidence'] < bin_end
            ]

            if bin_bets:
                wins = sum(1 for b in bin_bets if b['roi'] > 0)
                win_rate = wins / len(bin_bets)

                bin_labels.append((bin_start + bin_end) / 2)
                actual_win_rates.append(win_rate)
                bet_counts.append(len(bin_bets))

        if not bin_labels:
            logger.warning("Not enough data for calibration plot")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot perfect calibration line
        ax.plot([0.5, 1.0], [0.5, 1.0], 'k--', label='Perfect Calibration', linewidth=2)

        # Plot actual calibration
        ax.plot(bin_labels, actual_win_rates, 'bo-', label='Actual Win Rate', linewidth=2, markersize=8)

        # Add point sizes based on bet count
        for x, y, count in zip(bin_labels, actual_win_rates, bet_counts):
            ax.annotate(f'n={count}', (x, y), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8)

        ax.set_xlabel('Predicted Confidence', fontsize=12)
        ax.set_ylabel('Actual Win Rate', fontsize=12)
        ax.set_title('Confidence Calibration', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confidence calibration graph to {output_path}")
        return str(output_path)

    def plot_signal_performance(
        self,
        bets: List[Dict[str, Any]],
        filename: str = "signal_performance.png"
    ) -> Optional[str]:
        """
        Compare performance of individual signals.

        Args:
            bets: List of bet records
            filename: Output filename

        Returns:
            Path to saved graph or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not bets:
            return None

        # Filter resolved bets
        resolved_bets = [b for b in bets if b.get('resolved') and b.get('roi') is not None]

        if not resolved_bets:
            logger.warning("No resolved bets for signal performance plot")
            return None

        # Calculate signal accuracies
        signals = ['llm', 'historical', 'microstructure']
        signal_accuracies = {}

        for signal in signals:
            prob_key = f'{signal}_prob'
            correct = 0
            total = 0

            for bet in resolved_bets:
                if prob_key not in bet or bet['resolution'] not in ['YES', 'NO']:
                    continue

                signal_prob = bet[prob_key]
                market_prob = bet['market_prob']

                # Check if signal was "right" (closer to outcome than market)
                outcome = 1.0 if bet['resolution'] == 'YES' else 0.0

                signal_error = abs(signal_prob - outcome)
                market_error = abs(market_prob - outcome)

                if signal_error < market_error:
                    correct += 1
                total += 1

            if total > 0:
                signal_accuracies[signal] = {
                    'accuracy': correct / total,
                    'count': total
                }

        if not signal_accuracies:
            logger.warning("No signal data for performance plot")
            return None

        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        signal_names = [s.title() for s in signal_accuracies.keys()]
        accuracies = [signal_accuracies[s]['accuracy'] * 100 for s in signal_accuracies.keys()]
        counts = [signal_accuracies[s]['count'] for s in signal_accuracies.keys()]

        bars = ax.bar(signal_names, accuracies, color=['#2ecc71', '#f39c12', '#3498db'], alpha=0.7)

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}',
                   ha='center', va='bottom', fontsize=10)

        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline (50%)')
        ax.set_ylabel('Accuracy vs. Market (%)', fontsize=12)
        ax.set_title('Signal Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved signal performance graph to {output_path}")
        return str(output_path)

    def plot_bet_size_distribution(
        self,
        bets: List[Dict[str, Any]],
        filename: str = "bet_size_distribution.png"
    ) -> Optional[str]:
        """
        Plot distribution of bet sizes.

        Args:
            bets: List of bet records
            filename: Output filename

        Returns:
            Path to saved graph or None if failed
        """
        if not MATPLOTLIB_AVAILABLE or not bets:
            return None

        bet_amounts = [b['bet_amount'] for b in bets]

        if not bet_amounts:
            return None

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(bet_amounts, bins=20, color='#3498db', alpha=0.7, edgecolor='black')

        ax.set_xlabel('Bet Size (M$)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Bet Size Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add statistics
        avg_bet = sum(bet_amounts) / len(bet_amounts)
        ax.axvline(x=avg_bet, color='red', linestyle='--', linewidth=2,
                  label=f'Average: ${avg_bet:.2f}')
        ax.legend()

        plt.tight_layout()

        # Save
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved bet size distribution graph to {output_path}")
        return str(output_path)

    def generate_all_graphs(self, tracker) -> Dict[str, Optional[str]]:
        """
        Generate all available graphs.

        Args:
            tracker: PerformanceTracker instance

        Returns:
            Dictionary mapping graph names to file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate graphs - matplotlib not available")
            return {}

        graphs = {}

        # Load data
        sessions = tracker.load_all_sessions()
        bets = tracker.load_all_bets()

        # Generate each graph
        if sessions:
            graphs['pnl_progression'] = self.plot_pnl_progression(sessions)

        if bets:
            graphs['bet_size_distribution'] = self.plot_bet_size_distribution(bets)

            # Only for resolved bets
            resolved = [b for b in bets if b.get('resolved')]
            if resolved:
                graphs['roi_by_edge'] = self.plot_roi_by_edge(resolved)
                graphs['confidence_calibration'] = self.plot_confidence_calibration(resolved)
                graphs['signal_performance'] = self.plot_signal_performance(resolved)

        logger.info(f"Generated {len([p for p in graphs.values() if p])} graphs")
        return graphs


def create_ascii_graph(values: List[float], width: int = 50, height: int = 10) -> str:
    """
    Create simple ASCII graph for terminal display.

    Args:
        values: List of values to plot
        width: Width in characters
        height: Height in characters

    Returns:
        ASCII art graph as string
    """
    if not values:
        return "No data"

    # Normalize values to height
    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val if max_val != min_val else 1

    # Create graph
    lines = []
    for i in range(height, 0, -1):
        line = ""
        threshold = min_val + (value_range * i / height)

        for val in values[:width]:
            if val >= threshold:
                line += "█"
            else:
                line += " "

        lines.append(line)

    # Add baseline
    lines.append("─" * min(len(values), width))

    return "\n".join(lines)
