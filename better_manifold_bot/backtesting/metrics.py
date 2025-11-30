"""
Performance metrics and calibration analysis for backtesting.

Provides detailed statistical analysis of trading strategy performance.
"""

import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics for a trading strategy."""

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown analysis
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: int = 0  # Number of trades in drawdown

    # Trade statistics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_loss_ratio: float = 0.0
    expectancy: float = 0.0  # Expected value per trade

    # Distribution
    profit_factor: float = 0.0  # Gross profit / Gross loss

    @classmethod
    def from_trades(cls, trades: List, initial_balance: float = 1000.0):
        """Calculate metrics from a list of BacktestTrade objects."""
        if not trades:
            return cls()

        # Filter to only traded markets
        traded = [t for t in trades if t.decision != "SKIP"]

        if not traded:
            return cls()

        # Calculate equity curve
        balance = initial_balance
        equity_curve = [balance]
        for trade in traded:
            balance += trade.profit_loss
            equity_curve.append(balance)

        # Calculate returns
        returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]

        # Sharpe Ratio
        if returns and statistics.stdev(returns) > 0:
            sharpe = statistics.mean(returns) / statistics.stdev(returns) * math.sqrt(len(returns))
        else:
            sharpe = 0.0

        # Sortino Ratio (uses only downside deviation)
        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_dev = math.sqrt(statistics.mean([r**2 for r in downside_returns]))
            if downside_dev > 0:
                sortino = statistics.mean(returns) / downside_dev * math.sqrt(len(returns))
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        # Drawdown analysis
        peak = initial_balance
        max_dd = 0.0
        drawdowns = []
        dd_lengths = []
        current_dd_length = 0

        for value in equity_curve:
            if value > peak:
                peak = value
                if current_dd_length > 0:
                    dd_lengths.append(current_dd_length)
                    current_dd_length = 0
            else:
                dd = (peak - value) / peak
                drawdowns.append(dd)
                max_dd = max(max_dd, dd)
                current_dd_length += 1

        avg_dd = statistics.mean(drawdowns) if drawdowns else 0.0
        avg_dd_duration = statistics.mean(dd_lengths) if dd_lengths else 0

        # Calmar Ratio (return / max drawdown)
        total_return = (equity_curve[-1] - initial_balance) / initial_balance
        calmar = total_return / max_dd if max_dd > 0 else 0.0

        # Win/Loss statistics
        wins = [t.profit_loss for t in traded if t.profit_loss > 0]
        losses = [t.profit_loss for t in traded if t.profit_loss < 0]

        avg_win = statistics.mean(wins) if wins else 0.0
        avg_loss = statistics.mean(losses) if losses else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

        # Expectancy
        win_rate = len(wins) / len(traded) if traded else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Profit Factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return cls(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            drawdown_duration=int(avg_dd_duration),
            avg_win=avg_win,
            avg_loss=avg_loss,
            win_loss_ratio=win_loss_ratio,
            expectancy=expectancy,
            profit_factor=profit_factor
        )


@dataclass
class CalibrationMetrics:
    """
    Calibration metrics measuring how well predicted probabilities match reality.

    Good calibration means when you predict 70%, you're right 70% of the time.
    """

    brier_score: float = 0.0  # Lower is better (0 = perfect)
    log_loss: float = 0.0  # Lower is better
    calibration_error: float = 0.0  # Mean absolute calibration error

    # Calibration by confidence bucket
    calibration_by_bucket: Dict[str, Tuple[float, float, int]] = None
    # Maps bucket name -> (predicted_prob, actual_rate, num_samples)

    @classmethod
    def from_trades(cls, trades: List, num_buckets: int = 10):
        """
        Calculate calibration metrics from trades.

        Args:
            trades: List of BacktestTrade objects
            num_buckets: Number of buckets for calibration curve

        Returns:
            CalibrationMetrics object
        """
        # Filter to only traded markets with known outcomes
        traded = [
            t for t in trades
            if t.decision != "SKIP" and t.actual_outcome is not None
        ]

        if not traded:
            return cls(calibration_by_bucket={})

        # Calculate Brier Score
        brier_scores = [
            (t.predicted_probability - (1.0 if t.actual_outcome else 0.0)) ** 2
            for t in traded
        ]
        brier = statistics.mean(brier_scores)

        # Calculate Log Loss
        log_losses = []
        for t in traded:
            pred = max(0.001, min(0.999, t.predicted_probability))  # Clip to avoid log(0)
            actual = 1.0 if t.actual_outcome else 0.0
            ll = -(actual * math.log(pred) + (1 - actual) * math.log(1 - pred))
            log_losses.append(ll)
        log_loss = statistics.mean(log_losses)

        # Calibration by bucket
        buckets = {}
        bucket_size = 1.0 / num_buckets

        for i in range(num_buckets):
            bucket_min = i * bucket_size
            bucket_max = (i + 1) * bucket_size
            bucket_name = f"{bucket_min:.1f}-{bucket_max:.1f}"

            # Get trades in this bucket
            bucket_trades = [
                t for t in traded
                if bucket_min <= t.predicted_probability < bucket_max
            ]

            if bucket_trades:
                avg_predicted = statistics.mean([t.predicted_probability for t in bucket_trades])
                actual_rate = sum([1 for t in bucket_trades if t.actual_outcome]) / len(bucket_trades)
                buckets[bucket_name] = (avg_predicted, actual_rate, len(bucket_trades))

        # Calculate calibration error
        calibration_errors = [
            abs(pred - actual) for pred, actual, _ in buckets.values()
        ]
        cal_error = statistics.mean(calibration_errors) if calibration_errors else 0.0

        return cls(
            brier_score=brier,
            log_loss=log_loss,
            calibration_error=cal_error,
            calibration_by_bucket=buckets
        )

    def print_calibration_plot(self):
        """Print ASCII calibration plot."""
        print("\n📊 Calibration Plot (Predicted vs Actual)")
        print("=" * 60)

        if not self.calibration_by_bucket:
            print("No data available")
            return

        for bucket_name, (predicted, actual, count) in sorted(self.calibration_by_bucket.items()):
            # Create simple bar chart
            pred_bar = "█" * int(predicted * 50)
            actual_bar = "░" * int(actual * 50)

            print(f"{bucket_name}: n={count:3d}")
            print(f"  Predicted: {pred_bar} {predicted:.1%}")
            print(f"  Actual:    {actual_bar} {actual:.1%}")
            print()

        print(f"Calibration Error: {self.calibration_error:.3f}")
        print(f"Brier Score: {self.brier_score:.4f}")
        print("=" * 60)
