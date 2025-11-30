"""Tests for backtesting engine."""

import pytest
from unittest.mock import Mock, MagicMock
from better_manifold_bot.backtesting import BacktestEngine, BacktestResult, BacktestTrade
from better_manifold_bot.backtesting.metrics import PerformanceMetrics, CalibrationMetrics
from manifoldbot.manifold.bot import MarketDecision


class MockDecisionMaker:
    """Mock decision maker for testing."""

    def __init__(self, always_yes=False, always_skip=False):
        self.always_yes = always_yes
        self.always_skip = always_skip

    def analyze_market(self, market):
        if self.always_skip:
            decision = "SKIP"
        elif self.always_yes:
            decision = "YES"
        else:
            decision = "NO"

        return MarketDecision(
            market_id=market.get("id", ""),
            question=market.get("question", ""),
            current_probability=market.get("probability", 0.5),
            decision=decision,
            confidence=0.80,
            reasoning="Test decision",
            outcome_type="BINARY",
            metadata={"ensemble_probability": 0.7}
        )


class TestBacktestEngine:
    """Test backtesting engine."""

    def create_mock_market(self, market_id, resolution="YES", probability=0.5):
        """Create a mock resolved market."""
        return {
            "id": market_id,
            "question": f"Test market {market_id}",
            "probability": probability,
            "isResolved": True,
            "resolution": resolution,
            "outcomeType": "BINARY",
            "createdTime": "2024-01-01T00:00:00Z"
        }

    def test_backtest_all_wins(self):
        """Test backtest where all bets win."""
        # Create markets that all resolve YES
        markets = [
            self.create_mock_market(f"market{i}", resolution="YES", probability=0.5)
            for i in range(10)
        ]

        # Decision maker that always bets YES
        dm = MockDecisionMaker(always_yes=True)

        # Note: We can't easily test this without a real API
        # This test would need mocking of ManifoldAPI
        # For now, we test the calculation logic

    def test_calculate_pnl_win_yes(self):
        """Test P&L calculation for winning YES bet."""
        engine = BacktestEngine(api_key="test")

        pnl = engine._calculate_pnl(
            decision="YES",
            bet_amount=10,
            market_probability=0.50,
            actual_outcome=True  # Resolved YES
        )

        # Betting $10 at 50% odds, winning should pay ~$10 profit
        assert pnl > 0
        assert pnl < bet_amount * 2  # Can't make more than 2x at 50% odds

    def test_calculate_pnl_loss_yes(self):
        """Test P&L calculation for losing YES bet."""
        engine = BacktestEngine(api_key="test")

        pnl = engine._calculate_pnl(
            decision="YES",
            bet_amount=10,
            market_probability=0.50,
            actual_outcome=False  # Resolved NO
        )

        # Should lose the bet amount
        assert pnl == -10

    def test_calculate_pnl_win_no(self):
        """Test P&L calculation for winning NO bet."""
        engine = BacktestEngine(api_key="test")

        pnl = engine._calculate_pnl(
            decision="NO",
            bet_amount=10,
            market_probability=0.50,
            actual_outcome=False  # Resolved NO (we win!)
        )

        # Should profit
        assert pnl > 0

    def test_calculate_pnl_loss_no(self):
        """Test P&L calculation for losing NO bet."""
        engine = BacktestEngine(api_key="test")

        pnl = engine._calculate_pnl(
            decision="NO",
            bet_amount=10,
            market_probability=0.50,
            actual_outcome=True  # Resolved YES (we lose)
        )

        # Should lose
        assert pnl == -10


class TestPerformanceMetrics:
    """Test performance metrics calculations."""

    def create_mock_trade(self, pnl, decision="YES"):
        """Create a mock trade."""
        return BacktestTrade(
            market_id="test",
            question="Test",
            decision=decision,
            bet_amount=10,
            entry_probability=0.5,
            predicted_probability=0.7,
            confidence=0.8,
            actual_outcome=True if pnl > 0 else False,
            profit_loss=pnl
        )

    def test_all_wins(self):
        """Test metrics with all winning trades."""
        trades = [
            self.create_mock_trade(pnl=10) for _ in range(10)
        ]

        metrics = PerformanceMetrics.from_trades(trades)

        assert metrics.avg_win == 10.0
        assert metrics.avg_loss == 0.0
        assert metrics.expectancy > 0

    def test_all_losses(self):
        """Test metrics with all losing trades."""
        trades = [
            self.create_mock_trade(pnl=-10) for _ in range(10)
        ]

        metrics = PerformanceMetrics.from_trades(trades)

        assert metrics.avg_win == 0.0
        assert metrics.avg_loss == -10.0
        assert metrics.expectancy < 0

    def test_mixed_trades(self):
        """Test metrics with mixed wins and losses."""
        trades = [
            self.create_mock_trade(pnl=10),
            self.create_mock_trade(pnl=15),
            self.create_mock_trade(pnl=-10),
            self.create_mock_trade(pnl=-10),
        ]

        metrics = PerformanceMetrics.from_trades(trades)

        assert metrics.avg_win == 12.5  # (10+15)/2
        assert metrics.avg_loss == -10.0
        assert metrics.profit_factor == 25.0 / 20.0  # 1.25

    def test_skipped_trades_ignored(self):
        """Test that SKIP decisions are ignored in metrics."""
        trades = [
            self.create_mock_trade(pnl=10, decision="YES"),
            BacktestTrade(
                market_id="skip",
                question="Skipped",
                decision="SKIP",
                bet_amount=0,
                entry_probability=0.5,
                predicted_probability=0.5,
                confidence=0.5,
                actual_outcome=None,
                profit_loss=0
            ),
            self.create_mock_trade(pnl=-10, decision="NO"),
        ]

        metrics = PerformanceMetrics.from_trades(trades)

        # Should only count the 2 non-skipped trades
        assert metrics.expectancy == 0.0  # Break-even


class TestCalibrationMetrics:
    """Test calibration metrics."""

    def create_calibrated_trade(self, predicted_prob, actual_outcome):
        """Create a trade with specific prediction and outcome."""
        return BacktestTrade(
            market_id="test",
            question="Test",
            decision="YES" if predicted_prob > 0.5 else "NO",
            bet_amount=10,
            entry_probability=0.5,
            predicted_probability=predicted_prob,
            confidence=0.8,
            actual_outcome=actual_outcome,
            profit_loss=10 if actual_outcome else -10
        )

    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        # Predict 70% 10 times, and 7 of them are correct
        trades = [
            self.create_calibrated_trade(0.70, True) for _ in range(7)
        ] + [
            self.create_calibrated_trade(0.70, False) for _ in range(3)
        ]

        metrics = CalibrationMetrics.from_trades(trades)

        # Brier score should be low (good)
        assert metrics.brier_score < 0.3
        # Calibration error should be low
        assert metrics.calibration_error < 0.1

    def test_overconfident(self):
        """Test overconfident predictions (claim 90% but only 60% correct)."""
        trades = [
            self.create_calibrated_trade(0.90, True) for _ in range(6)
        ] + [
            self.create_calibrated_trade(0.90, False) for _ in range(4)
        ]

        metrics = CalibrationMetrics.from_trades(trades)

        # Should have higher calibration error
        assert metrics.calibration_error > 0.1

    def test_underconfident(self):
        """Test underconfident predictions (claim 60% but actually 90% correct)."""
        trades = [
            self.create_calibrated_trade(0.60, True) for _ in range(9)
        ] + [
            self.create_calibrated_trade(0.60, False) for _ in range(1)
        ]

        metrics = CalibrationMetrics.from_trades(trades)

        # Should have higher calibration error
        assert metrics.calibration_error > 0.1
