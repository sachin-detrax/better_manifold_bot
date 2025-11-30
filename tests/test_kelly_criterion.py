"""Tests for Kelly Criterion bet sizing."""

import pytest
from better_manifold_bot.kelly_criterion import KellyCalculator, calculate_position_size


class TestKellyCalculator:
    """Test Kelly Criterion calculations."""

    def test_positive_edge_yes_bet(self):
        """Test Kelly sizing with positive edge on YES bet."""
        calc = KellyCalculator(kelly_fraction=0.25)

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.70,
            market_prob=0.50,
            bet_on_yes=True
        )

        # Should bet something (we have edge)
        assert bet > 0
        # Should be less than max bet fraction
        assert bet <= 1000 * 0.10

    def test_positive_edge_no_bet(self):
        """Test Kelly sizing with positive edge on NO bet."""
        calc = KellyCalculator(kelly_fraction=0.25)

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.30,  # We think NO
            market_prob=0.60,     # Market thinks YES
            bet_on_yes=False
        )

        # Should bet something (we have edge)
        assert bet > 0

    def test_no_edge_returns_zero(self):
        """Test that no edge results in zero bet."""
        calc = KellyCalculator(kelly_fraction=0.25)

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.50,
            market_prob=0.50,
            bet_on_yes=True
        )

        assert bet == 0.0

    def test_negative_edge_returns_zero(self):
        """Test that negative edge results in zero bet."""
        calc = KellyCalculator(kelly_fraction=0.25)

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.40,  # We think it's 40%
            market_prob=0.60,     # Market thinks 60%
            bet_on_yes=True       # But we're betting YES (bad idea!)
        )

        assert bet == 0.0

    def test_respects_max_bet(self):
        """Test that max_bet constraint is respected."""
        calc = KellyCalculator(kelly_fraction=1.0)  # Full Kelly (aggressive)

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.90,
            market_prob=0.50,
            bet_on_yes=True,
            max_bet=50.0
        )

        assert bet <= 50.0

    def test_respects_min_bet(self):
        """Test that min_bet constraint is respected."""
        calc = KellyCalculator(kelly_fraction=0.01)  # Very conservative

        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=0.55,
            market_prob=0.50,
            bet_on_yes=True,
            min_bet=10.0
        )

        # Either bets min or bets zero (if no edge)
        assert bet >= 10.0 or bet == 0.0

    def test_invalid_probability_returns_min(self):
        """Test that invalid probabilities are handled gracefully."""
        calc = KellyCalculator(kelly_fraction=0.25)

        # Invalid predicted_prob
        bet = calc.calculate_bet_size(
            bankroll=1000,
            predicted_prob=1.5,  # Invalid!
            market_prob=0.50,
            bet_on_yes=True,
            min_bet=1.0
        )

        assert bet == 1.0  # Returns min_bet for invalid input

    def test_convenience_function(self):
        """Test convenience function."""
        bet = calculate_position_size(
            bankroll=1000,
            predicted_prob=0.70,
            market_prob=0.50,
            bet_on_yes=True,
            kelly_fraction=0.25
        )

        assert bet > 0


class TestEdgeCalculation:
    """Test edge calculation logic."""

    def test_edge_yes_bet(self):
        """Test edge calculation for YES bet."""
        calc = KellyCalculator()

        edge = calc._calculate_edge(
            predicted_prob=0.70,
            market_prob=0.50,
            bet_on_yes=True
        )

        assert edge == 0.20  # 70% - 50%

    def test_edge_no_bet(self):
        """Test edge calculation for NO bet."""
        calc = KellyCalculator()

        edge = calc._calculate_edge(
            predicted_prob=0.30,  # We think 30% YES = 70% NO
            market_prob=0.60,     # Market thinks 60% YES = 40% NO
            bet_on_yes=False
        )

        assert edge == 0.30  # Our 70% NO - Market's 40% NO


class TestOddsCalculation:
    """Test odds calculation for LMSR markets."""

    def test_odds_yes_bet(self):
        """Test odds for YES bet."""
        calc = KellyCalculator()

        odds = calc._calculate_odds(market_prob=0.50, bet_on_yes=True)
        assert odds == 1.0  # (1-0.5)/0.5 = 1.0 (even odds)

        odds = calc._calculate_odds(market_prob=0.25, bet_on_yes=True)
        assert odds == 3.0  # (1-0.25)/0.25 = 3.0 (3:1 odds)

    def test_odds_no_bet(self):
        """Test odds for NO bet."""
        calc = KellyCalculator()

        odds = calc._calculate_odds(market_prob=0.50, bet_on_yes=False)
        assert odds == 1.0  # 0.5/(1-0.5) = 1.0 (even odds)

        odds = calc._calculate_odds(market_prob=0.75, bet_on_yes=False)
        assert odds == 3.0  # 0.75/(1-0.75) = 3.0 (3:1 odds)
