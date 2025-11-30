"""
Unit tests for enhanced Claude signal.

Tests cover:
- Schema validation
- Decomposition validation
- Edge calculation
- Variance penalty
- Disagreement penalty
- Retry and fallback logic
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from better_manifold_bot.signals.claude_signal import (
    ClaudeSignal,
    ForecastRun,
    FORECAST_SCHEMA,
    create_user_prompt
)
from better_manifold_bot.signals.base_signal import SignalResult


# Sample valid forecast response
VALID_FORECAST = {
    "base_rate": 0.50,
    "factors": [
        {"name": "recent_polls", "impact": 0.10, "evidence_count": 5},
        {"name": "historical_trend", "impact": -0.05, "evidence_count": 3}
    ],
    "confounders": ["media_coverage", "sampling_bias"],
    "probability": 0.55,
    "margin": 0.15,
    "uncertainty_reasons": ["limited_data", "time_until_event"],
    "raw_reasoning": "Based on recent polls showing upward trend...",
    "decomposition": "0.50 (base) + 0.10 (polls) - 0.05 (trend) = 0.55",
    "is_mispriced": "yes",
    "direction": "YES_underpriced",
    "edge": 0.10
}


class TestClaudeSignalSchema:
    """Test schema validation."""

    def test_valid_schema(self):
        """Test that valid forecast passes schema validation."""
        from jsonschema import validate
        # Should not raise
        validate(instance=VALID_FORECAST, schema=FORECAST_SCHEMA)

    def test_missing_required_field(self):
        """Test that missing required field fails validation."""
        from jsonschema import ValidationError
        invalid = VALID_FORECAST.copy()
        del invalid["base_rate"]

        with pytest.raises(ValidationError):
            from jsonschema import validate
            validate(instance=invalid, schema=FORECAST_SCHEMA)

    def test_invalid_probability_range(self):
        """Test that probability outside [0,1] fails validation."""
        from jsonschema import ValidationError
        invalid = VALID_FORECAST.copy()
        invalid["probability"] = 1.5

        with pytest.raises(ValidationError):
            from jsonschema import validate
            validate(instance=invalid, schema=FORECAST_SCHEMA)

    def test_invalid_direction_enum(self):
        """Test that invalid direction value fails validation."""
        from jsonschema import ValidationError
        invalid = VALID_FORECAST.copy()
        invalid["direction"] = "INVALID_DIRECTION"

        with pytest.raises(ValidationError):
            from jsonschema import validate
            validate(instance=invalid, schema=FORECAST_SCHEMA)

    def test_factors_structure(self):
        """Test that factors array validates correctly."""
        from jsonschema import ValidationError
        invalid = VALID_FORECAST.copy()
        invalid["factors"] = [{"name": "test"}]  # Missing impact and evidence_count

        with pytest.raises(ValidationError):
            from jsonschema import validate
            validate(instance=invalid, schema=FORECAST_SCHEMA)


class TestClaudeSignalDecomposition:
    """Test probability decomposition validation."""

    def test_validate_decomposition_correct(self):
        """Test decomposition validation with correct values."""
        signal = ClaudeSignal()
        data = VALID_FORECAST.copy()

        # This should pass (0.50 + 0.10 - 0.05 = 0.55)
        result = signal._validate_decomposition(data)
        assert result is True
        assert "decomposition_mismatch" not in data

    def test_validate_decomposition_mismatch(self):
        """Test decomposition validation with mismatch."""
        signal = ClaudeSignal()
        data = VALID_FORECAST.copy()
        data["probability"] = 0.70  # Doesn't match 0.50 + 0.10 - 0.05

        result = signal._validate_decomposition(data)
        assert result is True  # Still returns True but auto-corrects
        assert data["decomposition_mismatch"] is True
        # Should auto-correct to calculated value
        assert abs(data["probability"] - 0.55) < 0.01

    def test_validate_decomposition_clamping(self):
        """Test that decomposition clamps to [0,1]."""
        signal = ClaudeSignal()
        data = VALID_FORECAST.copy()
        data["base_rate"] = 0.1
        data["factors"] = [{"name": "huge_factor", "impact": 2.0, "evidence_count": 1}]
        data["probability"] = 2.1  # Over 1.0

        result = signal._validate_decomposition(data)
        assert result is True
        # Should clamp to 1.0
        assert data["probability"] == 1.0


class TestClaudeSignalEdge:
    """Test edge calculation validation."""

    def test_validate_edge_correct(self):
        """Test edge validation with correct calculation."""
        signal = ClaudeSignal()
        data = VALID_FORECAST.copy()
        data["probability"] = 0.60
        data["edge"] = 0.10
        market_prob = 0.50

        result = signal._validate_edge(data, market_prob)
        assert result is True
        assert abs(data["edge"] - 0.10) < 0.001

    def test_validate_edge_mismatch(self):
        """Test edge validation with mismatch."""
        signal = ClaudeSignal()
        data = VALID_FORECAST.copy()
        data["probability"] = 0.60
        data["edge"] = 0.20  # Wrong, should be 0.10
        market_prob = 0.50

        result = signal._validate_edge(data, market_prob)
        assert result is True  # Still returns True but auto-corrects
        assert abs(data["edge"] - 0.10) < 0.001  # Corrected


class TestClaudeSignalAggregation:
    """Test self-consistency aggregation."""

    def test_aggregate_single_run(self):
        """Test aggregation with single run."""
        signal = ClaudeSignal()
        runs = [
            ForecastRun(
                probability=0.60,
                confidence=0.80,
                edge=0.10,
                base_rate=0.50,
                factors=[],
                raw_data={"margin": 0.15, "market_prob": 0.50}
            )
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        assert abs(final_prob - 0.60) < 0.01  # No variance penalty
        assert metadata["n_runs"] == 1
        assert metadata["var_prob"] == 0.0

    def test_aggregate_consistent_runs(self):
        """Test aggregation with consistent runs."""
        signal = ClaudeSignal(variance_penalty_k=0.5)
        runs = [
            ForecastRun(0.60, 0.80, 0.10, 0.50, [], {"margin": 0.15, "market_prob": 0.50}),
            ForecastRun(0.61, 0.81, 0.11, 0.50, [], {"margin": 0.15, "market_prob": 0.50}),
            ForecastRun(0.59, 0.79, 0.09, 0.50, [], {"margin": 0.15, "market_prob": 0.50})
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        assert 0.55 < final_prob < 0.65  # Around mean with small penalty
        assert metadata["n_runs"] == 3
        assert metadata["std_prob"] < 0.02  # Low variance
        assert final_conf > 0.7  # High confidence due to consistency

    def test_aggregate_inconsistent_runs(self):
        """Test aggregation with inconsistent runs."""
        signal = ClaudeSignal(variance_penalty_k=0.5)
        runs = [
            ForecastRun(0.30, 0.70, -0.20, 0.50, [], {"margin": 0.20, "market_prob": 0.50}),
            ForecastRun(0.70, 0.75, 0.20, 0.50, [], {"margin": 0.20, "market_prob": 0.50}),
            ForecastRun(0.50, 0.65, 0.00, 0.50, [], {"margin": 0.20, "market_prob": 0.50})
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        # High variance should reduce both probability and confidence
        mean_prob = 0.50
        assert final_prob < mean_prob  # Variance penalty applied
        assert metadata["std_prob"] > 0.10  # High variance
        assert final_conf < 0.7  # Lower confidence due to inconsistency

    def test_aggregate_variance_penalty_applied(self):
        """Test that variance penalty is correctly applied."""
        signal = ClaudeSignal(variance_penalty_k=0.5)
        runs = [
            ForecastRun(0.40, 0.70, 0.0, 0.40, [], {"margin": 0.20, "market_prob": 0.40}),
            ForecastRun(0.60, 0.70, 0.0, 0.40, [], {"margin": 0.20, "market_prob": 0.40})
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        mean_prob = 0.50
        var_prob = 0.01  # (0.40-0.50)^2 + (0.60-0.50)^2 / 2
        expected_penalty = 0.5 * (var_prob ** 0.5)

        assert abs(final_prob - (mean_prob - expected_penalty)) < 0.01
        assert metadata["variance_penalty"] > 0


class TestClaudeSignalDisagreementPenalty:
    """Test cross-signal disagreement penalty."""

    def test_no_penalty_when_close(self):
        """Test no penalty when close to consensus."""
        signal = ClaudeSignal(
            disagreement_threshold_low=0.03,
            disagreement_threshold_high=0.10
        )
        signal.consensus_prob = 0.60

        prob = 0.62  # Within 0.03
        conf = 0.80
        metadata = {}

        adjusted_conf, meta = signal._apply_disagreement_penalty(prob, conf, metadata)

        assert adjusted_conf == 0.80  # No change
        assert meta["disagreement_penalty"] == 0.0

    def test_moderate_penalty(self):
        """Test moderate penalty for medium disagreement."""
        signal = ClaudeSignal(
            disagreement_threshold_low=0.03,
            disagreement_threshold_high=0.10
        )
        signal.consensus_prob = 0.60

        prob = 0.65  # Between 0.03 and 0.10
        conf = 0.80
        metadata = {}

        adjusted_conf, meta = signal._apply_disagreement_penalty(prob, conf, metadata)

        assert adjusted_conf == 0.80 * 0.8  # 20% penalty
        assert meta["disagreement_penalty"] == 0.2

    def test_high_penalty(self):
        """Test high penalty for large disagreement."""
        signal = ClaudeSignal(
            disagreement_threshold_low=0.03,
            disagreement_threshold_high=0.10
        )
        signal.consensus_prob = 0.60

        prob = 0.75  # Over 0.10
        conf = 0.80
        metadata = {}

        adjusted_conf, meta = signal._apply_disagreement_penalty(prob, conf, metadata)

        assert adjusted_conf == 0.80 * 0.5  # 50% penalty
        assert meta["disagreement_penalty"] == 0.5

    def test_no_consensus_available(self):
        """Test no penalty when consensus not available."""
        signal = ClaudeSignal()
        signal.consensus_prob = None

        prob = 0.99
        conf = 0.80
        metadata = {}

        adjusted_conf, meta = signal._apply_disagreement_penalty(prob, conf, metadata)

        assert adjusted_conf == 0.80  # No change
        assert meta["disagreement_penalty"] == 0.0
        assert meta["consensus_prob"] is None


class TestClaudeSignalRetryLogic:
    """Test retry and fallback mechanisms."""

    @patch('better_manifold_bot.signals.claude_signal.anthropic.Anthropic')
    def test_retry_on_json_error(self, mock_anthropic):
        """Test retry logic when JSON parsing fails."""
        signal = ClaudeSignal(n_runs=1)

        # Mock client to return invalid JSON first, then valid
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="invalid json")]

        mock_client.messages.create.side_effect = [
            mock_response,  # First call fails
            MagicMock(content=[MagicMock(text=json.dumps(VALID_FORECAST))])  # Second succeeds
        ]

        signal.client = mock_client

        result = signal._call_claude("test prompt")

        # Should retry and eventually succeed
        assert mock_client.messages.create.call_count == 2

    @patch('better_manifold_bot.signals.claude_signal.anthropic.Anthropic')
    def test_fallback_on_all_failures(self, mock_anthropic):
        """Test fallback when all retries fail."""
        signal = ClaudeSignal(n_runs=2)

        # Mock client to always fail
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API Error")
        signal.client = mock_client

        market = {
            "question": "Test question?",
            "description": "Test description",
            "probability": 0.50
        }

        result = signal.analyze(market)

        # Should return fallback result
        assert result is not None
        assert result.probability == 0.50  # Market probability
        assert result.confidence < 0.5  # Low confidence
        assert "failed" in result.reasoning.lower()


class TestClaudeSignalIntegration:
    """Integration tests."""

    @patch('better_manifold_bot.signals.claude_signal.anthropic.Anthropic')
    def test_full_analysis_flow(self, mock_anthropic):
        """Test complete analysis flow."""
        signal = ClaudeSignal(n_runs=2)

        # Mock successful Claude responses
        mock_client = MagicMock()
        forecast1 = VALID_FORECAST.copy()
        forecast1["probability"] = 0.60
        forecast1["edge"] = 0.10

        forecast2 = VALID_FORECAST.copy()
        forecast2["probability"] = 0.62
        forecast2["edge"] = 0.12

        mock_client.messages.create.side_effect = [
            MagicMock(content=[MagicMock(text=json.dumps(forecast1))]),
            MagicMock(content=[MagicMock(text=json.dumps(forecast2))])
        ]
        signal.client = mock_client

        market = {
            "question": "Will it rain tomorrow?",
            "description": "Testing market",
            "probability": 0.50
        }

        # Create mock other signals
        historical_result = SignalResult(0.55, 0.70, "Historical analysis")
        micro_result = SignalResult(0.53, 0.60, "Microstructure analysis")

        result = signal.analyze(market, historical_result, micro_result)

        assert result is not None
        assert 0.0 <= result.probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert "Claude ensemble" in result.reasoning
        assert result.metadata["n_runs"] == 2

    def test_prompt_creation(self):
        """Test user prompt creation."""
        prompt = create_user_prompt(
            "Will it rain?",
            "Weather forecast market",
            0.60
        )

        assert "Will it rain?" in prompt
        assert "Weather forecast market" in prompt
        assert "60.0%" in prompt or "0.60" in prompt
        assert "base_rate" in prompt
        assert "factors" in prompt
        assert "edge" in prompt


class TestClaudeSignalEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_runs_list(self):
        """Test aggregation with empty runs list."""
        signal = ClaudeSignal()

        with pytest.raises(ValueError, match="No runs to aggregate"):
            signal._aggregate_runs([])

    def test_confidence_clamping(self):
        """Test that confidence is clamped to max_confidence."""
        signal = ClaudeSignal(max_confidence=0.90)
        runs = [
            ForecastRun(0.60, 0.95, 0.10, 0.50, [], {"margin": 0.05, "market_prob": 0.50})
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        assert final_conf <= 0.90  # Clamped to max

    def test_probability_clamping(self):
        """Test that probability is clamped to [0, 1]."""
        signal = ClaudeSignal(variance_penalty_k=10.0)  # Very high penalty
        runs = [
            ForecastRun(0.10, 0.70, 0.0, 0.10, [], {"margin": 0.20, "market_prob": 0.10}),
            ForecastRun(0.12, 0.70, 0.0, 0.10, [], {"margin": 0.20, "market_prob": 0.10})
        ]

        final_prob, final_conf, metadata = signal._aggregate_runs(runs)

        assert 0.0 <= final_prob <= 1.0  # Must be in valid range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
