"""
Enhanced Ensemble Decision Maker with cross-signal integration.

Supports passing signal results between signals for cross-validation.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from manifoldbot.manifold.bot import DecisionMaker, MarketDecision

logger = logging.getLogger(__name__)


class EnhancedEnsembleDecisionMaker(DecisionMaker):
    """
    Enhanced ensemble decision maker with cross-signal validation.

    Features:
    - Sequential signal execution (allows cross-signal dependency)
    - LLM signal (Claude/OpenAI) receives other signals' results for disagreement penalty
    - Weighted Bayesian aggregation
    """

    def __init__(
        self,
        signals: List[Any],
        min_confidence: float = 0.65,
        min_edge: float = 0.10
    ):
        """
        Initialize ensemble with signals.

        Args:
            signals: List of signal objects (ordered by execution)
            min_confidence: Minimum confidence to bet
            min_edge: Minimum probability difference to bet
        """
        self.signals = signals
        self.min_confidence = min_confidence
        self.min_edge = min_edge

    def analyze_market(self, market: Dict[str, Any]) -> MarketDecision:
        """
        Analyze market using all signals with cross-validation.

        Signals are executed in order:
        1. Historical signal
        2. Microstructure signal
        3. LLM signal (Claude/OpenAI) (receives 1 & 2 for cross-check)

        Args:
            market: Market data from Manifold API

        Returns:
            MarketDecision with ensemble prediction
        """
        market_id = market.get("id", "")
        question = market.get("question", "")
        current_prob = market.get("probability", 0.5)
        outcome_type = market.get("outcomeType", "UNKNOWN")

        # Execute signals in order, collecting results
        signal_results = []
        historical_result = None
        micro_result = None
        llm_result = None

        total_weight = 0.0

        for signal in self.signals:
            if not signal.enabled:
                continue

            try:
                # Check if this is LLM signal (Claude or OpenAI) - needs cross-signal data
                if signal.name in ["claude", "openai"]:
                    result = signal.analyze(
                        market,
                        historical_result=historical_result,
                        micro_result=micro_result
                    )
                else:
                    result = signal.analyze(market)

                if result is not None:
                    signal_results.append(result)
                    total_weight += signal.weight

                    # Store for cross-signal use
                    if signal.name == "historical":
                        historical_result = result
                    elif signal.name == "microstructure":
                        micro_result = result
                    elif signal.name in ["claude", "openai"]:
                        llm_result = result

            except Exception as e:
                logger.error(f"Error in signal {signal.name}: {e}", exc_info=True)

        # Store for external access (e.g., logging)
        self.last_signal_results = signal_results

        if not signal_results:
            return MarketDecision(
                market_id=market_id,
                question=question,
                current_probability=current_prob,
                decision="SKIP",
                confidence=0.0,
                reasoning="No signals available",
                outcome_type=outcome_type
            )

        # Weighted average of probabilities
        ensemble_prob = sum(
            r.probability * s.weight
            for r, s in zip(signal_results, [sig for sig in self.signals if sig.enabled])
            if s.enabled
        ) / total_weight

        # Weighted average of confidences
        ensemble_confidence = sum(
            r.confidence * s.weight
            for r, s in zip(signal_results, [sig for sig in self.signals if sig.enabled])
            if s.enabled
        ) / total_weight

        # Calculate edge
        edge = abs(ensemble_prob - current_prob)

        # Make decision
        if ensemble_confidence < self.min_confidence or edge < self.min_edge:
            decision = "SKIP"
            reasoning = f"Low confidence ({ensemble_confidence:.0%}) or edge ({edge:.0%})"
        elif ensemble_prob > current_prob + self.min_edge:
            decision = "YES"
            reasoning = f"Ensemble: {ensemble_prob:.0%} vs Market: {current_prob:.0%}"
        elif ensemble_prob < current_prob - self.min_edge:
            decision = "NO"
            reasoning = f"Ensemble: {ensemble_prob:.0%} vs Market: {current_prob:.0%}"
        else:
            decision = "SKIP"
            reasoning = f"Market fairly priced ({current_prob:.0%})"

        # Aggregate reasoning from signals
        signal_reasoning = "\n".join([
            f"- {s.name}: {r.probability:.0%} (conf: {r.confidence:.0%})"
            for r, s in zip(signal_results, [sig for sig in self.signals if sig.enabled])
        ])
        full_reasoning = f"{reasoning}\n\nSignals:\n{signal_reasoning}"

        # Add disagreement warning if LLM was heavily penalized
        if llm_result and "disagreement_penalty" in llm_result.metadata:
            penalty = llm_result.metadata["disagreement_penalty"]
            if penalty >= 0.2:
                full_reasoning += f"\n\n LLM disagreement penalty: {penalty:.0%}"

        return MarketDecision(
            market_id=market_id,
            question=question,
            current_probability=current_prob,
            decision=decision,
            confidence=ensemble_confidence,
            reasoning=full_reasoning,
            outcome_type=outcome_type,
            metadata={
                "ensemble_probability": ensemble_prob,
                "edge": edge,
                "num_signals": len(signal_results),
                "signal_details": {
                    s.name: {
                        "probability": r.probability,
                        "confidence": r.confidence,
                        "weight": s.weight
                    }
                    for r, s in zip(signal_results, [sig for sig in self.signals if sig.enabled])
                }
            }
        )


# Alias for backwards compatibility
EnsembleDecisionMaker = EnhancedEnsembleDecisionMaker
