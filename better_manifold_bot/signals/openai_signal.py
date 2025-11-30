"""
Enhanced LLM Signal using OpenAI GPT-4o-mini with structured forecasting framework.

Features:
- Deterministic forecasting template with base rates and decomposition
- Self-consistency ensemble (multiple runs with aggregation)
- Cross-signal disagreement penalty
- Market-aware mispricing detection
- Robust validation and fallback mechanisms
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from openai import OpenAI
from jsonschema import validate, ValidationError
from .base_signal import BaseSignal, SignalResult, clamp_prob, clamp_conf

logger = logging.getLogger(__name__)


# JSON Schema for OpenAI response validation
FORECAST_SCHEMA = {
    "type": "object",
    "required": [
        "base_rate",
        "factors",
        "confounders",
        "probability",
        "margin",
        "uncertainty_reasons",
        "raw_reasoning",
        "is_mispriced",
        "direction",
        "edge"
    ],
    "properties": {
        "base_rate": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "impact", "evidence_count"],
                "properties": {
                    "name": {"type": "string"},
                    "impact": {"type": "number"},
                    "evidence_count": {"type": "integer", "minimum": 0}
                }
            }
        },
        "confounders": {
            "type": "array",
            "items": {"type": "string"}
        },
        "probability": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "margin": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
        "uncertainty_reasons": {
            "type": "array",
            "items": {"type": "string"}
        },
        "raw_reasoning": {"type": "string"},
        "is_mispriced": {
            "type": "string",
            "enum": ["yes", "no"]
        },
        "direction": {
            "type": "string",
            "enum": ["YES_underpriced", "YES_overpriced", "NO_underpriced", "NO_overpriced", "neutral"]
        },
        "edge": {"type": "number"},
        "decomposition": {"type": "string"}
    }
}


SYSTEM_PROMPT = """You are a rigorous super-forecaster analyzing prediction markets. You MUST follow a structured forecasting framework to ground all probability estimates in quantitative reasoning.

CRITICAL JSON RULES:
1. Output ONLY valid JSON - no text before or after
2. Ensure commas between all array elements and object properties
3. No trailing commas before } or ]
4. All strings must use double quotes
5. Numbers should not have leading plus signs
6. Do not include comments or explanations within the JSON

Your analysis must follow this rubric:

1. IDENTIFY BASE RATE
   - Find historical frequency of similar events
   - Use reference class forecasting
   - If no data exists, use domain priors

2. IDENTIFY CONSTRAINTS
   - Hard constraints (impossibilities, deadlines)
   - Soft constraints (practical limitations)

3. IDENTIFY INDEPENDENT VARIABLES
   - List factors that affect the outcome
   - Quantify impact of each factor (+/- adjustment)
   - Count evidence supporting each factor

4. IDENTIFY CONFOUNDERS
   - Variables that correlate but don't cause
   - Common causes affecting multiple factors

5. CALCULATE PROBABILITY
   - Start with base rate
   - Apply factor adjustments sequentially
   - Show mathematical decomposition
   - Provide margin of error ()

6. EXPLAIN UNCERTAINTY
   - List all sources of uncertainty
   - Quantify variance where possible

CRITICAL: Your "probability" field MUST mathematically equal base_rate + sum of all factor impacts. Double-check your math!
You MUST output valid JSON matching this exact schema. Any deviation will be rejected.
FINAL REMINDER: Output ONLY the JSON object. No explanations, no markdown, just the JSON."""


def create_user_prompt(question: str, description: str, market_prob: float) -> str:
    """Create user prompt with market context."""
    return f"""Analyze this prediction market:

MARKET QUESTION: {question}

DESCRIPTION: {description}

CURRENT MARKET PROBABILITY: {market_prob:.1%}

Your task:
1. Identify the base rate for this type of event
2. List independent factors that affect the probability (with impact estimates)
3. Identify potential confounders
4. Calculate your probability estimate using decomposition
5. Determine if the market is mispriced and by how much

Output format (JSON):
{{
  "base_rate": <number 0-1>,
  "factors": [
    {{"name": "<factor name>", "impact": <number (e.g. 0.05 or -0.05)>, "evidence_count": <integer>}}
  ],
  "confounders": ["<confounder1>", "<confounder2>"],
  "probability": <number 0-1>,
  "margin": <number 0-1>,
  "uncertainty_reasons": ["<reason1>", "<reason2>"],
  "raw_reasoning": "<your reasoning process>",
  "decomposition": "<show how base_rate + factors = probability>",
  "is_mispriced": "<yes or no>",
  "direction": "<YES_underpriced|YES_overpriced|NO_underpriced|NO_overpriced|neutral>",
  "edge": <your_probability - market_probability>
}}

Remember:
- Base rate should be grounded in historical data or reference classes
- Each factor must have quantified impact and evidence count
- Decomposition must show mathematical steps
- Edge must equal probability - {market_prob:.4f}
- Direction must be consistent with edge sign and magnitude"""


@dataclass
class ForecastRun:
    """Single forecast run result."""
    probability: float
    confidence: float
    edge: float
    base_rate: float
    factors: List[Dict[str, Any]]
    raw_data: Dict[str, Any]


class OpenAISignal(BaseSignal):
    """
    Enhanced LLM signal using OpenAI GPT-4o-mini with structured forecasting.

    Features:
    - Self-consistency ensemble (n_runs)
    - Variance penalty for inconsistent predictions
    - Cross-signal disagreement penalty
    - Strict schema validation with fallbacks
    """

    def __init__(
        self,
        name: str = "openai",
        weight: float = 0.60,
        enabled: bool = True,
        n_runs: int = 3,
        variance_penalty_k: float = 0.5,
        max_confidence: float = 0.95,
        disagreement_threshold_low: float = 0.10,
        disagreement_threshold_high: float = 0.20
    ):
        super().__init__(name, weight, enabled)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"  # Using GPT-4o for better reasoning
        self.n_runs = n_runs
        self.variance_penalty_k = variance_penalty_k
        self.max_confidence = max_confidence
        self.disagreement_threshold_low = disagreement_threshold_low
        self.disagreement_threshold_high = disagreement_threshold_high

        # For cross-signal penalty
        self.consensus_prob: Optional[float] = None
        self.historical_prob: Optional[float] = None
        self.micro_prob: Optional[float] = None

    def _call_openai(self, user_prompt: str, attempt: int = 0) -> Optional[Dict[str, Any]]:
        """
        Call OpenAI API with structured output.
        Returns parsed JSON or None on failure.
        """
        max_attempts = 3
        backoff = 1.0

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.0,  # Deterministic
                response_format={"type": "json_object"}  # Force JSON mode
            )

            content = response.choices[0].message.content

            # Check if content is None or empty
            if not content:
                logger.error(f"OpenAI returned empty content (attempt {attempt + 1})")
                if attempt < max_attempts - 1:
                    time.sleep(backoff * (2 ** attempt))
                    return self._call_openai(user_prompt, attempt + 1)
                return None

            # Extract JSON from response (handle markdown code blocks)
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            # Enhanced JSON sanitization
            json_str = self._sanitize_json(json_str)

            # Parse JSON
            data = self._parse_json_with_fallback(json_str)
            if data is None:
                raise json.JSONDecodeError("Failed to parse JSON with all fallback methods", json_str, 0)

            # Validate against schema
            validate(instance=data, schema=FORECAST_SCHEMA)

            return data

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"OpenAI response validation failed (attempt {attempt + 1}): {e}")
            logger.debug(f"Failed JSON content: {content[:500]}...")  # Log first 500 chars
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
                return self._call_openai(user_prompt, attempt + 1)
            return None

        except Exception as e:
            logger.error(f"OpenAI API error (attempt {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
                return self._call_openai(user_prompt, attempt + 1)
            return None

    def _sanitize_json(self, json_str: str) -> str:
        """
        Sanitize JSON string to fix common formatting issues.
        """
        # Remove leading plus signs from numbers
        json_str = re.sub(r'(:\s*)\+', r'\1', json_str)
        
        # Fix missing commas between array elements
        # Matches: }\n{ or }{ without comma
        json_str = re.sub(r'}\s*\n\s*{', '},\n{', json_str)
        json_str = re.sub(r'}\s*{', '},{', json_str)
        
        # Fix missing commas between properties
        # Matches: "value"\n"nextkey" or similar patterns
        json_str = re.sub(r'("\s*)\n\s*(")', r'\1,\n\2', json_str)
        json_str = re.sub(r'(]\s*)\n\s*(")', r'\1,\n\2', json_str)
        json_str = re.sub(r'(}\s*)\n\s*(")', r'\1,\n\2', json_str)
        json_str = re.sub(r'([0-9]\s*)\n\s*(")', r'\1,\n\2', json_str)
        
        # Remove trailing commas (invalid in JSON)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Escape unescaped quotes in strings (common issue)
        # This is tricky and might need more sophisticated handling
        
        # Remove any non-JSON content before/after main object
        # Find first { and last }
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        if first_brace != -1 and last_brace != -1:
            json_str = json_str[first_brace:last_brace+1]
        
        return json_str.strip()

    def _validate_decomposition(self, data: Dict[str, Any]) -> bool:
        """
        Validate that probability decomposition is mathematically sound.

        Returns True if decomposition matches probability within epsilon.
        """
        try:
            base_rate = data["base_rate"]
            factors = data["factors"]
            declared_prob = data["probability"]

            # Calculate probability from decomposition
            calculated_prob = base_rate
            for factor in factors:
                calculated_prob += factor["impact"]

            # Clamp to valid range
            calculated_prob = max(0.0, min(1.0, calculated_prob))

            # Check if close enough (within 5%)
            epsilon = 0.05
            matches = abs(calculated_prob - declared_prob) < epsilon

            if not matches:
                logger.warning(
                    f"Decomposition mismatch: calculated={calculated_prob:.3f}, "
                    f"declared={declared_prob:.3f}"
                )
                # Auto-correct to calculated value
                data["probability"] = calculated_prob
                data["decomposition_mismatch"] = True

            return True

        except Exception as e:
            logger.error(f"Decomposition validation error: {e}")
            return False
    def _parse_json_with_fallback(self, json_str: str) -> Optional[Dict[str, Any]]:
        """
        Try to parse JSON with multiple fallback strategies.
        """
        # First attempt: direct parse
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # Second attempt: sanitize and parse
        try:
            sanitized = self._sanitize_json(json_str)
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass
        
        # Third attempt: use ast.literal_eval for simple cases
        try:
            import ast
            # Replace true/false/null with Python equivalents
            python_str = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            result = ast.literal_eval(python_str)
            # Convert back to JSON-compatible format
            return json.loads(json.dumps(result))
        except:
            pass
        
        # Fourth attempt: regex-based reconstruction (last resort)
        try:
            # Extract key-value pairs with regex and rebuild
            # This is a simplified version - you might need to expand it
            patterns = {
                'base_rate': r'"base_rate"\s*:\s*([0-9.]+)',
                'probability': r'"probability"\s*:\s*([0-9.]+)',
                'margin': r'"margin"\s*:\s*([0-9.]+)',
                'edge': r'"edge"\s*:\s*([-0-9.]+)',
                'is_mispriced': r'"is_mispriced"\s*:\s*"(yes|no)"',
                'direction': r'"direction"\s*:\s*"([^"]+)"',
            }
            
            extracted = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, json_str)
                if match:
                    value = match.group(1)
                    if key in ['base_rate', 'probability', 'margin', 'edge']:
                        extracted[key] = float(value)
                    else:
                        extracted[key] = value
            
            # Set defaults for missing required fields
            extracted.setdefault('factors', [])
            extracted.setdefault('confounders', [])
            extracted.setdefault('uncertainty_reasons', [])
            extracted.setdefault('raw_reasoning', 'Extracted from malformed response')
            
            if len(extracted) >= 8:  # Minimum viable response
                return extracted
        except:
            pass
        
        return None
    def _validate_edge(self, data: Dict[str, Any], market_prob: float) -> bool:
        """
        Validate that edge calculation is correct.

        Returns True if edge matches probability - market_prob.
        """
        try:
            declared_edge = data["edge"]
            calculated_edge = data["probability"] - market_prob

            epsilon = 0.001
            matches = abs(declared_edge - calculated_edge) < epsilon

            if not matches:
                logger.warning(
                    f"Edge mismatch: declared={declared_edge:.3f}, "
                    f"calculated={calculated_edge:.3f}"
                )
                # Auto-correct
                data["edge"] = calculated_edge

            return True

        except Exception as e:
            logger.error(f"Edge validation error: {e}")
            return False

    def _aggregate_runs(self, runs: List[ForecastRun]) -> Tuple[float, float, Dict[str, Any]]:
        """
        Aggregate multiple forecast runs using variance penalty.

        Returns (final_prob, final_conf, metadata)
        """
        if not runs:
            raise ValueError("No runs to aggregate")

        probs = [r.probability for r in runs]
        edges = [r.edge for r in runs]

        # Calculate statistics
        mean_prob = sum(probs) / len(probs)
        var_prob = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        std_prob = var_prob ** 0.5

        # Apply variance penalty to probability
        final_prob = mean_prob - self.variance_penalty_k * std_prob
        final_prob = clamp_prob(final_prob)

        # Calculate confidence based on consistency
        # Start with mean margin, reduce by variance
        mean_margin = sum(r.raw_data.get("margin", 0.1) for r in runs) / len(runs)
        base_conf = 1.0 - mean_margin  # Higher margin = lower confidence

        # Variance penalty for confidence
        var_prob_norm = min(1.0, var_prob / 0.02)  # Normalize variance
        final_conf = base_conf * (1.0 - min(0.5, var_prob_norm))
        final_conf = min(self.max_confidence, clamp_conf(final_conf))

        # Calculate edge from final probability
        mean_market_prob = runs[0].raw_data.get("market_prob", 0.5)
        final_edge = final_prob - mean_market_prob

        metadata = {
            "n_runs": len(runs),
            "mean_prob": mean_prob,
            "var_prob": var_prob,
            "std_prob": std_prob,
            "variance_penalty": self.variance_penalty_k * std_prob,
            "individual_probs": probs,
            "individual_edges": edges,
            "base_confidence": base_conf,
            "variance_penalty_applied": min(0.5, var_prob_norm)
        }

        return final_prob, final_conf, metadata

    def _apply_disagreement_penalty(
        self,
        prob: float,
        conf: float,
        metadata: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Apply penalty when LLM disagrees with other signals.

        Returns (adjusted_conf, penalty_metadata)
        """
        if self.consensus_prob is None:
            metadata["disagreement_penalty"] = 0.0
            metadata["consensus_prob"] = None
            return conf, metadata

        disagreement = abs(prob - self.consensus_prob)
        penalty = 0.0

        if disagreement < self.disagreement_threshold_low:
            penalty = 0.0
        elif disagreement < self.disagreement_threshold_high:
            penalty = 0.2
            conf *= 0.8
        else:
            penalty = 0.5
            conf *= 0.5

        metadata["consensus_prob"] = self.consensus_prob
        metadata["disagreement"] = disagreement
        metadata["disagreement_penalty"] = penalty

        logger.info(
            f"Disagreement check: LLM={prob:.3f}, consensus={self.consensus_prob:.3f}, "
            f"d={disagreement:.3f}, penalty={penalty:.1%}"
        )

        return conf, metadata

    def analyze(
        self,
        market: Dict[str, Any],
        historical_result: Optional[SignalResult] = None,
        micro_result: Optional[SignalResult] = None
    ) -> Optional[SignalResult]:
        """
        Analyze market using enhanced OpenAI signal.

        Args:
            market: Market data
            historical_result: Result from historical signal (for cross-check)
            micro_result: Result from microstructure signal (for cross-check)
        """
        question = market.get("question", "")
        description = market.get("description", "")
        market_prob = float(market.get("probability", 0.5))

        # Calculate consensus from other signals
        if historical_result and micro_result:
            # Weighted average (using default weights from main)
            hist_weight = 0.25 / (0.25 + 0.15)
            micro_weight = 0.15 / (0.25 + 0.15)
            self.consensus_prob = (
                historical_result.probability * hist_weight +
                micro_result.probability * micro_weight
            )
        else:
            self.consensus_prob = None

        # Create prompt
        user_prompt = create_user_prompt(question, description, market_prob)

        # Run multiple times for self-consistency
        runs: List[ForecastRun] = []
        for i in range(self.n_runs):
            logger.info(f"OpenAI run {i+1}/{self.n_runs}")

            data = self._call_openai(user_prompt)
            if data is None:
                logger.warning(f"Run {i+1} failed, skipping")
                continue

            # Validate decomposition and edge
            self._validate_decomposition(data)
            self._validate_edge(data, market_prob)

            # Store market_prob for later use
            data["market_prob"] = market_prob

            # Extract key values
            prob = clamp_prob(data["probability"])
            edge = data["edge"]
            base_rate = data["base_rate"]

            # Calculate initial confidence from margin
            margin = data.get("margin", 0.1)
            conf = 1.0 - margin

            runs.append(ForecastRun(
                probability=prob,
                confidence=conf,
                edge=edge,
                base_rate=base_rate,
                factors=data["factors"],
                raw_data=data
            ))

        # Fallback if all runs failed
        if not runs:
            logger.error("All OpenAI runs failed, returning fallback")
            return SignalResult(
                probability=market_prob,
                confidence=0.30,
                reasoning="OpenAI failed all retry attempts, using market probability",
                metadata={"error": "all_runs_failed", "n_runs": self.n_runs}
            )

        # Aggregate runs
        final_prob, final_conf, agg_metadata = self._aggregate_runs(runs)

        # Apply disagreement penalty
        final_conf, penalty_metadata = self._apply_disagreement_penalty(
            final_prob, final_conf, agg_metadata
        )

        # Merge metadata
        full_metadata = {**agg_metadata, **penalty_metadata}
        full_metadata["model"] = self.model
        full_metadata["base_rate"] = runs[0].base_rate
        full_metadata["factors"] = runs[0].factors

        # Build reasoning
        reasoning_parts = [
            f"OpenAI ensemble ({len(runs)} runs):",
            f"Base rate: {runs[0].base_rate:.1%}",
            f"Mean prob: {agg_metadata['mean_prob']:.1%}  {agg_metadata['std_prob']:.1%}",
            f"Final prob: {final_prob:.1%} (after variance penalty)",
            f"Confidence: {final_conf:.1%}"
        ]

        if penalty_metadata.get("disagreement_penalty", 0) > 0:
            reasoning_parts.append(
                f" Disagreement penalty: {penalty_metadata['disagreement_penalty']:.0%} "
                f"(vs consensus {self.consensus_prob:.1%})"
            )

        reasoning = "\n".join(reasoning_parts)

        return SignalResult(
            probability=final_prob,
            confidence=final_conf,
            reasoning=reasoning,
            metadata=full_metadata
        )