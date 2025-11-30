"""LLM Signal using GPT-4 for analysis."""

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from openai import OpenAI
from .base_signal import BaseSignal, SignalResult, clamp_prob, clamp_conf

logger = logging.getLogger(__name__)


class LLMSignal(BaseSignal):
    """LLM-based signal using OpenAI GPT-4o-mini."""

    def __init__(self, name: str = "llm", weight: float = 0.40, enabled: bool = True):
        super().__init__(name, weight, enabled)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o-mini"

    # --------------------------------------------------------
    # SAFE PARSER
    # --------------------------------------------------------
    @staticmethod
    def _safe_parse_result(raw: Dict[str, Any]):
        """Extract (prob, conf, reasoning) safely from dict or text."""

        try:
            prob = float(raw.get("probability"))
            conf = float(raw.get("confidence"))
            reason = raw.get("reasoning", "") or raw.get("explanation", "")
            return prob, conf, reason
        except Exception:
            pass

        # Fallback: raw text mode
        text = raw.get("text") if isinstance(raw, dict) else str(raw)
        reason = text
        prob, conf = None, None

        import re
        m = re.search(r"prob(?:ability)?[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
        if m:
            prob = float(m.group(1))

        m2 = re.search(r"conf(?:idence)?[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
        if m2:
            conf = float(m2.group(1))

        return prob, conf, reason

    # --------------------------------------------------------
    # MAIN ANALYSIS METHOD
    # --------------------------------------------------------
    def analyze(self, market: Dict[str, Any]) -> Optional[SignalResult]:

        question = market.get("question") or ""
        description = market.get("description", "")
        current_prob = float(market.get("probability", 0.5) or 0.5)

        system_msg = (
            "You are a careful super-forecaster. "
            "Output JSON with keys: probability, confidence, reasoning."
        )

        user_prompt = (
            f"Market: {question}\n"
            f"Description: {description}\n"
            f"Current Probability: {current_prob:.2f}\n\n"
            "Return a JSON object like: "
            "{\"probability\":0.0-1.0, \"confidence\":0.0-1.0, \"reasoning\":\"...\"}"
        )

        attempts = 0
        backoff = 1.0
        max_attempts = 3

        while attempts < max_attempts:
            try:
                completion = self.client.beta.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.0
                )

                message = completion.choices[0].message
                parsed = getattr(message, "parsed", None)

                # Parse structured object if exists
                if parsed:
                    prob = parsed.probability
                    conf = parsed.confidence
                    reason = parsed.reasoning

                else:
                    # Parse raw content
                    text = message.content if hasattr(message, "content") else str(message)

                    # Attempt JSON decode
                    try:
                        payload = json.loads(text)
                        prob, conf, reason = LLMSignal._safe_parse_result(payload)
                    except Exception:
                        prob, conf, reason = LLMSignal._safe_parse_result({"text": text})

                # Clamp everything safely
                prob = clamp_prob(prob if prob is not None else current_prob)
                conf = clamp_conf(conf if conf is not None else 0.6)

                return SignalResult(
                    probability=prob,
                    confidence=conf,
                    reasoning=str(reason)[:1000],
                    metadata={"model": self.model, "attempts": attempts + 1}
                )

            except Exception as e:
                logger.warning(f"LLM analyze attempt {attempts+1} failed: {e}")
                attempts += 1
                time.sleep(backoff)
                backoff *= 2.0

        # --------------------------------------------------------
        # Hard fallback if all attempts fail
        # --------------------------------------------------------
        return SignalResult(
            probability=clamp_prob(current_prob),
            confidence=0.30,
            reasoning="LLM failed, fallback to market probability.",
            metadata={"model": self.model, "error": "llm_fail"}
        )
