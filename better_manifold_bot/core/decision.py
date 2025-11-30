import logging
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

logger = logging.getLogger(__name__)

class MarketDecision(BaseModel):
    """Structured output for market decision."""
    decision: str = Field(..., description="The decision: 'YES', 'NO', or 'SKIP'")
    confidence: float = Field(..., description="Confidence level between 0.0 and 1.0")
    reasoning: str = Field(..., description="Explanation for the decision")
    estimated_probability: float = Field(..., description="Your estimated true probability of the event (0.0 to 1.0)")
    bet_amount: int = Field(10, description="Suggested bet amount in M$")

class DecisionMaker:
    """Base class for decision makers."""
    def analyze(self, market: Dict[str, Any]) -> MarketDecision:
        raise NotImplementedError

class LLMDecisionMaker(DecisionMaker):
    """Uses OpenAI to analyze markets."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        
    def analyze(self, market: Dict[str, Any]) -> MarketDecision:
        """Analyze a market using LLM."""
        
        question = market.get('question')
        description = market.get('description') or "No description provided."
        current_prob = market.get('probability')
        
        if current_prob is None:
            return MarketDecision(
                decision="SKIP",
                confidence=0.0,
                reasoning="Market has no probability.",
                estimated_probability=0.5
            )
        
        prompt = f"""
You are an autonomous Manifold super-forecaster agent.
Your job is to exploit mispriced prediction markets with cold, probabilistic reasoning.
No vibes. No deference. Only edge.

For the market below:

Market: {question}
Description: {description}
Current Probability: {current_prob:.2%}

Your mandate

Infer the true probability using first-principles reasoning, base rates, relevant historical data, and causal logic.

Identify mispricing magnitude (your probability vs. market probability).

Take a position ONLY when you detect exploitable edge.

Avoid uncertainty traps — if info is insufficient to form a strong prior, SKIP.

Decision Rules

If your estimated probability is >=10 percentage points ABOVE the market: bet YES.

If it is >=10 percentage points BELOW the market: bet NO.

If mispricing is smaller or confidence is low: SKIP.

(You’re not here to be cute; you’re here to outperform the market.)
        """
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a super-forecaster analyzing prediction markets."},
                    {"role": "user", "content": prompt}
                ],
                response_format=MarketDecision
            )
            
            return completion.choices[0].message.parsed
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return MarketDecision(
                decision="SKIP",
                confidence=0.0,
                reasoning=f"Error during analysis: {str(e)}",
                estimated_probability=current_prob
            )

class RandomDecisionMaker(DecisionMaker):
    """Random decision maker for testing/fallback."""
    def analyze(self, market: Dict[str, Any]) -> MarketDecision:
        import random
        return MarketDecision(
            decision=random.choice(["YES", "NO", "SKIP"]),
            confidence=0.5,
            reasoning="Random choice",
            estimated_probability=0.5
        )
