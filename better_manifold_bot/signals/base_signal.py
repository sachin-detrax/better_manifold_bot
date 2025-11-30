"""
Base signal class for the ensemble decision maker.

All signals inherit from this base class and implement the analyze method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

def clamp_prob(p: float) -> float:
    """Clamp probability into safe range [0.01, 0.99]."""
    if p is None:
        return 0.5
    try:
        p = float(p)
    except Exception:
        return 0.5
    return max(0.01, min(0.99, p))

def clamp_conf(c: float) -> float:
    """Clamp confidence into [0.0, 0.95] to avoid overconfidence."""
    if c is None:
        return 0.5
    try:
        c = float(c)
    except Exception:
        return 0.5
    return max(0.0, min(0.95, c))

    
class SignalResult(BaseModel):
    """Result from a signal analysis."""
    probability: float = Field(..., description="Estimated probability of YES outcome (0.0 to 1.0)")
    confidence: float = Field(..., description="Confidence in this estimate (0.0 to 1.0)")
    reasoning: str = Field(..., description="Explanation for this signal's assessment")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional signal-specific data")


class BaseSignal(ABC):
    """
    Abstract base class for all signals in the ensemble.
    
    Each signal analyzes a market and returns a probability estimate
    with confidence level and reasoning.
    """
    
    def __init__(self, name: str, weight: float = 1.0, enabled: bool = True):
        """
        Initialize the signal.
        
        Args:
            name: Signal identifier
            weight: Weight in ensemble (will be normalized)
            enabled: Whether this signal is active
        """
        self.name = name
        self.weight = weight
        self.enabled = enabled
    
    @abstractmethod
    def analyze(self, market: Dict[str, Any]) -> Optional[SignalResult]:
        """
        Analyze a market and return a signal result.
        
        Args:
            market: Market data from Manifold API
            
        Returns:
            SignalResult with probability estimate and confidence,
            or None if signal cannot analyze this market
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self.weight}, enabled={self.enabled})"
