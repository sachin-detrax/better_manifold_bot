# Better Manifold Bot - Complete System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Decision-Making Process](#decision-making-process)
4. [Signal Analysis](#signal-analysis)
5. [Bet Sizing Strategy](#bet-sizing-strategy)
6. [Complete Workflow](#complete-workflow)
7. [Configuration](#configuration)
8. [Mathematical Foundations](#mathematical-foundations)

---

## System Overview

The Better Manifold Bot is an **ensemble-based prediction market trading system** that combines multiple analytical signals to make informed betting decisions on Manifold Markets. It uses a sophisticated multi-signal approach, optimal bet sizing via the Kelly Criterion, and statistical learning from historical data.

### Key Features
- **Multi-signal ensemble**: Combines 3 independent signals (LLM, Historical, Microstructure)
- **Kelly Criterion bet sizing**: Optimizes position sizes based on edge and bankroll
- **Historical learning**: Learns from past market resolutions to identify patterns
- **Risk management**: Built-in safeguards prevent over-betting
- **Adaptive confidence**: Adjusts confidence based on data quality and market conditions

### Core Philosophy
The bot doesn't try to predict the future - it identifies **mispriced markets** where the ensemble's probability estimate differs significantly from the market price, creating a positive expected value (EV) betting opportunity.

---

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Entry Point                         │
│                    (main_openai.py)                          │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├──> KellyBot (kelly_bot.py)
               │    ├─> Manages bet execution
               │    └─> Calculates Kelly-optimal bet sizes
               │
               ├──> EnhancedEnsembleDecisionMaker
               │    ├─> Aggregates signals
               │    ├─> Makes YES/NO/SKIP decisions
               │    └─> Enforces min_confidence and min_edge
               │
               └──> Signals (signals/)
                    ├─> OpenAISignal (Structured Forecasting)
                    ├─> HistoricalSignal (Pattern Learning)
                    └─> MicrostructureSignal (Liquidity Analysis)
```

### Data Flow

```
Markets from Manifold API
         │
         ├──> Signal 1: LLM Analysis
         │    └──> probability, confidence, reasoning
         │
         ├──> Signal 2: Historical Patterns
         │    └──> probability, confidence, reasoning
         │
         └──> Signal 3: Microstructure
              └──> probability, confidence, reasoning
                         │
                         ▼
              Ensemble Decision Maker
              (Weighted Bayesian aggregation)
                         │
                         ├──> Ensemble Probability (weighted avg)
                         ├──> Ensemble Confidence (weighted avg)
                         └──> Edge = |Ensemble - Market|
                                      │
                                      ▼
                         Decision Logic:
                         - SKIP if confidence < 55% or edge < 3%
                         - YES if ensemble > market + edge
                         - NO if ensemble < market - edge
                                      │
                                      ▼
                         Kelly Criterion Calculator
                         - Calculates optimal bet size
                         - Applies safety caps (max 15% bankroll)
                         - Returns $ amount to bet
                                      │
                                      ▼
                         Place Bet on Manifold Markets
```

---

## Decision-Making Process

### Step 1: Signal Collection

Each signal independently analyzes the market and returns:
- **Probability**: Estimated probability of YES outcome (0.0 to 1.0)
- **Confidence**: How confident the signal is (0.0 to 1.0)
- **Reasoning**: Explanation for the estimate
- **Metadata**: Additional signal-specific data

### Step 2: Ensemble Aggregation

The ensemble combines signals using **weighted averaging**:

```python
# Weighted average probability
ensemble_prob = Σ(signal_prob × signal_weight) / Σ(signal_weight)

# Current weights:
# - LLM: 60%
# - Historical: 25%
# - Microstructure: 15%

# Weighted average confidence
ensemble_confidence = Σ(signal_confidence × signal_weight) / Σ(signal_weight)

# Calculate edge
edge = |ensemble_prob - market_prob|
```

### Step 3: Decision Thresholds

The bot only bets when BOTH conditions are met:

1. **Minimum Confidence**: `ensemble_confidence >= 0.55` (55%)
   - Ensures we're reasonably sure about our estimate

2. **Minimum Edge**: `edge >= 0.03` (3%)
   - Ensures sufficient profit opportunity to overcome slippage

### Step 4: Direction Decision

```python
if ensemble_confidence < min_confidence or edge < min_edge:
    decision = "SKIP"

elif ensemble_prob > market_prob + min_edge:
    decision = "YES"  # Market underpriced

elif ensemble_prob < market_prob - min_edge:
    decision = "NO"   # Market overpriced

else:
    decision = "SKIP"  # Market fairly priced
```

---

## Signal Analysis

### 1. OpenAISignal (Weight: 60%)

**Purpose**: Leverage GPT-4o-mini with a structured forecasting framework to analyze market questions.

**Key Features**:
- **Structured Forecasting**: Forces the model to identify base rates, factors, and confounders before estimating probability.
- **Self-Consistency Ensemble**: Runs the analysis multiple times (`n_runs=3`) and aggregates results to reduce variance.
- **Variance Penalty**: Penalizes the probability estimate if the multiple runs disagree significantly.
- **Disagreement Penalty**: Reduces confidence if the LLM signal strongly disagrees with the consensus of other signals.

**Process**:
1. **Prompting**: Sends market question, description, and current probability to GPT-4o-mini with a strict JSON schema.
2. **Decomposition**: The model must provide a mathematical decomposition (Base Rate + Factors = Probability).
3. **Validation**: The system validates that the decomposition is mathematically sound and matches the declared probability.
4. **Aggregation**: Combines multiple runs, applying penalties for high variance.

**Confidence**: Derived from the model's self-assessed margin of error, adjusted by variance and disagreement penalties.

**Strengths**:
- **Reasoning**: Can understand complex questions and context.
- **Robustness**: Self-consistency reduces hallucination and random errors.
- **Calibration**: Structured approach improves probability calibration.

**Weaknesses**:
- **Cost/Latency**: Multiple runs increase API costs and time per market.
- **Knowledge Cutoff**: Limited by the model's training data.

**Example Output**:
```python
SignalResult(
    probability=0.65,
    confidence=0.72,
    reasoning="OpenAI ensemble (3 runs):\nBase rate: 60.0%\nMean prob: 68.3% ± 4.2%\nFinal prob: 65.0% (after variance penalty)...",
    metadata={
        "n_runs": 3,
        "variance_penalty": 0.033,
        "disagreement_penalty": 0.0
    }
)
```

---

### 2. Historical Signal (Weight: 25%)

**Purpose**: Learn patterns from resolved markets to identify creator biases and market tendencies

**Initialization Process**:
1. Fetches ~2000 markets from target creator (MikhailTal)
2. Filters for resolved binary markets (YES/NO outcomes only)
3. Calculates creator statistics:
   - YES resolution rate
   - Sample size
   - Recent outcomes (last 20 markets)

**Analysis Process**:
```python
# Creator bias adjustment
if creator in creator_stats:
    creator_yes_rate = stats["yes_rate"]
    sample_size = stats["count"]

    # Weight historical data based on sample size
    history_weight = 0.3 if sample_size >= 20 else 0.15

    # Blend market price with historical YES rate
    adjusted_prob = (1 - history_weight) * market_prob + history_weight * creator_yes_rate
```

**Confidence Calculation**:
```python
confidence = min(0.75, 0.50 + (sample_size / 100) * 0.25)
# More resolved markets = higher confidence in pattern
```

**Keyword Analysis**:
- Adds +2% for positive keywords: "will", "likely", "expected", "predicted"
- Subtracts -2% for negative keywords: "won't", "unlikely", "impossible", "never"

**Example**: If MikhailTal historically resolves 58% of markets to YES, this signal adjusts the probability toward 58%, weighted by sample size.

**Confidence**: 50-75% (increases with sample size)

**Strengths**:
- Identifies systematic creator biases
- Based on actual historical data
- Improves over time with more data

**Weaknesses**:
- Assumes past patterns continue
- Limited by sample size
- Can be slow to adapt to changes

---

### 3. Microstructure Signal (Weight: 15%)

**Purpose**: Assess market quality and information efficiency based on liquidity and trading volume

**Analysis Logic**:
```python
liquidity = market.get("totalLiquidity", 0)
volume = market.get("volume", 0)

# Confidence based on liquidity
if liquidity < 100:
    confidence = 0.40  # Low liquidity = less reliable price
elif liquidity < 500:
    confidence = 0.60
else:
    confidence = 0.70  # High liquidity = more reliable

# Boost for high volume (indicates informed trading)
if volume > 1000:
    confidence *= 1.1

# This signal trusts the market price itself
probability = current_market_prob
```

**Confidence**: 40-77% (based on liquidity and volume)

**Strengths**:
- Identifies thin/illiquid markets
- Prevents betting on low-quality prices
- Fast and computation-free

**Weaknesses**:
- Doesn't predict direction
- Only moderates confidence
- Can't detect manipulation

**Interpretation**: This signal acts as a "trust factor" for the market price. High liquidity means the price likely reflects available information.

---

## Bet Sizing Strategy

### Kelly Criterion Fundamentals

The bot uses the **Kelly Criterion**, a mathematically optimal bet sizing formula that maximizes long-term logarithmic growth of capital.

**Formula**:
```
f* = (p × (b + 1) - 1) / b

Where:
f* = fraction of bankroll to bet
p = our probability of winning
b = net odds received (payout/stake - 1)
```

**For Manifold's LMSR Markets**:
```python
# Betting YES at market price m:
odds = (1 - m) / m

# Betting NO at market price m:
odds = m / (1 - m)
```

### Fractional Kelly (Safety First)

The bot uses **Quarter Kelly** (25% of full Kelly):
```python
kelly_fraction = 0.25
bet_size = kelly_pct * kelly_fraction * bankroll
```

**Why Fractional Kelly?**
- Full Kelly can be aggressive and volatile
- Quarter Kelly provides ~99% of growth with ~50% less volatility
- More forgiving of probability estimation errors

### Safety Caps

Multiple safeguards prevent over-betting:

```python
# 1. Maximum bet fraction: 15% of bankroll
max_bet_fraction = 0.15
kelly_pct = min(kelly_pct, max_bet_fraction)

# 2. Minimum bet: $2.00
min_bet = 2.0

# 3. Maximum bet: $50.00
max_bet = 50.0

# 4. Final bet size
bet_size = max(min(kelly_pct * bankroll, max_bet), min_bet)
```

### Example Calculation

**Scenario**:
- Bankroll: $1000
- Ensemble probability: 60%
- Market probability: 50%
- Direction: YES
- Kelly fraction: 0.25

**Steps**:
```python
# 1. Calculate edge
edge = 0.60 - 0.50 = 0.10 (10% edge)

# 2. Calculate odds
odds = (1 - 0.50) / 0.50 = 1.0

# 3. Calculate full Kelly
kelly_pct = ((1.0 + 1) * 0.60 - 1) / 1.0 = 0.20 (20% of bankroll)

# 4. Apply fractional Kelly
kelly_pct = 0.20 * 0.25 = 0.05 (5% of bankroll)

# 5. Calculate bet size
bet_size = 0.05 * $1000 = $50.00

# 6. Apply max bet cap
final_bet = min($50.00, $50.00) = $50.00
```

**Result**: Bet $50 on YES

---

## Complete Workflow

### Initialization Phase
```
1. Load environment variables (.env)
   - MANIFOLD_API_KEY
   - OPENAI_API_KEY

2. Initialize signals:
   - HistoricalSignal: Fetch ~2000 markets, calculate creator stats
   - MicrostructureSignal: Ready
   - OpenAISignal: Initialize client, set n_runs=3

3. Create EnhancedEnsembleDecisionMaker:
   - Set min_confidence = 0.60
   - Set min_edge = 0.05
   - Load signals with weights (OpenAI: 0.60, Historical: 0.25, Microstructure: 0.15)

4. Create KellyBot:
   - Set kelly_fraction = 0.25
   - Set max_bet_fraction = 0.15
   - Set min_bet = $2.00
   - Set max_bet = $50.00
   - Initialize PerformanceTracker and Dashboard
```

### Market Analysis Phase
```
1. Fetch market data from Manifold API
2. Run each signal's analyze() method:

   a) OpenAISignal:
      - Create prompt with market context
      - Run n_runs (3) times for self-consistency
      - Validate JSON structure and mathematical decomposition
      - Aggregate results and apply variance penalty
      - Check for disagreement with other signals
      - Return SignalResult(prob, conf, reasoning)

   b) HistoricalSignal:
      - Look up creator stats, apply bias adjustment
      - Return SignalResult(prob, conf, reasoning)

   c) MicrostructureSignal:
      - Check liquidity/volume, adjust confidence
      - Return SignalResult(prob, conf, reasoning)

3. Ensemble aggregation:
   - Calculate weighted average probability and confidence
   - Calculate edge = |ensemble_prob - market_prob|

4. Make decision:
   - Check min_confidence (60%) and min_edge (5%)
   - Determine direction (YES/NO/SKIP)
```

### Betting Phase

If decision is YES or NO:

```
1. Calculate Kelly bet size:
   - Get current bankroll from Manifold API
   - Calculate edge: ensemble_prob - market_prob
   - Calculate odds based on market price
   - Compute full Kelly percentage
   - Apply kelly_fraction (0.25)
   - Apply safety caps (min/max bet, max fraction)

2. Execute bet:
   - If bet_size < min_bet ($2): SKIP
   - Otherwise: Place bet on Manifold Markets
   - Log bet details and reasoning

3. Update state:
   - Bankroll decreases by bet amount
   - Position recorded (for future P&L tracking)
   - Rate limit delay (1 second between bets)
```

### Results Tracking

```
After all markets analyzed:
- Markets analyzed: 20
- Bets placed: 13
- Initial balance: $1000.00
- Final balance: $843.11
- P&L: -$156.89 (due to slippage)
```

---

## Configuration

### Current Settings (main_openai.py)

```python
# Signal Weights
signals = [
    OpenAISignal(weight=0.60),        # 60% - Primary signal (Structured Forecasting)
    HistoricalSignal(weight=0.25),    # 25% - Secondary signal
    MicrostructureSignal(weight=0.15) # 15% - Quality check
]

# Decision Thresholds
min_confidence = 0.60  # 60% minimum confidence
min_edge = 0.05        # 5% minimum edge

# OpenAI Parameters
n_runs = 3             # Self-consistency runs
variance_penalty = 0.5 # Penalty for inconsistent predictions
disagreement_low = 0.03
disagreement_high = 0.10

# Kelly Parameters
kelly_fraction = 0.25      # Quarter Kelly
max_bet_fraction = 0.15    # Max 15% of bankroll per bet
min_bet = 2.0              # Minimum $2 bet
max_bet = 50.0             # Maximum $50 bet

# Execution Limits
max_bets = 20              # Maximum bets per run
```

### Tuning Guidelines

**To increase trading frequency**:
- Lower `min_confidence` (e.g., 0.50)
- Lower `min_edge` (e.g., 0.02)
- Increase `max_bets`

**To increase bet sizes**:
- Increase `kelly_fraction` (e.g., 0.50 for Half Kelly)
- Increase `max_bet_fraction` (e.g., 0.20)
- Increase `max_bet`

**To be more conservative**:
- Increase `min_confidence` (e.g., 0.65)
- Increase `min_edge` (e.g., 0.05)
- Decrease `kelly_fraction` (e.g., 0.10 for Tenth Kelly)

**To change signal influence**:
- Adjust signal weights (must sum to 1.0 after normalization)
- Example: Increase LLM to 0.70, decrease Historical to 0.15

---

## Mathematical Foundations

### Bayesian Ensemble Aggregation

The ensemble uses **weighted averaging** rather than true Bayesian updating for computational efficiency:

```python
P_ensemble = Σ(w_i × P_i) / Σ(w_i)

Where:
w_i = weight of signal i
P_i = probability estimate from signal i
```

**Alternative approaches considered**:
- **Bayesian Model Averaging**: More sophisticated but computationally expensive
- **Logistic Regression**: Requires extensive training data
- **Simple Averaging**: Ignores signal quality differences

### Kelly Criterion Derivation

The Kelly Criterion maximizes the **expected logarithmic growth rate**:

```
E[log(wealth)] = p × log(1 + f × b) + (1 - p) × log(1 - f)

Taking derivative and setting to zero:
f* = (p × b - q) / b = (p × (b + 1) - 1) / b

Where:
p = probability of win
q = probability of loss = 1 - p
b = net odds
f* = optimal fraction to bet
```

**Properties**:
- Never bets entire bankroll (unless p = 1)
- Bets zero if no edge (p × (b + 1) = 1)
- Maximizes geometric mean return
- Minimizes time to reach wealth goal

### Edge and Expected Value

**Edge**: The probability difference between your estimate and the market

```
Edge = |P_ensemble - P_market|
```

**Expected Value (EV)**: Expected profit per dollar bet

```python
# Betting YES at price p_market with true probability p_true:
EV = p_true × (1 - p_market) - (1 - p_true) × p_market
EV = p_true - p_market

# For EV > 0, need: p_true > p_market (market underpriced)
```

**Example**:
- Ensemble: 60% YES
- Market: 50% YES
- Edge: 10%
- EV: +0.10 (10% expected profit per dollar)

### Confidence Intervals

Each signal provides a confidence score representing certainty in the estimate:

- **High confidence (70-80%)**: Strong evidence, large sample, clear reasoning
- **Medium confidence (55-70%)**: Moderate evidence, some uncertainty
- **Low confidence (40-55%)**: Weak evidence, high uncertainty

The ensemble confidence is the weighted average:

```
Confidence_ensemble = Σ(w_i × Confidence_i) / Σ(w_i)
```

---

## Risk Management

### Position Sizing Limits

1. **Fractional Kelly**: Uses 25% of full Kelly to reduce variance
2. **Max Bet Fraction**: Never risk > 15% of bankroll on single bet
3. **Max Absolute Bet**: Never bet > $50 regardless of bankroll
4. **Min Absolute Bet**: Only bet if size ≥ $2 (avoid tiny positions)

### Diversification

- **Max bets per run**: 20 bets
- **Rate limiting**: 1 second delay between bets
- **Multiple markets**: Spreads risk across different questions

### Testing & Safety

- **Dry Run Mode**: Run with `--dry-run` to simulate betting.
  - Calculates signals and decisions exactly as in live mode.
  - Logs "Would place bet" messages.
  - Tracks simulated performance in the dashboard.
  - **Crucial**: Does NOT send any bet requests to the Manifold API.

### Quality Filters

- **Min confidence**: 60% (skip low-confidence opportunities)
- **Min edge**: 5% (skip marginally mispriced markets)
- **Binary markets only**: No multi-choice or free-response markets
- **Liquidity check**: Microstructure signal flags thin markets

---

## Performance Considerations

### Expected Slippage

In Manifold's LMSR markets, placing a bet moves the price against you:

- **Typical slippage**: 60-70% of bet amount as immediate unrealized loss
- **Why it happens**: AMM adjusts price based on your order
- **Example**: Bet $10 YES at 50%, price moves to 54%, your shares worth ~$5.40

**Implication**: Need markets to resolve in your favor just to break even. This is why positive EV and large edges are critical.

### Historical Performance Metrics

Key metrics to track:

1. **Win Rate**: Percentage of bets that profit
2. **ROI**: Return on investment per bet
3. **Sharpe Ratio**: Risk-adjusted returns
4. **Kelly Growth Rate**: Actual vs theoretical growth
5. **Calibration**: Are 60% confidence bets correct 60% of the time?

---

## Performance Tracking
 
The system includes a comprehensive dashboard for monitoring performance in real-time and analyzing historical results.
 
### Dashboard Features
 
- **Session Summary**: Real-time stats on bets placed, P&L, and win rates.
- **Live Bets Table**: Detailed view of recent bets with signal breakdown.
- **P&L Chart**: ASCII-based visualization of profit/loss trends.
- **ROI Metrics**: Return on Investment, average P&L per bet, and winning session rates.
 
### Reports
 
- **Live View**: Shows active session stats during execution.
- **End-of-Run Report**: Detailed summary after completion.
- **Historical Report**: Run with `--show-report` to see long-term performance (7-day, 30-day).
- **Graphs**: Run with `--generate-graphs` to create visual plots (saved in `performance_data/`).
 
---
 
## Future Improvements

### Short-term Enhancements
1. **Calibration module**: Track signal accuracy and adjust weights
2. **Market type detection**: Different strategies for sports, politics, etc.
3. **Time decay**: Adjust probabilities as resolution date approaches
4. **Correlation detection**: Avoid betting on correlated markets

### Medium-term Features
1. **Limit orders**: Reduce slippage by using limit orders
2. **Portfolio optimization**: Consider existing positions when betting
3. **Dynamic thresholds**: Adjust min_confidence based on recent performance
4. **Multi-market signals**: Analyze related markets for information

### Long-term Vision
1. **Reinforcement learning**: Let bot learn optimal strategies
2. **Causal inference**: Understand cause-effect in market movements
3. **Real-time data integration**: Fetch news and data feeds
4. **Social signals**: Analyze trader behavior and sentiment

---

## Debugging and Monitoring

### Log Levels

The system logs at multiple levels:

- **INFO**: Key decisions and bet placements
- **DEBUG**: Detailed signal analysis and calculations
- **WARNING**: Potential issues (low liquidity, API errors)
- **ERROR**: Failures that don't stop execution

### Key Metrics to Monitor

1. **Signal agreement**: How often do signals agree?
2. **Edge distribution**: Are we finding enough mispriced markets?
3. **Bet size distribution**: Is Kelly sizing working correctly?
4. **Confidence calibration**: Are we overconfident or underconfident?
5. **P&L vs expected EV**: Are we achieving theoretical returns?

### Common Issues

**Problem**: No bets placed
- **Check**: min_confidence and min_edge thresholds too high
- **Check**: All signals returning similar probabilities to market
- **Check**: Historical signal not loaded (API key missing)

**Problem**: Large negative P&L
- **Check**: Slippage is expected short-term
- **Check**: Bets need time to resolve
- **Check**: Signals may be miscalibrated

**Problem**: Too many bets
- **Check**: Thresholds too low
- **Check**: May be finding false edges
- **Check**: Risk of over-diversification

---

## Conclusion

The Better Manifold Bot is a sophisticated ensemble system that:

1. **Analyzes** markets using multiple independent signals
2. **Aggregates** signal estimates using weighted averaging
3. **Decides** YES/NO/SKIP based on confidence and edge thresholds
4. **Sizes** bets optimally using the Kelly Criterion
5. **Manages** risk through multiple safety mechanisms

The system is designed for **long-term positive expected value (EV)**, not short-term profits. Success requires patience, calibration, and continuous monitoring.

**Key Success Factors**:
- Quality signals that identify true mispricing
- Proper calibration of probability estimates
- Disciplined adherence to Kelly sizing
- Sufficient capital to withstand variance
- Continuous learning and adaptation

---

## Appendix: File Structure

```
better_manifold_bot/
├── main_openai.py              # Primary entry point (OpenAI-enhanced)
├── main_ensemble.py            # Legacy entry point
├── better_manifold_bot/
│   ├── dashboard.py            # Rich terminal dashboard
│   ├── performance_tracker.py  # Tracks bets and P&L
│   ├── visualizations.py       # Generates performance graphs
│   ├── ensemble_decision_maker.py
│   ├── kelly_bot.py
│   ├── kelly_criterion.py
│   ├── signals/
│   │   ├── base_signal.py
│   │   ├── openai_signal.py    # Structured forecasting signal
│   │   ├── historical_signal.py
│   │   └── microstructure_signal.py
│   ├── manifold/
│   │   └── api.py
│   └── utils/
├── performance_data/           # Stored performance logs and graphs
├── .env
└── requirements.txt
```

---

**Last Updated**: 2025-11-30
**Version**: 1.0
**Author**: Better Manifold Bot Team
