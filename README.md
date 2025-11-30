# Better Manifold Bot

A sophisticated trading bot for [Manifold Markets](https://manifold.markets/), designed to make intelligent betting decisions using a multi-signal ensemble approach, including OpenAI's GPT-4o-mini for structured forecasting.

## Features

- **Ensemble Decision Making**: Combines multiple signals to make robust betting decisions.
  - **Historical Signal**: Analyzes past market trends and user accuracy.
  - **Microstructure Signal**: Looks at order book depth and recent activity.
  - **OpenAI Signal**: Uses GPT-4o-mini with a structured forecasting framework (decomposition, base rates, etc.) and self-consistency checks.
- **Kelly Criterion Betting**: Optimizes bet sizing based on calculated edge and confidence to maximize long-term growth while minimizing ruin.
- **Performance Tracking**: Detailed tracking of bets, P&L, and ROI.
- **Visualizations**: Generates performance graphs (cumulative P&L, win rate, etc.).
- **Dry Run Mode**: Test strategies without risking real mana.
- **Targeted Trading**: Configurable to trade on markets by specific creators (default: `MikhailTal`).

## System Architecture & Decision Logic

The core logic is orchestrated by `main_openai.py`, which implements a sophisticated pipeline for market analysis and execution.

### 1. Signal Generation
The bot aggregates insights from three distinct signal sources:

*   **Historical Signal (Weight: 25%)**:
    *   Analyzes the creator's past market resolution history.
    *   Calculates the "Real World Resolution Rate" vs. "Market Probability" to identify systematic biases (e.g., markets consistently resolving lower than predicted).
*   **Microstructure Signal (Weight: 15%)**:
    *   Examines the order book for liquidity imbalances.
    *   Detects recent buying/selling pressure that might indicate insider knowledge or breaking news.
*   **OpenAI Signal (Weight: 60%)**:
    *   **Structured Forecasting**: Uses a specialized prompt that forces the LLM to decompose the question into sub-factors, estimate base rates, and consider counter-arguments.
    *   **Self-Consistency**: Runs the analysis multiple times (default: 3) in parallel.
    *   **Variance Penalty**: Penalizes the confidence score if the multiple runs disagree significantly, reducing the risk of hallucinations.

### 2. Ensemble Aggregation
The `EnhancedEnsembleDecisionMaker` combines these weighted signals into a single "True Probability" estimate.
*   It applies a **Disagreement Penalty**: If the signals contradict each other (e.g., Historical says BUY but OpenAI says SELL), the final confidence is reduced.
*   It calculates the **Edge**: `Edge = |True Probability - Market Probability|`.

### 3. Bet Sizing (Kelly Criterion)
If the Edge exceeds the minimum threshold (5%), the bot calculates the optimal bet size using the **Kelly Criterion**:
*   `f* = (bp - q) / b`
    *   `f*`: Fraction of bankroll to bet.
    *   `b`: Net odds received on the wager.
    *   `p`: Probability of winning (our "True Probability").
    *   `q`: Probability of losing (1 - p).
*   **Safety Limits**: The bot uses "Quarter Kelly" (0.25x) to reduce volatility and caps the maximum bet at 15% of the bankroll.

### 4. Execution
*   The bot places the bet via the Manifold API.
*   It logs the detailed rationale, including the contribution of each signal, to `performance_data/` for review.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd better_manifold_bot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    MANIFOLD_API_KEY=your_manifold_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### Basic Usage

Run the bot in dry-run mode (safe for testing):

```bash
python main_openai.py --dry-run --limit 10
```

### Live Trading

To enable real betting (ensure your API keys are set):

```bash
python main_openai.py --limit 20
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dry-run` | Run in simulation mode (no real bets) | `False` |
| `--limit N` | Number of markets to analyze | `10` |
| `--bet-amount N` | Fixed bet amount (overrides Kelly sizing) | `None` (Kelly) |
| `--show-report` | Show historical performance report | `False` |
| `--generate-graphs` | Generate performance visualization graphs | `False` |
| `--n-runs N` | Number of OpenAI runs for self-consistency | `3` |
| `--disable-openai` | Disable OpenAI signal (for testing/cost saving) | `False` |

### Examples

**Run with custom OpenAI settings:**

```bash
python main_openai.py --dry-run --n-runs 5 --variance-penalty 0.8
```

**Generate performance report and graphs:**

```bash
python main_openai.py --show-report --generate-graphs
```

## Project Structure

- `main_openai.py`: Main entry point for the OpenAI-enhanced bot.
- `better_manifold_bot/`: Core package.
  - `core/`: Core bot logic and decision making interfaces.
  - `manifold/`: Manifold API client.
  - `signals/`: Signal implementations (OpenAI, Historical, Microstructure).
  - `ensemble_decision_maker.py`: Logic for combining signals.
  - `kelly_bot.py`: Bot implementation with Kelly betting.
- `performance_data/`: Stores bet history and performance logs.

## License

[MIT License](LICENSE)
