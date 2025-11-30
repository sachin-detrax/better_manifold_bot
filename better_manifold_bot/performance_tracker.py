"""
Performance tracking system for Better Manifold Bot.

Tracks bets, P&L, ROI, and generates comprehensive reports.
"""

import json
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BetRecord:
    """Record of a single bet."""
    timestamp: str
    market_id: str
    question: str
    direction: str  # YES or NO
    bet_amount: float
    market_prob: float
    ensemble_prob: float
    confidence: float
    edge: float
    current_balance: float

    # Signal breakdown
    llm_prob: float
    llm_conf: float
    historical_prob: float
    historical_conf: float
    microstructure_prob: float
    microstructure_conf: float

    # Outcomes (filled later when resolved)
    resolved: bool = False
    resolution: Optional[str] = None  # YES, NO, CANCEL, etc.
    profit_loss: Optional[float] = None
    roi: Optional[float] = None
    resolution_date: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary of a trading session."""
    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float

    markets_analyzed: int
    bets_placed: int

    initial_balance: float
    final_balance: float
    gross_pnl: float

    total_bet_amount: float
    avg_bet_size: float
    max_bet_size: float
    min_bet_size: float

    avg_confidence: float
    avg_edge: float

    yes_bets: int
    no_bets: int

    # Distribution
    confidence_distribution: Dict[str, int]  # "50-60%": count, "60-70%": count, etc.
    edge_distribution: Dict[str, int]


class PerformanceTracker:
    """
    Comprehensive performance tracking system.

    Features:
    - Bet logging to CSV and JSON
    - Real-time P&L tracking
    - Session summaries
    - Historical analysis
    - ROI calculations
    """

    def __init__(self, data_dir: str = "performance_data"):
        """
        Initialize performance tracker.

        Args:
            data_dir: Directory to store performance data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = datetime.now()
        self.session_bets: List[BetRecord] = []

        # Files
        self.bets_csv = self.data_dir / "bets_history.csv"
        self.bets_json = self.data_dir / "bets_history.jsonl"
        self.sessions_file = self.data_dir / "sessions.jsonl"

        # Initialize CSV if needed
        self._init_csv()

        logger.info(f"Performance tracker initialized (session: {self.session_id})")

    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.bets_csv.exists():
            with open(self.bets_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'session_id', 'market_id', 'question', 'direction',
                    'bet_amount', 'market_prob', 'ensemble_prob', 'confidence', 'edge',
                    'current_balance', 'llm_prob', 'llm_conf', 'historical_prob',
                    'historical_conf', 'microstructure_prob', 'microstructure_conf',
                    'resolved', 'resolution', 'profit_loss', 'roi', 'resolution_date'
                ])

    def log_bet(
        self,
        market: Dict[str, Any],
        decision: Any,  # MarketDecision
        bet_amount: float,
        current_balance: float,
        signal_results: List[Any]  # List of SignalResults
    ) -> BetRecord:
        """
        Log a bet to the tracking system.

        Args:
            market: Market data
            decision: MarketDecision object
            bet_amount: Amount bet
            current_balance: Balance after bet
            signal_results: Results from individual signals

        Returns:
            BetRecord object
        """
        # Extract signal data
        signal_data = {}
        for signal, result in zip(['llm', 'historical', 'microstructure'], signal_results):
            signal_data[f'{signal}_prob'] = result.probability if result else 0.0
            signal_data[f'{signal}_conf'] = result.confidence if result else 0.0

        # Create bet record
        bet = BetRecord(
            timestamp=datetime.now().isoformat(),
            market_id=market.get('id', ''),
            question=market.get('question', '')[:200],  # Truncate long questions
            direction=decision.decision,
            bet_amount=bet_amount,
            market_prob=decision.current_probability,
            ensemble_prob=decision.metadata.get('ensemble_probability', decision.current_probability),
            confidence=decision.confidence,
            edge=decision.metadata.get('edge', 0.0),
            current_balance=current_balance,
            **signal_data
        )

        # Add to session
        self.session_bets.append(bet)

        # Write to CSV
        self._write_csv(bet)

        # Write to JSONL
        self._write_jsonl(bet)

        logger.debug(f"Logged bet: {bet.direction} ${bet.bet_amount:.2f} on {bet.question[:50]}...")

        return bet

    def _write_csv(self, bet: BetRecord):
        """Append bet to CSV file."""
        with open(self.bets_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [
                bet.timestamp, self.session_id, bet.market_id, bet.question, bet.direction,
                bet.bet_amount, bet.market_prob, bet.ensemble_prob, bet.confidence, bet.edge,
                bet.current_balance, bet.llm_prob, bet.llm_conf, bet.historical_prob,
                bet.historical_conf, bet.microstructure_prob, bet.microstructure_conf,
                bet.resolved, bet.resolution, bet.profit_loss, bet.roi, bet.resolution_date
            ]
            writer.writerow(row)

    def _write_jsonl(self, bet: BetRecord):
        """Append bet to JSONL file."""
        with open(self.bets_json, 'a', encoding='utf-8') as f:
            record = asdict(bet)
            record['session_id'] = self.session_id
            f.write(json.dumps(record) + '\n')

    def end_session(
        self,
        markets_analyzed: int,
        initial_balance: float,
        final_balance: float
    ) -> SessionSummary:
        """
        End the current session and generate summary.

        Args:
            markets_analyzed: Number of markets analyzed
            initial_balance: Starting balance
            final_balance: Ending balance

        Returns:
            SessionSummary object
        """
        session_end = datetime.now()
        duration = (session_end - self.session_start).total_seconds()

        # Calculate statistics
        bet_amounts = [b.bet_amount for b in self.session_bets]
        confidences = [b.confidence for b in self.session_bets]
        edges = [b.edge for b in self.session_bets]

        # Direction counts
        yes_bets = sum(1 for b in self.session_bets if b.direction == "YES")
        no_bets = sum(1 for b in self.session_bets if b.direction == "NO")

        # Distributions
        conf_dist = self._calculate_distribution(confidences, 0.1)  # 10% bins
        edge_dist = self._calculate_distribution(edges, 0.05)  # 5% bins

        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start.isoformat(),
            end_time=session_end.isoformat(),
            duration_seconds=duration,
            markets_analyzed=markets_analyzed,
            bets_placed=len(self.session_bets),
            initial_balance=initial_balance,
            final_balance=final_balance,
            gross_pnl=final_balance - initial_balance,
            total_bet_amount=sum(bet_amounts) if bet_amounts else 0.0,
            avg_bet_size=statistics.mean(bet_amounts) if bet_amounts else 0.0,
            max_bet_size=max(bet_amounts) if bet_amounts else 0.0,
            min_bet_size=min(bet_amounts) if bet_amounts else 0.0,
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            avg_edge=statistics.mean(edges) if edges else 0.0,
            yes_bets=yes_bets,
            no_bets=no_bets,
            confidence_distribution=conf_dist,
            edge_distribution=edge_dist
        )

        # Save session summary
        with open(self.sessions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(summary)) + '\n')

        logger.info(f"Session {self.session_id} ended: {len(self.session_bets)} bets, P&L: ${summary.gross_pnl:+.2f}")

        return summary

    def _calculate_distribution(self, values: List[float], bin_size: float) -> Dict[str, int]:
        """Calculate distribution of values into bins."""
        distribution = defaultdict(int)

        for value in values:
            bin_start = int(value / bin_size) * bin_size
            bin_end = bin_start + bin_size
            bin_label = f"{bin_start:.0%}-{bin_end:.0%}"
            distribution[bin_label] += 1

        return dict(distribution)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        if not self.session_bets:
            return {
                'bets_placed': 0,
                'total_bet': 0.0,
                'avg_bet': 0.0,
                'avg_confidence': 0.0,
                'avg_edge': 0.0
            }

        bet_amounts = [b.bet_amount for b in self.session_bets]
        confidences = [b.confidence for b in self.session_bets]
        edges = [b.edge for b in self.session_bets]

        return {
            'bets_placed': len(self.session_bets),
            'total_bet': sum(bet_amounts),
            'avg_bet': statistics.mean(bet_amounts),
            'min_bet': min(bet_amounts),
            'max_bet': max(bet_amounts),
            'avg_confidence': statistics.mean(confidences),
            'avg_edge': statistics.mean(edges),
            'yes_bets': sum(1 for b in self.session_bets if b.direction == "YES"),
            'no_bets': sum(1 for b in self.session_bets if b.direction == "NO"),
        }

    def load_all_bets(self) -> List[Dict[str, Any]]:
        """Load all historical bets from JSONL file."""
        if not self.bets_json.exists():
            return []

        bets = []
        with open(self.bets_json, 'r', encoding='utf-8') as f:
            for line in f:
                bets.append(json.loads(line))

        return bets

    def load_all_sessions(self) -> List[Dict[str, Any]]:
        """Load all session summaries."""
        if not self.sessions_file.exists():
            return []

        sessions = []
        with open(self.sessions_file, 'r', encoding='utf-8') as f:
            for line in f:
                sessions.append(json.loads(line))

        return sessions

    def get_historical_performance(self, days: int = 30) -> Dict[str, Any]:
        """
        Get historical performance metrics.

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with performance metrics
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        sessions = self.load_all_sessions()
        recent_sessions = [
            s for s in sessions
            if datetime.fromisoformat(s['start_time']) >= cutoff_date
        ]

        if not recent_sessions:
            return {
                'total_sessions': 0,
                'total_bets': 0,
                'total_pnl': 0.0,
                'avg_session_pnl': 0.0,
                'win_rate': 0.0
            }

        total_bets = sum(s['bets_placed'] for s in recent_sessions)
        total_pnl = sum(s['gross_pnl'] for s in recent_sessions)

        return {
            'total_sessions': len(recent_sessions),
            'total_bets': total_bets,
            'total_pnl': total_pnl,
            'avg_session_pnl': total_pnl / len(recent_sessions),
            'avg_bets_per_session': total_bets / len(recent_sessions),
            'total_bet_amount': sum(s['total_bet_amount'] for s in recent_sessions),
        }
