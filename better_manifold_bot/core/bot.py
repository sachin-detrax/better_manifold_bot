import logging
import time
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from ..manifold.api import ManifoldAPI
from .decision import DecisionMaker, MarketDecision

logger = logging.getLogger(__name__)
console = Console()

class SmartBot:
    """
    The main bot class that orchestrates market fetching, analysis, and betting.
    """
    
    def __init__(
        self, 
        api: ManifoldAPI, 
        decision_maker: DecisionMaker,
        dry_run: bool = False,
        target_creator: str = "MikhailTal"
    ):
        self.api = api
        self.decision_maker = decision_maker
        self.dry_run = dry_run
        self.target_creator = target_creator
        self.decisions = []
        
    def run(self, limit: int = 20, interval: int = 3600):
        """
        Main loop to run the bot.
        
        Args:
            limit: Max markets to process per cycle.
            interval: Sleep time between cycles (seconds).
        """
        mode = "[yellow]DRY RUN[/yellow]" if self.dry_run else "[green]LIVE[/green]"
        console.print(Panel(f"[bold cyan]Starting SmartBot[/bold cyan] - Mode: {mode}", 
                          border_style="cyan"))
        
        while True:
            try:
                self.process_cycle(limit)
            except Exception as e:
                console.print(f"[bold red]Error in bot cycle:[/bold red] {e}")
                logger.error(f"Error in bot cycle: {e}", exc_info=True)
                
            console.print(f"\n[dim] Sleeping for {interval} seconds...[/dim]\n")
            time.sleep(interval)

    def process_cycle(self, limit: int):
        """Process a single cycle of market analysis and betting."""
        console.print("\n[bold blue] Starting market scan cycle...[/bold blue]")
        
        # 1. Fetch markets
        markets = self.api.get_all_markets_by_creator(self.target_creator, limit=limit)
        console.print(f"[green][/green] Found {len(markets)} markets by [bold]{self.target_creator}[/bold]\n")
        
        # Reset decisions for this cycle
        self.decisions = []
        
        # 2. Analyze and Bet
        for i, market in enumerate(markets, 1):
            if market.get('isResolved'):
                continue
                
            if market.get('closeTime') and market['closeTime'] < time.time() * 1000:
                continue
                
            console.print(f"[dim]({i}/{len(markets)})[/dim] ", end="")
            self.process_market(market)
        
        # 3. Show summary
        self.show_summary()

    def process_market(self, market):
        """Analyze a single market and place bet if needed."""
        market_id = market['id']
        question = market['question']
        current_prob = market.get('probability', 0)
        
        console.print(f"[cyan]Analyzing:[/cyan] {question[:80]}{'...' if len(question) > 80 else ''}")
        
        decision = self.decision_maker.analyze(market)
        
        # Store decision for summary
        self.decisions.append({
            'question': question,
            'decision': decision.decision,
            'confidence': decision.confidence,
            'current_prob': current_prob,
            'estimated_prob': decision.estimated_probability,
            'reasoning': decision.reasoning,
            'bet_amount': decision.bet_amount
        })
        
        # Color code the decision
        if decision.decision == "YES":
            decision_color = "green"
            symbol = ""
        elif decision.decision == "NO":
            decision_color = "red"
            symbol = ""
        else:
            decision_color = "yellow"
            symbol = ""
        
        console.print(f"  [{decision_color}]{symbol} {decision.decision}[/{decision_color}] "
                     f"(Confidence: {decision.confidence:.0%}, "
                     f"Est. Prob: {decision.estimated_probability:.0%})")
        
        if decision.decision in ["YES", "NO"]:
            if self.dry_run:
                console.print(f"  [dim] Would bet {decision.bet_amount} M$ on {decision.decision}[/dim]")
            else:
                self.place_bet(market_id, decision.decision, decision.bet_amount)
        
        console.print()

    def show_summary(self):
        """Display a summary table of all decisions."""
        if not self.decisions:
            return
        
        # Count decisions
        yes_count = sum(1 for d in self.decisions if d['decision'] == 'YES')
        no_count = sum(1 for d in self.decisions if d['decision'] == 'NO')
        skip_count = sum(1 for d in self.decisions if d['decision'] == 'SKIP')
        
        # Create summary table
        table = Table(title=" Decision Summary", box=box.ROUNDED, show_header=True, 
                     header_style="bold magenta")
        table.add_column("Market", style="cyan", no_wrap=False, max_width=50)
        table.add_column("Decision", justify="center", style="bold")
        table.add_column("Conf.", justify="right")
        table.add_column("Mkt %", justify="right")
        table.add_column("Est %", justify="right")
        table.add_column("Bet", justify="right")
        
        for d in self.decisions:
            # Color code decision
            if d['decision'] == "YES":
                decision_text = "[green] YES[/green]"
            elif d['decision'] == "NO":
                decision_text = "[red] NO[/red]"
            else:
                decision_text = "[yellow]  SKIP[/yellow]"
            
            # Truncate question
            question = d['question'][:47] + "..." if len(d['question']) > 50 else d['question']
            
            table.add_row(
                question,
                decision_text,
                f"{d['confidence']:.0%}",
                f"{d['current_prob']:.0%}",
                f"{d['estimated_prob']:.0%}",
                f"{d['bet_amount']} M$" if d['decision'] != 'SKIP' else "-"
            )
        
        console.print("\n")
        console.print(table)
        
        # Summary stats
        total_bets = yes_count + no_count
        total_amount = sum(d['bet_amount'] for d in self.decisions if d['decision'] != 'SKIP')
        
        summary_text = (
            f"[bold]Total:[/bold] {len(self.decisions)} markets | "
            f"[green]{yes_count} YES[/green] | "
            f"[red]{no_count} NO[/red] | "
            f"[yellow]{skip_count} SKIP[/yellow]"
        )
        
        if total_bets > 0:
            summary_text += f" | [bold cyan]Total Bet: {total_amount} M$[/bold cyan]"
        
        console.print(Panel(summary_text, border_style="blue"))

    def place_bet(self, market_id: str, outcome: str, amount: int):
        """Place a bet with error handling."""
        try:
            self.api.place_bet(market_id, outcome, amount)
            console.print(f"  [bold green] Bet placed:[/bold green] {amount} M$ on {outcome}")
        except Exception as e:
            console.print(f"  [bold red] Bet failed:[/bold red] {e}")
            logger.error(f"Failed to place bet: {e}")
