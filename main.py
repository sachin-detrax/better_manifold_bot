import argparse
import logging
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from better_manifold_bot.manifold.api import ManifoldAPI
from better_manifold_bot.core.decision import LLMDecisionMaker, RandomDecisionMaker
from better_manifold_bot.core.bot import SmartBot

console = Console()

# Configure logging - suppress verbose logs
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Better Manifold Bot")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (no bets placed)")
    parser.add_argument("--limit", type=int, default=10, help="Max markets to process")
    parser.add_argument("--interval", type=int, default=3600, help="Sleep interval in seconds")
    parser.add_argument("--creator", type=str, default="MikhailTal", help="Target creator username")
    parser.add_argument("--use-random", action="store_true", help="Use random decision maker (for testing without LLM)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)
    
    load_dotenv()
    
    # Print header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan] Better Manifold Bot[/bold cyan]\n"
        f"[dim]Target: {args.creator} | Limit: {args.limit} markets[/dim]",
        border_style="cyan"
    ))
    
    api_key = os.getenv("MANIFOLD_API_KEY")
    if not api_key and not args.dry_run:
        console.print("[yellow]  No MANIFOLD_API_KEY found. Forcing dry-run mode.[/yellow]")
        args.dry_run = True
        
    api = ManifoldAPI(api_key=api_key)
    
    if args.use_random:
        console.print("[yellow] Using RandomDecisionMaker[/yellow]")
        decision_maker = RandomDecisionMaker()
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            console.print("[yellow]  No OPENAI_API_KEY found. Falling back to RandomDecisionMaker.[/yellow]")
            decision_maker = RandomDecisionMaker()
        else:
            console.print("[green] Using LLMDecisionMaker (GPT-4o-mini)[/green]")
            decision_maker = LLMDecisionMaker()
            
    bot = SmartBot(
        api=api,
        decision_maker=decision_maker,
        dry_run=args.dry_run,
        target_creator=args.creator
    )
    
    try:
        # Run one cycle immediately
        bot.process_cycle(limit=args.limit)
        
        # Then enter loop if interval > 0
        if args.interval > 0:
            bot.run(limit=args.limit, interval=args.interval)
            
    except KeyboardInterrupt:
        console.print("\n[yellow] Bot stopped by user.[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red] Fatal error:[/bold red] {e}\n")
        logger.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
