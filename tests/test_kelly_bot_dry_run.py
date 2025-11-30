import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock manifoldbot before importing kelly_bot
sys.modules['manifoldbot'] = MagicMock()
sys.modules['manifoldbot.manifold'] = MagicMock()
sys.modules['manifoldbot.manifold.bot'] = MagicMock()

# Define the mock classes that KellyBot inherits from/uses
class MockManifoldBot:
    def __init__(self, api_key, decision_maker):
        self.writer = MagicMock()
    
    def place_bet_if_decision(self, decision, bet_amount):
        pass

sys.modules['manifoldbot.manifold.bot'].ManifoldBot = MockManifoldBot
sys.modules['manifoldbot.manifold.bot'].MarketDecision = MagicMock

from better_manifold_bot.kelly_bot import KellyBot

class TestKellyBotDryRun(unittest.TestCase):
    def setUp(self):
        self.mock_api_key = "test_key"
        self.mock_decision_maker = MagicMock()
        self.bot = KellyBot(
            manifold_api_key=self.mock_api_key,
            decision_maker=self.mock_decision_maker,
            dry_run=True
        )
        # Mock the writer to return a balance
        self.bot.writer = MagicMock()
        self.bot.writer.get_balance.return_value = 1000.0
        
    def test_dry_run_does_not_place_bet(self):
        decision = MagicMock()
        decision.decision = "YES"
        decision.current_probability = 0.5
        decision.metadata = {"ensemble_probability": 0.6}
        
        # We need to mock place_bet_if_decision of the parent class (MockManifoldBot)
        # We can patch the method on the class
        
        with patch.object(MockManifoldBot, 'place_bet_if_decision') as mock_super_place_bet:
            success, amount = self.bot.place_bet_if_decision(decision, bet_amount=10.0)
            
            self.assertTrue(success)
            self.assertEqual(amount, 10.0)
            mock_super_place_bet.assert_not_called()

    def test_live_run_places_bet(self):
        bot_live = KellyBot(
            manifold_api_key=self.mock_api_key,
            decision_maker=self.mock_decision_maker,
            dry_run=False
        )
        bot_live.writer = MagicMock()
        bot_live.writer.get_balance.return_value = 1000.0
        
        decision = MagicMock()
        decision.decision = "YES"
        decision.current_probability = 0.5
        decision.metadata = {"ensemble_probability": 0.6}
        
        with patch.object(MockManifoldBot, 'place_bet_if_decision') as mock_super_place_bet:
            mock_super_place_bet.return_value = (True, 10.0)
            
            success, amount = bot_live.place_bet_if_decision(decision, bet_amount=10.0)
            
            self.assertTrue(success)
            self.assertEqual(amount, 10.0)
            mock_super_place_bet.assert_called_once()

if __name__ == '__main__':
    unittest.main()
