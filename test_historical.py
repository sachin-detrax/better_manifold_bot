"""Test Historical Signal data loading."""

import os
import logging
from dotenv import load_dotenv
from better_manifold_bot.signals.historical_signal import HistoricalSignal

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Test loading with MikhailTal as target
h = HistoricalSignal(api_key=os.getenv('MANIFOLD_API_KEY'), target_creator='MikhailTal')

print(f"\nLoaded: {h.loaded}")
print(f"Total creators with data: {len(h.creator_stats)}")
print(f"MikhailTal in stats: {'MikhailTal' in h.creator_stats}")

if 'MikhailTal' in h.creator_stats:
    stats = h.creator_stats['MikhailTal']
    print(f"MikhailTal YES rate: {stats['yes_rate']:.1%} ({stats['count']} markets)")
    print(f"\nSUCCESS! Historical Signal now has MikhailTal data!")
else:
    print("ERROR: MikhailTal NOT found in stats")
    print(f"Available creators: {list(h.creator_stats.keys())}")
