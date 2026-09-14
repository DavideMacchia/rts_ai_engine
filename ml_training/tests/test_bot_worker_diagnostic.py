"""
Detailed diagnostic of NormalBot worker assignments and productivity.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.opponents import NormalBot  # deprecated alias of TestingBot
from simulator.config import WORKERS_NEEDED


def analyze_faction(faction, name):
    """Analyze worker assignments and productivity for a faction."""
    print(f"\n{'='*60}")
    print(f"{name} WORKER ANALYSIS")
    print(f"{'='*60}")

    # Basic stats
    available_workers = faction.get_available_workers()
    total_needed = faction.get_total_workers_needed()

    print(f"Population: {faction.population}")
    print(f"Available Workers: {available_workers}")
    print(f"Total Workers Needed: {total_needed}")
    print(f"Worker Deficit: {total_needed - available_workers}")

    # Worker assignments
    assignments = faction.calculate_worker_assignments()
    print(f"\nWorker Assignments:")
    for building_type in sorted(faction.buildings.keys()):
        count = faction.buildings[building_type]
        if count > 0:
            workers_per_building = WORKERS_NEEDED.get(building_type, 0)
            total_needed_for_type = workers_per_building * count
            assigned = assignments.get(building_type, 0)
            productivity = faction.get_building_productivity(building_type)

            print(f"  {building_type} ({count}x):")
            print(f"    Need: {total_needed_for_type} workers ({workers_per_building} each)")
            print(f"    Assigned: {assigned} workers")
            print(f"    Productivity: {productivity:.1%}")

    # Resources
    print(f"\nResources:")
    for resource in ['wood', 'stone', 'grain', 'water']:
        amount = faction.resources.get(resource, 0)
        print(f"  {resource}: {amount:.1f}")


def test_worker_diagnostic():
    """Run a short simulation and analyze worker assignments."""
    print("NORMALBOT WORKER DIAGNOSTIC TEST")
    print("="*60)

    # Create simulator
    sim = RealTimeRTSSimulator(num_factions=1)
    bot = NormalBot(faction_id=0)

    # Simulate for 8 hours
    duration_seconds = 8 * 3600
    time_step = 60.0

    game_time = 0.0
    while game_time < duration_seconds:
        action = bot.act(sim.state, game_time)
        sim.step({0: action}, delta_time=time_step)
        game_time += time_step

    # Analyze
    analyze_faction(sim.state.factions[0], "FACTION 0")

    # Check if stone is producing
    faction = sim.state.factions[0]
    quarry_productivity = faction.get_building_productivity('quarry')
    quarries = faction.get_building_count('quarry')

    print(f"\n{'='*60}")
    print("STONE PRODUCTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Quarries: {quarries}")
    print(f"Quarry Productivity: {quarry_productivity:.1%}")
    print(f"Expected stone/hour: {30 * quarries * quarry_productivity:.1f}")
    print(f"Current stone: {faction.resources.get('stone', 0):.1f}")

    if quarry_productivity < 0.5:
        print(f"\n⚠️  LOW PRODUCTIVITY! Quarries running at {quarry_productivity:.0%}")
        print(f"   Need more workers to increase stone production!")


if __name__ == '__main__':
    test_worker_diagnostic()
