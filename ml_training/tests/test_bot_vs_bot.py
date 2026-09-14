"""
Test NormalBot vs NormalBot to analyze behavior tree bot performance.

This test helps diagnose issues when training against the bot:
- Checks if bot can play properly against itself
- Monitors population, buildings, and military growth
- Identifies potential balance or implementation issues
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.opponents import NormalBot  # deprecated alias of TestingBot
import time


def print_faction_stats(faction, faction_name, game_time_hours):
    """Print detailed statistics for a faction."""
    # Buildings is a dict mapping type to count
    total_buildings = sum(faction.buildings.values())

    # Units is a dict mapping unit type to count
    soldiers = faction.units.get('soldier', 0)
    archers = faction.units.get('archer', 0)
    total_military = soldiers + archers

    print(f"\n{'='*60}")
    print(f"{faction_name} @ {game_time_hours:.1f} hours")
    print(f"{'='*60}")
    print(f"Population: {faction.population}/{faction.population_capacity}")
    print(f"Resources: Wood={faction.resources.get('wood', 0):.0f}, "
          f"Stone={faction.resources.get('stone', 0):.0f}, "
          f"Grain={faction.resources.get('grain', 0):.0f}")
    print(f"\nBuildings ({total_buildings} total):")
    for btype, count in sorted(faction.buildings.items()):
        if count > 0:
            print(f"  {btype}: {count}")
    print(f"\nMilitary (strength={faction.military_strength:.1f}):")
    print(f"  Soldiers: {soldiers}")
    print(f"  Archers: {archers}")
    print(f"  Total: {total_military}")


def test_bot_vs_bot(duration_hours=24, print_interval_hours=4):
    """
    Run NormalBot vs NormalBot and monitor the game.

    Args:
        duration_hours: How long to run the simulation (in game hours)
        print_interval_hours: How often to print stats
    """
    print("\n" + "="*60)
    print("NORMALBOT VS NORMALBOT TEST")
    print("="*60)
    print(f"Duration: {duration_hours} game hours")
    print(f"Stats interval: {print_interval_hours} hours")
    print()

    # Create simulator with 2 factions
    sim = RealTimeRTSSimulator(num_factions=2)

    # Create two NormalBot opponents
    bot1 = NormalBot(faction_id=0)
    bot2 = NormalBot(faction_id=1)

    # Time settings
    decision_interval = 60.0  # Bots decide every 60 seconds
    time_step = 60.0  # Simulate in 60-second chunks
    total_seconds = duration_hours * 3600
    next_print_time = print_interval_hours * 3600

    print("Starting simulation...")
    start_real_time = time.time()

    game_time = 0.0
    while game_time < total_seconds:
        # Get actions from both bots
        action1 = bot1.act(sim.state, game_time)
        action2 = bot2.act(sim.state, game_time)

        # Step simulation
        actions = {0: action1, 1: action2}
        sim.step(actions, delta_time=time_step)

        game_time += time_step

        # Print stats at intervals
        if game_time >= next_print_time:
            game_time_hours = game_time / 3600
            print_faction_stats(sim.state.factions[0], "FACTION 0 (Bot 1)", game_time_hours)
            print_faction_stats(sim.state.factions[1], "FACTION 1 (Bot 2)", game_time_hours)

            # Check if either faction is dead
            if sim.state.factions[0].population <= 0:
                print("\n⚠️  FACTION 0 IS DEAD!")
                break
            if sim.state.factions[1].population <= 0:
                print("\n⚠️  FACTION 1 IS DEAD!")
                break

            next_print_time += print_interval_hours * 3600

    # Final stats
    game_time_hours = game_time / 3600
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print_faction_stats(sim.state.factions[0], "FACTION 0 (Bot 1)", game_time_hours)
    print_faction_stats(sim.state.factions[1], "FACTION 1 (Bot 2)", game_time_hours)

    # Summary
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)

    f0 = sim.state.factions[0]
    f1 = sim.state.factions[1]

    # Compare populations
    print(f"\nPopulation Growth:")
    print(f"  Faction 0: {f0.population} / {f0.population_capacity}")
    print(f"  Faction 1: {f1.population} / {f1.population_capacity}")

    # Compare military
    mil0 = f0.units.get('soldier', 0) + f0.units.get('archer', 0)
    mil1 = f1.units.get('soldier', 0) + f1.units.get('archer', 0)
    print(f"\nMilitary Strength:")
    print(f"  Faction 0: {mil0} units (strength={f0.military_strength:.1f})")
    print(f"  Faction 1: {mil1} units (strength={f1.military_strength:.1f})")

    # Compare buildings
    total_buildings0 = sum(f0.buildings.values())
    total_buildings1 = sum(f1.buildings.values())
    print(f"\nBuildings:")
    print(f"  Faction 0: {total_buildings0} buildings")
    print(f"  Faction 1: {total_buildings1} buildings")

    # Check for stagnation
    print(f"\nPotential Issues:")
    if f0.population <= 5 or f1.population <= 5:
        print("  ⚠️  LOW POPULATION - Bot may be struggling with economy")
    if mil0 == 0 and mil1 == 0 and game_time_hours > 12:
        print("  ⚠️  NO MILITARY - Bots not building army")
    if total_buildings0 < 5 or total_buildings1 < 5:
        print("  ⚠️  FEW BUILDINGS - Bot may be stuck or inefficient")
    if abs(f0.population - f1.population) < 2 and abs(mil0 - mil1) < 2:
        print("  ℹ️  PERFECTLY BALANCED - Both bots performing identically")
        print("     This could explain why training plateaus at ~60% win rate")

    elapsed_real_time = time.time() - start_real_time
    print(f"\nSimulation completed in {elapsed_real_time:.1f} real seconds")
    print("="*60)


if __name__ == '__main__':
    # Run a 24-hour game simulation, printing stats every 4 hours
    test_bot_vs_bot(duration_hours=24, print_interval_hours=4)
