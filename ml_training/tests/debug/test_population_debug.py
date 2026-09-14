"""
Debug population issues
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: debug -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator

print("=" * 60)
print("DEBUG: Population Growth Issue")
print("=" * 60)

sim = RealTimeRTSSimulator(num_factions=1)
faction = sim.state.factions[0]

print(f"Starting state:")
print(f"  Population: {faction.population}")
print(f"  Capacity: {faction.population_capacity}")
print(f"  Population < Capacity: {faction.population < faction.population_capacity}")
print(f"  Grain: {faction.get_resource('grain')}")
print(f"  Bread: {faction.get_resource('bread')}")
print(f"  Buildings: {dict(faction.buildings)}")

faction.add_resource('grain', 1000)
faction.add_resource('water', 1000)

total_food = faction.get_resource('grain') + faction.get_resource('bread')
food_per_capita = total_food / faction.population

print(f"\nAfter adding food:")
print(f"  Total food: {total_food}")
print(f"  Food per capita: {food_per_capita:.1f}")
print(f"  Need for growth: 2.0")
print(f"  Meets food requirement: {food_per_capita >= 2.0}")

print(f"\nSimulating 1 hour...")
sim.step({})

print(f"\nAfter 1 hour:")
print(f"  Population: {faction.population}")
print(f"  Grain: {faction.get_resource('grain')}")
print(f"  Bread: {faction.get_resource('bread')}")

# Check why growth isn't happening
print(f"\n" + "=" * 60)
print("GROWTH CONDITIONS CHECK")
print("=" * 60)
print(f"1. Food per capita >= 2.0: {food_per_capita >= 2.0}")
print(f"2. Happiness >= 50: True (hardcoded to 75)")
print(f"3. Population < Capacity: {faction.population < faction.population_capacity}")
print(f"   Population: {faction.population}")
print(f"   Capacity: {faction.population_capacity}")
print(f"\n⚠️  ISSUE: Population equals capacity, so growth check fails!")
print(f"   The condition uses '<' instead of '<=' ")

print(f"\n" + "=" * 60)
print("DEBUG: Starvation Issue")
print("=" * 60)

sim2 = RealTimeRTSSimulator(num_factions=1)
faction2 = sim2.state.factions[0]

print(f"Starting population: {faction2.population}")
print(f"Starting grain: {faction2.get_resource('grain')}")
print(f"Starting bread: {faction2.get_resource('bread')}")
print(f"Buildings: {dict(faction2.buildings)}")

faction2.resources['grain'] = 0
faction2.resources['bread'] = 0

print(f"\nAfter removing food:")
print(f"  Grain: {faction2.get_resource('grain')}")
print(f"  Bread: {faction2.get_resource('bread')}")
print(f"  Total food: {faction2.get_resource('grain') + faction2.get_resource('bread')}")

print(f"\nSimulating 1 hour...")
sim2.step({})

print(f"\nAfter 1 hour:")
print(f"  Population: {faction2.population}")
print(f"  Grain: {faction2.get_resource('grain')}")
print(f"  Bread: {faction2.get_resource('bread')}")

print(f"\n⚠️  ISSUE: Food was produced during step (farms produced grain)")
print(f"   Total food > 0, so starvation check (total_food <= 0) fails!")
