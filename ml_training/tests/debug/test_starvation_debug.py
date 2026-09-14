"""
Deep debug of starvation mechanics
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: debug -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.config import STARVATION_DEATH_RATE

print("=" * 60)
print("DEEP DEBUG: Starvation Mechanics")
print("=" * 60)

sim = RealTimeRTSSimulator(num_factions=1)
faction = sim.state.factions[0]

print(f"Initial state:")
print(f"  Population: {faction.population}")
print(f"  Grain: {faction.get_resource('grain')}")
print(f"  Bread: {faction.get_resource('bread')}")

# Remove all food
faction.resources['grain'] = 0
faction.resources['bread'] = 0

print(f"\nAfter removing food:")
print(f"  Grain: {faction.get_resource('grain')}")
print(f"  Bread: {faction.get_resource('bread')}")
print(f"  Total food: {faction.get_resource('grain') + faction.get_resource('bread')}")
print(f"  Total food <= 0: {(faction.get_resource('grain') + faction.get_resource('bread')) <= 0}")

# Calculate expected deaths
population = faction.population
delta_time = 3600.0  # 1 hour
death_rate_per_second = STARVATION_DEATH_RATE / 3600.0
deaths = population * death_rate_per_second * delta_time

print(f"\nExpected death calculation:")
print(f"  STARVATION_DEATH_RATE: {STARVATION_DEATH_RATE}")
print(f"  Death rate per second: {death_rate_per_second}")
print(f"  Deaths (population * rate * delta_time): {deaths}")
print(f"  deaths >= 1.0: {deaths >= 1.0}")
print(f"  Expected population loss: {int(deaths)}")

# Manually call _update_population to see what happens
print(f"\nCalling _update_population directly...")
sim._update_population(delta_time=3600.0)

print(f"  Population after: {faction.population}")
print(f"  Deaths occurred: {population - faction.population}")

# Also check what happens in full step
print(f"\n" + "=" * 60)
print("Full step test")
print("=" * 60)

sim2 = RealTimeRTSSimulator(num_factions=1)
faction2 = sim2.state.factions[0]

faction2.resources['grain'] = 0
faction2.resources['bread'] = 0

print(f"Before step:")
print(f"  Population: {faction2.population}")
print(f"  Grain: {faction2.get_resource('grain')}")
print(f"  Bread: {faction2.get_resource('bread')}")

sim2.step({})

print(f"\nAfter full step:")
print(f"  Population: {faction2.population}")
print(f"  Grain: {faction2.get_resource('grain')}")
print(f"  Bread: {faction2.get_resource('bread')}")
print(f"  Total food after step: {faction2.get_resource('grain') + faction2.get_resource('bread')}")
