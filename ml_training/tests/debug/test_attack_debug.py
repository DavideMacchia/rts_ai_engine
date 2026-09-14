"""
Quick debug test to understand attack behavior
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: debug -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType

# Test 1: Attack without army
print("=" * 60)
print("DEBUG: Attack without army")
print("=" * 60)

sim = RealTimeRTSSimulator(num_factions=2)
attacker = sim.state.factions[0]
defender = sim.state.factions[1]

print(f"Before setup:")
print(f"  Attacker units: {attacker.units}")
print(f"  Attacker strength: {attacker.military_strength}")

# Set attacker to have 0 units
attacker.units['soldier'] = 0
attacker.calculate_military_strength()

print(f"\nAfter setting units to 0:")
print(f"  Attacker units: {attacker.units}")
print(f"  Attacker strength: {attacker.military_strength}")
print(f"  Attacker strength <= 0: {attacker.military_strength <= 0}")

# Give defender some army
defender.units['soldier'] = 5
defender.calculate_military_strength()

print(f"\nDefender:")
print(f"  Defender units: {defender.units}")
print(f"  Defender strength: {defender.military_strength}")

# Execute attack
action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
_, rewards, _ = sim.step({0: action})

print(f"\nAfter attack:")
print(f"  Attacker reward: {rewards[0]}")
print(f"  Defender units after: {defender.units}")
print(f"  Attack was {'BLOCKED' if rewards[0] < 0 else 'ALLOWED'}")

# Test 2: Units not taking casualties
print("\n" + "=" * 60)
print("DEBUG: Large army attack to see casualties")
print("=" * 60)

sim2 = RealTimeRTSSimulator(num_factions=2)
attacker2 = sim2.state.factions[0]
defender2 = sim2.state.factions[1]

attacker2.units['soldier'] = 100
attacker2.calculate_military_strength()

defender2.units['soldier'] = 10
defender2.calculate_military_strength()

print(f"Before attack:")
print(f"  Attacker: {attacker2.units['soldier']} soldiers")
print(f"  Defender: {defender2.units['soldier']} soldiers")

action2 = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
sim2.step({0: action2})

print(f"\nAfter attack:")
print(f"  Attacker: {attacker2.units['soldier']} soldiers (expected ~90, lost 10%)")
print(f"  Defender: {defender2.units['soldier']} soldiers (expected ~6, lost 40%)")

# Calculate expected
expected_attacker_remaining = 100 - int(100 * 0.1)
expected_defender_remaining = 10 - int(10 * 0.4)
print(f"\nExpected values:")
print(f"  Attacker: {expected_attacker_remaining}")
print(f"  Defender: {expected_defender_remaining}")
