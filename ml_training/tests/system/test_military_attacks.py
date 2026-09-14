"""
Test military attack system to verify combat mechanics work correctly
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: system -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType
from simulator.config import (
    DEFENDER_BONUS,
    COMBAT_CASUALTIES_RATE,
    WINNER_CASUALTIES_RATE,
    BUILDING_DAMAGE_CHANCE,
    UNIT_STRENGTH
)


def test_basic_attack_attacker_wins():
    """Test basic attack where attacker has superior force"""
    print("=" * 60)
    print("TEST 1: Basic Attack - Attacker Wins")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    attacker = sim.state.factions[0]
    defender = sim.state.factions[1]

    # Give attacker strong army (10 soldiers = 10 strength)
    attacker.units['soldier'] = 10
    attacker.calculate_military_strength()
    print(f"Attacker soldiers: {attacker.units['soldier']}")
    print(f"Attacker strength: {attacker.military_strength}")

    # Give defender weak army (2 soldiers = 2 strength, 2 * 1.3 = 2.6 with bonus)
    defender.units['soldier'] = 2
    defender.calculate_military_strength()
    print(f"Defender soldiers: {defender.units['soldier']}")
    print(f"Defender strength: {defender.military_strength}")
    print(f"Defender adjusted strength (with bonus): {defender.military_strength * DEFENDER_BONUS}")

    initial_attacker_units = attacker.units['soldier']
    initial_defender_units = defender.units['soldier']

    print("\nExecuting attack...")
    action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
    sim.step({0: action})

    attacker_units_after = attacker.units['soldier']
    defender_units_after = defender.units['soldier']

    # With the minimum casualty fix, small armies lose at least 1 unit
    expected_attacker_casualties = max(1, int(initial_attacker_units * WINNER_CASUALTIES_RATE))
    expected_defender_casualties = max(1, int(initial_defender_units * COMBAT_CASUALTIES_RATE))

    actual_attacker_casualties = initial_attacker_units - attacker_units_after
    actual_defender_casualties = initial_defender_units - defender_units_after

    print(f"\nAttacker casualties: {actual_attacker_casualties} (expected: {expected_attacker_casualties})")
    print(f"Defender casualties: {actual_defender_casualties} (expected: {expected_defender_casualties})")
    print(f"Attacker soldiers remaining: {attacker_units_after}")
    print(f"Defender soldiers remaining: {defender_units_after}")

    # Verify attacker lost ~10% and defender lost ~40% (minimum 1)
    attacker_ok = actual_attacker_casualties == expected_attacker_casualties
    defender_ok = actual_defender_casualties == expected_defender_casualties

    if attacker_ok and defender_ok:
        print("✅ PASS: Attacker wins, casualties applied correctly\n")
        return True
    else:
        print("❌ FAIL: Casualties not applied correctly\n")
        return False


def test_defender_wins():
    """Test attack where defender has superior force"""
    print("=" * 60)
    print("TEST 2: Attack - Defender Wins")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    attacker = sim.state.factions[0]
    defender = sim.state.factions[1]

    # Give attacker weak army (3 soldiers = 3 strength)
    attacker.units['soldier'] = 3
    attacker.calculate_military_strength()
    print(f"Attacker soldiers: {attacker.units['soldier']}")
    print(f"Attacker strength: {attacker.military_strength}")

    # Give defender strong army (10 soldiers = 10 strength, 10 * 1.3 = 13 with bonus)
    defender.units['soldier'] = 10
    defender.calculate_military_strength()
    print(f"Defender soldiers: {defender.units['soldier']}")
    print(f"Defender strength: {defender.military_strength}")
    print(f"Defender adjusted strength (with bonus): {defender.military_strength * DEFENDER_BONUS}")

    initial_attacker_units = attacker.units['soldier']
    initial_defender_units = defender.units['soldier']

    print("\nExecuting attack...")
    action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
    sim.step({0: action})

    attacker_units_after = attacker.units['soldier']
    defender_units_after = defender.units['soldier']

    # With the minimum casualty fix, small armies lose at least 1 unit
    expected_attacker_casualties = max(1, int(initial_attacker_units * COMBAT_CASUALTIES_RATE))
    expected_defender_casualties = max(1, int(initial_defender_units * WINNER_CASUALTIES_RATE))

    actual_attacker_casualties = initial_attacker_units - attacker_units_after
    actual_defender_casualties = initial_defender_units - defender_units_after

    print(f"\nAttacker casualties: {actual_attacker_casualties} (expected: {expected_attacker_casualties})")
    print(f"Defender casualties: {actual_defender_casualties} (expected: {expected_defender_casualties})")
    print(f"Attacker soldiers remaining: {attacker_units_after}")
    print(f"Defender soldiers remaining: {defender_units_after}")

    # Verify attacker lost ~40% and defender lost ~10% (minimum 1)
    attacker_ok = actual_attacker_casualties == expected_attacker_casualties
    defender_ok = actual_defender_casualties == expected_defender_casualties

    if attacker_ok and defender_ok:
        print("✅ PASS: Defender wins, casualties applied correctly\n")
        return True
    else:
        print("❌ FAIL: Casualties not applied correctly\n")
        return False


def test_attack_without_army():
    """Test that attack fails when attacker has no military"""
    print("=" * 60)
    print("TEST 3: Attack Without Army (Should Fail)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    attacker = sim.state.factions[0]
    defender = sim.state.factions[1]

    attacker.units['soldier'] = 0
    attacker.calculate_military_strength()
    print(f"Attacker soldiers: {attacker.units['soldier']}")
    print(f"Attacker strength: {attacker.military_strength}")

    defender.units['soldier'] = 5
    defender.calculate_military_strength()
    print(f"Defender soldiers: {defender.units['soldier']}")
    print(f"Defender strength: {defender.military_strength}")

    initial_defender_units = defender.units['soldier']

    print("\nAttempting attack without army...")
    action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
    _, rewards, _ = sim.step({0: action})

    # Check that defender was not damaged (attack should be rejected)
    defender_units_after = defender.units['soldier']

    print(f"Defender soldiers after: {defender_units_after}")
    print(f"Attacker total reward: {rewards[0]}")

    # Attack should be blocked - defender should be unharmed
    # (Note: total reward may be positive due to milestone rewards,
    # but the key check is that defender took no damage)
    if defender_units_after == initial_defender_units:
        print("✅ PASS: Attack without army correctly blocked (defender undamaged)\n")
        return True
    else:
        print("❌ FAIL: Invalid attack damaged defender\n")
        return False


def test_mixed_unit_types():
    """Test combat with mixed unit types (soldiers, archers, cavalry)"""
    print("=" * 60)
    print("TEST 4: Mixed Unit Types Combat")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    attacker = sim.state.factions[0]
    defender = sim.state.factions[1]

    # Give attacker mixed army: 5 soldiers (5.0) + 5 cavalry (10.0) = 15.0 strength
    attacker.units['soldier'] = 5
    attacker.units['cavalry'] = 5
    attacker.calculate_military_strength()
    print(f"Attacker composition:")
    print(f"  Soldiers: {attacker.units['soldier']} (strength: {5 * UNIT_STRENGTH['soldier']})")
    print(f"  Cavalry: {attacker.units['cavalry']} (strength: {5 * UNIT_STRENGTH['cavalry']})")
    print(f"  Total strength: {attacker.military_strength}")

    # Give defender: 10 archers (8.0 strength, 8.0 * 1.3 = 10.4 with bonus)
    defender.units['archer'] = 10
    defender.calculate_military_strength()
    print(f"\nDefender composition:")
    print(f"  Archers: {defender.units['archer']} (strength: {10 * UNIT_STRENGTH['archer']})")
    print(f"  Total strength: {defender.military_strength}")
    print(f"  Adjusted strength (with bonus): {defender.military_strength * DEFENDER_BONUS}")

    initial_attacker_soldiers = attacker.units['soldier']
    initial_attacker_cavalry = attacker.units['cavalry']
    initial_defender_archers = defender.units['archer']

    print("\nExecuting attack...")
    action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
    sim.step({0: action})

    # Check that casualties were applied to all unit types
    print(f"\nAttacker casualties:")
    print(f"  Soldiers: {initial_attacker_soldiers} -> {attacker.units['soldier']}")
    print(f"  Cavalry: {initial_attacker_cavalry} -> {attacker.units['cavalry']}")
    print(f"\nDefender casualties:")
    print(f"  Archers: {initial_defender_archers} -> {defender.units['archer']}")

    # Verify both attacker unit types took casualties
    soldiers_took_casualties = attacker.units['soldier'] < initial_attacker_soldiers
    cavalry_took_casualties = attacker.units['cavalry'] < initial_attacker_cavalry
    archers_took_casualties = defender.units['archer'] < initial_defender_archers

    if soldiers_took_casualties and cavalry_took_casualties and archers_took_casualties:
        print("✅ PASS: Casualties applied to all unit types\n")
        return True
    else:
        print("❌ FAIL: Not all unit types took casualties\n")
        return False


def test_building_damage():
    """Test that buildings can be damaged in combat"""
    print("=" * 60)
    print("TEST 5: Building Damage (Multiple Attacks)")
    print("=" * 60)

    # Run multiple attacks to ensure building damage occurs at least once
    building_damaged = False
    num_attempts = 20  # With 30% chance, we should see damage

    for attempt in range(num_attempts):
        sim = RealTimeRTSSimulator(num_factions=2)
        attacker = sim.state.factions[0]
        defender = sim.state.factions[1]

        # Give attacker strong army
        attacker.units['soldier'] = 20
        attacker.calculate_military_strength()

        defender.units['soldier'] = 2
        defender.calculate_military_strength()

        # Give defender some buildings to damage
        defender.buildings['farm'] = 3
        defender.buildings['barracks'] = 2

        initial_total_buildings = sum(defender.buildings.values())

        action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
        sim.step({0: action})

        final_total_buildings = sum(defender.buildings.values())

        if final_total_buildings < initial_total_buildings:
            building_damaged = True
            print(f"Building damaged on attempt {attempt + 1}")
            print(f"Buildings before: {initial_total_buildings}")
            print(f"Buildings after: {final_total_buildings}")
            print(f"Defender buildings: {dict(defender.buildings)}")
            break

    if building_damaged:
        print("✅ PASS: Building damage occurs on successful attacks\n")
        return True
    else:
        print(f"❌ FAIL: No building damage after {num_attempts} attacks (expected ~30% chance)\n")
        return False


def test_military_strength_calculation():
    """Test that military strength is calculated correctly for different unit types"""
    print("=" * 60)
    print("TEST 6: Military Strength Calculation")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Add different unit types
    faction.units['soldier'] = 10   # 10 * 1.0 = 10.0
    faction.units['archer'] = 5     # 5 * 0.8 = 4.0
    faction.units['cavalry'] = 2    # 2 * 2.0 = 4.0
    # Total expected: 18.0

    expected_strength = (
        10 * UNIT_STRENGTH['soldier'] +
        5 * UNIT_STRENGTH['archer'] +
        2 * UNIT_STRENGTH['cavalry']
    )
    faction.calculate_military_strength()
    actual_strength = faction.military_strength

    print(f"Unit composition:")
    print(f"  10 Soldiers × {UNIT_STRENGTH['soldier']} = {10 * UNIT_STRENGTH['soldier']}")
    print(f"  5 Archers × {UNIT_STRENGTH['archer']} = {5 * UNIT_STRENGTH['archer']}")
    print(f"  2 Cavalry × {UNIT_STRENGTH['cavalry']} = {2 * UNIT_STRENGTH['cavalry']}")
    print(f"\nExpected total strength: {expected_strength}")
    print(f"Actual total strength: {actual_strength}")

    if abs(actual_strength - expected_strength) < 0.01:
        print("✅ PASS: Military strength calculated correctly\n")
        return True
    else:
        print("❌ FAIL: Military strength calculation incorrect\n")
        return False


def test_units_die_completely():
    """Test that units can be completely eliminated in combat"""
    print("=" * 60)
    print("TEST 7: Complete Unit Elimination")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    attacker = sim.state.factions[0]
    defender = sim.state.factions[1]

    # Give attacker overwhelming force
    attacker.units['soldier'] = 100
    attacker.calculate_military_strength()
    print(f"Attacker soldiers: {attacker.units['soldier']}")
    print(f"Attacker strength: {attacker.military_strength}")

    # Give defender minimal army (will be wiped out)
    defender.units['soldier'] = 2
    defender.calculate_military_strength()
    print(f"Defender soldiers: {defender.units['soldier']}")
    print(f"Defender strength: {defender.military_strength}")

    # Execute multiple attacks to eliminate defender
    print("\nExecuting multiple attacks...")
    for i in range(10):  # Increased to ensure elimination
        # Recalculate strength to check current state
        defender.calculate_military_strength()
        attacker.calculate_military_strength()

        if defender.military_strength <= 0 or defender.units['soldier'] <= 0:
            print(f"  Defender eliminated after {i} attacks")
            break

        action = Action(ActionType.ATTACK, faction_id=0, target_faction_id=1)
        sim.step({0: action})
        print(f"  Attack {i+1}: Defender soldiers: {defender.units['soldier']}")

    # Final check - recalculate one more time
    defender.calculate_military_strength()

    # Check if defender was completely eliminated
    if defender.military_strength == 0 and defender.units['soldier'] == 0:
        print("\n✅ PASS: Units can be completely eliminated\n")
        return True
    else:
        print(f"\n❌ FAIL: Defender still has {defender.units['soldier']} soldiers, {defender.military_strength} strength\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MILITARY ATTACK SYSTEM TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("Basic Attack - Attacker Wins", test_basic_attack_attacker_wins()))
    results.append(("Attack - Defender Wins", test_defender_wins()))
    results.append(("Attack Without Army", test_attack_without_army()))
    results.append(("Mixed Unit Types", test_mixed_unit_types()))
    results.append(("Building Damage", test_building_damage()))
    results.append(("Military Strength Calculation", test_military_strength_calculation()))
    results.append(("Complete Unit Elimination", test_units_die_completely()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
