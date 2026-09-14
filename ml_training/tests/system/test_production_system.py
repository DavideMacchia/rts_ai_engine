"""
Test production system to verify resource processing works correctly
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: system -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType


def test_raw_production():
    """Test that raw resource producers work without consuming inputs"""
    print("=" * 60)
    print("TEST 1: Raw Resource Production (Farm)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')
    faction.population = 2  # Farm needs 2 workers

    initial_grain = faction.get_resource('grain')
    print(f"Initial grain: {initial_grain}")
    print(f"Farm count: {faction.get_building_count('farm')}")

    # Simulate 1 hour (3600 seconds)
    sim.game_time = 0
    sim.resource_mgr.produce_resources(faction, delta_time=3600.0)

    # Check grain increased (farm produces 3 grain per hour)
    final_grain = faction.get_resource('grain')
    expected_gain = 3.0  # 3 grain per hour
    actual_gain = final_grain - initial_grain

    print(f"Final grain: {final_grain}")
    print(f"Expected gain: {expected_gain}")
    print(f"Actual gain: {actual_gain}")

    if abs(actual_gain - expected_gain) < 0.1:
        print("✅ PASS: Farm produces grain without consuming inputs\n")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_gain} grain, got {actual_gain}\n")
        return False


def test_processing_with_inputs():
    """Test that processing buildings consume inputs and produce outputs"""
    print("=" * 60)
    print("TEST 2: Processing Building With Inputs (Mill)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Add grain for processing
    faction.add_resource('grain', 10.0)

    faction.add_building('mill')
    faction.population = 1  # Mill needs 1 worker

    # Record initial resources
    initial_grain = faction.get_resource('grain')
    initial_flour = faction.get_resource('flour')

    print(f"Initial grain: {initial_grain}")
    print(f"Initial flour: {initial_flour}")
    print(f"Mill count: {faction.get_building_count('mill')}")

    # Simulate 1 hour (3600 seconds)
    # Mill should consume 2 grain and produce 2 flour per hour
    sim.game_time = 0
    sim.resource_mgr.produce_resources(faction, delta_time=3600.0)

    final_grain = faction.get_resource('grain')
    final_flour = faction.get_resource('flour')

    expected_grain_consumed = 2.0
    expected_flour_produced = 2.0

    actual_grain_consumed = initial_grain - final_grain
    actual_flour_produced = final_flour - initial_flour

    print(f"Final grain: {final_grain}")
    print(f"Final flour: {final_flour}")
    print(f"Expected grain consumed: {expected_grain_consumed}")
    print(f"Actual grain consumed: {actual_grain_consumed}")
    print(f"Expected flour produced: {expected_flour_produced}")
    print(f"Actual flour produced: {actual_flour_produced}")

    grain_ok = abs(actual_grain_consumed - expected_grain_consumed) < 0.1
    flour_ok = abs(actual_flour_produced - expected_flour_produced) < 0.1

    if grain_ok and flour_ok:
        print("✅ PASS: Mill consumes grain and produces flour\n")
        return True
    else:
        print("❌ FAIL: Resource processing incorrect\n")
        return False


def test_processing_without_inputs():
    """Test that processing buildings don't produce without inputs"""
    print("=" * 60)
    print("TEST 3: Processing Building Without Inputs (Mill)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('mill')
    faction.population = 1  # Mill needs 1 worker

    # Ensure no grain available
    faction.resources['grain'] = 0

    initial_flour = faction.get_resource('flour')
    print(f"Initial grain: {faction.get_resource('grain')}")
    print(f"Initial flour: {initial_flour}")
    print(f"Mill count: {faction.get_building_count('mill')}")

    sim.game_time = 0
    sim.resource_mgr.produce_resources(faction, delta_time=3600.0)

    final_flour = faction.get_resource('flour')

    print(f"Final flour: {final_flour}")

    if final_flour == initial_flour:
        print("✅ PASS: Mill does not produce without grain\n")
        return True
    else:
        print(f"❌ FAIL: Mill produced flour without inputs!\n")
        return False


def test_multi_input_processing():
    """Test processing building with multiple inputs (Bakery)"""
    print("=" * 60)
    print("TEST 4: Multi-Input Processing (Bakery)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_resource('flour', 5.0)
    faction.add_resource('water', 5.0)

    faction.add_building('bakery')
    faction.population = 1  # Bakery needs 1 worker

    initial_flour = faction.get_resource('flour')
    initial_water = faction.get_resource('water')
    initial_bread = faction.get_resource('bread')

    print(f"Initial flour: {initial_flour}")
    print(f"Initial water: {initial_water}")
    print(f"Initial bread: {initial_bread}")
    print(f"Bakery count: {faction.get_building_count('bakery')}")

    # Bakery should consume 1 flour + 1 water and produce 1 bread per hour
    sim.game_time = 0
    sim.resource_mgr.produce_resources(faction, delta_time=3600.0)

    final_flour = faction.get_resource('flour')
    final_water = faction.get_resource('water')
    final_bread = faction.get_resource('bread')

    print(f"Final flour: {final_flour}")
    print(f"Final water: {final_water}")
    print(f"Final bread: {final_bread}")

    flour_consumed = initial_flour - final_flour
    water_consumed = initial_water - final_water
    bread_produced = final_bread - initial_bread

    print(f"Flour consumed: {flour_consumed} (expected: 1.0)")
    print(f"Water consumed: {water_consumed} (expected: 1.0)")
    print(f"Bread produced: {bread_produced} (expected: 1.0)")

    flour_ok = abs(flour_consumed - 1.0) < 0.1
    water_ok = abs(water_consumed - 1.0) < 0.1
    bread_ok = abs(bread_produced - 1.0) < 0.1

    if flour_ok and water_ok and bread_ok:
        print("✅ PASS: Bakery consumes flour+water and produces bread\n")
        return True
    else:
        print("❌ FAIL: Multi-input processing incorrect\n")
        return False


def test_production_chain():
    """Test complete production chain: Farm -> Mill -> Bakery"""
    print("=" * 60)
    print("TEST 5: Complete Production Chain (Farm -> Mill -> Bakery)")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')      # Produces grain
    faction.add_building('mill')      # grain -> flour
    faction.add_building('bakery')    # flour + water -> bread
    faction.add_building('well')      # Produces water
    faction.population = 4  # Farm(2) + Mill(1) + Bakery(1) = 4 workers needed

    print("Buildings:")
    print(f"  Farm: {faction.get_building_count('farm')}")
    print(f"  Mill: {faction.get_building_count('mill')}")
    print(f"  Bakery: {faction.get_building_count('bakery')}")
    print(f"  Well: {faction.get_building_count('well')}")

    initial_bread = faction.get_resource('bread')

    print(f"\nInitial bread: {initial_bread}")

    # Simulate several hours to let the chain work
    sim.game_time = 0
    for hour in range(5):
        print(f"\n--- Hour {hour + 1} ---")
        sim.resource_mgr.produce_resources(faction, delta_time=3600.0)

        print(f"Grain: {faction.get_resource('grain'):.2f}")
        print(f"Water: {faction.get_resource('water'):.2f}")
        print(f"Flour: {faction.get_resource('flour'):.2f}")
        print(f"Bread: {faction.get_resource('bread'):.2f}")

    final_bread = faction.get_resource('bread')

    print(f"\nFinal bread: {final_bread}")

    if final_bread > initial_bread:
        print("✅ PASS: Production chain produces bread\n")
        return True
    else:
        print("❌ FAIL: Production chain did not produce bread\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESOURCE PRODUCTION SYSTEM TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("Raw Production", test_raw_production()))
    results.append(("Processing With Inputs", test_processing_with_inputs()))
    results.append(("Processing Without Inputs", test_processing_without_inputs()))
    results.append(("Multi-Input Processing", test_multi_input_processing()))
    results.append(("Production Chain", test_production_chain()))

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
