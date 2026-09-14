"""
Test population growth and death mechanics
"""
import sys
import os
# Add rl directory to path (go up 3 levels: system -> tests -> rl)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.config import (
    MIN_FOOD_PER_CAPITA_FOR_GROWTH,
    BIRTH_CHANCE_PER_TURN,
    STARVATION_DEATH_RATE,
    POPULATION_FOOD_CONSUMPTION,
    POPULATION_WATER_CONSUMPTION,
    STARTING_POPULATION
)


def test_population_growth_with_food():
    """Test that population grows when conditions are met"""
    print("=" * 60)
    print("TEST 1: Population Growth with Sufficient Food")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    initial_population = faction.population
    print(f"Initial population: {initial_population}")
    print(f"Population capacity: {faction.population_capacity}")

    # Give plenty of food (need 2.0 per capita for growth)
    food_needed = initial_population * MIN_FOOD_PER_CAPITA_FOR_GROWTH * 50  # Extra food
    faction.add_resource('grain', food_needed)
    faction.add_resource('water', 1000)  # Plenty of water

    print(f"Food provided: {food_needed} grain")
    print(f"Food per capita: {food_needed / initial_population:.1f}")

    # Simulate many hours to trigger growth (10% chance per hour)
    print("\nSimulating 100 hours (expecting growth)...")
    growth_occurred = False
    for hour in range(100):
        sim.step({}, delta_time=3600.0)  # 3600 seconds = 1 hour
        if faction.population > initial_population:
            growth_occurred = True
            print(f"  Growth at hour {hour + 1}: {faction.population} population")
            break

    final_population = faction.population

    print(f"\nFinal population: {final_population}")
    print(f"Growth occurred: {growth_occurred}")

    if growth_occurred and final_population > initial_population:
        print("✅ PASS: Population grew with sufficient food\n")
        return True
    else:
        print("❌ FAIL: Population did not grow\n")
        return False


def test_population_no_growth_without_food():
    """Test that population doesn't grow without sufficient food"""
    print("=" * 60)
    print("TEST 2: No Population Growth Without Food")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    initial_population = faction.population
    print(f"Initial population: {initial_population}")

    faction.resources['grain'] = 0
    faction.resources['bread'] = 0
    faction.add_resource('water', 1000)  # Still have water

    print("Food: 0 (insufficient for growth)")

    print("\nSimulating 50 hours (expecting NO growth)...")
    for hour in range(50):
        sim.step({}, delta_time=3600.0)

    final_population = faction.population

    print(f"Final population: {final_population}")

    # Population should not grow (it might die from starvation though)
    if final_population <= initial_population:
        print("✅ PASS: Population did not grow without food\n")
        return True
    else:
        print("❌ FAIL: Population grew without food\n")
        return False


def test_population_no_growth_at_capacity():
    """Test that population doesn't grow beyond capacity"""
    print("=" * 60)
    print("TEST 3: No Growth Beyond Population Capacity")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Set population to capacity
    faction.calculate_population_capacity()
    faction.population = faction.population_capacity

    initial_population = faction.population
    print(f"Population: {initial_population}")
    print(f"Capacity: {faction.population_capacity}")
    print(f"At capacity: {initial_population >= faction.population_capacity}")

    # Give plenty of food (extra to account for consumption over 100 hours)
    food_needed = initial_population * MIN_FOOD_PER_CAPITA_FOR_GROWTH * 200
    faction.add_resource('grain', food_needed)
    faction.add_resource('water', 10000)  # Lots of water

    print(f"Food per capita: {food_needed / initial_population:.1f} (sufficient)")

    print("\nSimulating 100 hours (expecting NO growth - at capacity)...")
    for hour in range(100):
        sim.step({}, delta_time=3600.0)

    final_population = faction.population

    print(f"Final population: {final_population}")
    print(f"Capacity: {faction.population_capacity}")

    if final_population == initial_population:
        print("✅ PASS: Population did not grow beyond capacity\n")
        return True
    else:
        print("❌ FAIL: Population grew beyond capacity\n")
        return False


def test_population_death_from_starvation():
    """Test that population dies from starvation"""
    print("=" * 60)
    print("TEST 4: Population Death from Starvation")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    initial_population = faction.population
    print(f"Initial population: {initial_population}")

    # Remove ALL food to trigger starvation
    faction.resources['grain'] = 0
    faction.resources['bread'] = 0
    faction.add_resource('water', 1000)  # Water doesn't prevent starvation

    print("Food: 0 (starvation conditions)")
    print(f"Expected death rate: {STARVATION_DEATH_RATE * 100}% per turn")

    # Simulate several hours - should see deaths
    print("\nSimulating 20 hours (expecting population decline)...")
    deaths_occurred = False
    for hour in range(20):
        sim.step({}, delta_time=3600.0)
        if faction.population < initial_population:
            deaths_occurred = True
            print(f"  Deaths at hour {hour + 1}: {faction.population} population")
            break

    final_population = faction.population

    print(f"\nInitial population: {initial_population}")
    print(f"Final population: {final_population}")
    print(f"Deaths: {initial_population - final_population}")

    if deaths_occurred and final_population < initial_population:
        print("✅ PASS: Population died from starvation\n")
        return True
    else:
        print("❌ FAIL: Population did not die from starvation\n")
        return False


def test_food_consumption():
    """Test that population consumes food correctly"""
    print("=" * 60)
    print("TEST 5: Food Consumption")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    population = faction.population
    print(f"Population: {population}")

    # Remove all production buildings to isolate consumption
    faction.buildings['farm'] = 0

    # Check actual starting grain (includes STARTING_RESOURCES)
    initial_grain_total = faction.get_resource('grain')

    faction.add_resource('grain', 100.0)
    faction.add_resource('water', 1000)

    initial_grain_total = faction.get_resource('grain')  # Get actual total
    print(f"Initial grain (total): {initial_grain_total}")
    print(f"Expected consumption per hour: {POPULATION_FOOD_CONSUMPTION * population}")
    print("(Note: Farms removed to isolate consumption)")

    # Simulate 1 hour (3600 seconds)
    sim.step({}, delta_time=3600.0)

    final_grain = faction.get_resource('grain')
    consumed = initial_grain_total - final_grain

    print(f"Final grain: {final_grain}")
    print(f"Consumed: {consumed}")

    expected_consumption = POPULATION_FOOD_CONSUMPTION * population

    # Allow small floating point error
    if abs(consumed - expected_consumption) < 0.1:
        print("✅ PASS: Food consumption correct\n")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_consumption}, got {consumed}\n")
        return False


def test_bread_preferred_over_grain():
    """Test that bread is consumed before grain"""
    print("=" * 60)
    print("TEST 6: Bread Consumed Before Grain")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    population = faction.population
    print(f"Population: {population}")

    # Give both bread and grain
    initial_bread = 10.0
    initial_grain = 100.0
    faction.add_resource('bread', initial_bread)
    faction.add_resource('grain', initial_grain)
    faction.add_resource('water', 1000)

    print(f"Initial bread: {initial_bread}")
    print(f"Initial grain: {initial_grain}")

    sim.step({}, delta_time=3600.0)

    final_bread = faction.get_resource('bread')
    final_grain = faction.get_resource('grain')

    bread_consumed = initial_bread - final_bread
    grain_consumed = initial_grain - final_grain

    print(f"Final bread: {final_bread}")
    print(f"Final grain: {final_grain}")
    print(f"Bread consumed: {bread_consumed}")
    print(f"Grain consumed: {grain_consumed}")

    expected_total = POPULATION_FOOD_CONSUMPTION * population

    # Bread should be consumed first
    # If we have enough bread, grain shouldn't be touched
    if bread_consumed > 0:
        if initial_bread >= expected_total:
            # Had enough bread, grain should be untouched
            if grain_consumed < 0.1:
                print("✅ PASS: Bread consumed, grain untouched\n")
                return True
        else:
            # Didn't have enough bread, should consume remaining from grain
            total_consumed = bread_consumed + grain_consumed
            if abs(total_consumed - expected_total) < 0.1 and bread_consumed == initial_bread:
                print("✅ PASS: Bread fully consumed, then grain\n")
                return True

    print("❌ FAIL: Bread not prioritized correctly\n")
    return False


def test_water_consumption():
    """Test that population consumes water"""
    print("=" * 60)
    print("TEST 7: Water Consumption")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    population = faction.population
    print(f"Population: {population}")

    # Remove wells to isolate consumption
    if 'well' in faction.buildings:
        faction.buildings['well'] = 0

    # Check actual starting water (includes STARTING_RESOURCES)
    faction.add_resource('water', 100.0)
    faction.add_resource('grain', 1000)  # Plenty of food

    initial_water_total = faction.get_resource('water')  # Get actual total
    print(f"Initial water (total): {initial_water_total}")
    print(f"Expected consumption per hour: {POPULATION_WATER_CONSUMPTION * population}")
    print("(Note: Wells removed to isolate consumption)")

    sim.step({}, delta_time=3600.0)

    final_water = faction.get_resource('water')
    consumed = initial_water_total - final_water

    print(f"Final water: {final_water}")
    print(f"Consumed: {consumed}")

    expected_consumption = POPULATION_WATER_CONSUMPTION * population

    if abs(consumed - expected_consumption) < 0.1:
        print("✅ PASS: Water consumption correct\n")
        return True
    else:
        print(f"❌ FAIL: Expected {expected_consumption}, got {consumed}\n")
        return False


def test_population_capacity_from_houses():
    """Test that houses increase population capacity"""
    print("=" * 60)
    print("TEST 8: Population Capacity from Houses")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Calculate initial capacity
    faction.calculate_population_capacity()
    initial_capacity = faction.population_capacity

    print(f"Initial capacity: {initial_capacity}")
    print(f"Initial houses: {faction.get_building_count('house')}")

    faction.add_building('house')
    faction.calculate_population_capacity()

    final_capacity = faction.population_capacity

    print(f"Final capacity: {final_capacity}")
    print(f"Final houses: {faction.get_building_count('house')}")

    if final_capacity > initial_capacity:
        increase = final_capacity - initial_capacity
        print(f"Capacity increased by: {increase}")
        print("✅ PASS: Houses increase population capacity\n")
        return True
    else:
        print("❌ FAIL: Houses did not increase capacity\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("POPULATION SYSTEM TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("Population Growth with Food", test_population_growth_with_food()))
    results.append(("No Growth Without Food", test_population_no_growth_without_food()))
    results.append(("No Growth at Capacity", test_population_no_growth_at_capacity()))
    results.append(("Death from Starvation", test_population_death_from_starvation()))
    results.append(("Food Consumption", test_food_consumption()))
    results.append(("Bread Preferred Over Grain", test_bread_preferred_over_grain()))
    results.append(("Water Consumption", test_water_consumption()))
    results.append(("Capacity from Houses", test_population_capacity_from_houses()))

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
