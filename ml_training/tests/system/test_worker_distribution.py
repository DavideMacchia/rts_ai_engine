"""
Test worker distribution system
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: system -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.config import UNIT_TRAINING_TIME


def test_worker_calculation():
    """Test that available workers = population - military"""
    print("=" * 60)
    print("TEST 1: Worker Calculation")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.population = 20
    faction.add_unit('soldier', 5)

    available = faction.get_available_workers()
    expected = 15  # 20 - 5

    print(f"Population: {faction.population}")
    print(f"Military units: {sum(faction.units.values())}")
    print(f"Available workers: {available}")
    print(f"Expected: {expected}")

    if available == expected:
        print("✅ PASS: Worker calculation correct\n")
        return True
    else:
        print(f"❌ FAIL: Expected {expected}, got {available}\n")
        return False


def test_worker_priority_allocation():
    """Test workers assigned by priority (food > resource > processing)"""
    print("=" * 60)
    print("TEST 2: Worker Priority Allocation")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')      # Priority 1, needs 2 workers
    faction.add_building('lumberyard') # Priority 2, needs 2 workers
    faction.add_building('mill')      # Priority 3, needs 1 worker

    # Only 3 workers available (can't staff all)
    faction.population = 3

    print("Buildings:")
    print(f"  Farm (Priority 1): needs 2 workers")
    print(f"  Lumberyard (Priority 2): needs 2 workers")
    print(f"  Mill (Priority 3): needs 1 worker")
    print(f"Population: {faction.population}")

    assignments = faction.calculate_worker_assignments()

    print(f"\nWorker Assignments:")
    print(f"  Farm: {assignments.get('farm', 0)} workers")
    print(f"  Lumberyard: {assignments.get('lumberyard', 0)} workers")
    print(f"  Mill: {assignments.get('mill', 0)} workers")

    # Farm should get full staffing (2 workers)
    farm_ok = assignments.get('farm', 0) == 2
    # Lumberyard should get 1 worker (partial)
    lumberyard_ok = assignments.get('lumberyard', 0) == 1
    # Mill should get 0 workers
    mill_ok = assignments.get('mill', 0) == 0

    if farm_ok and lumberyard_ok and mill_ok:
        print("✅ PASS: Worker priority allocation correct\n")
        return True
    else:
        print("❌ FAIL: Worker priority allocation incorrect\n")
        return False


def test_production_with_full_workers():
    """Test 100% workers = 100% production"""
    print("=" * 60)
    print("TEST 3: Production With Full Workers")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')
    faction.population = 2  # Farm needs 2 workers

    print(f"Farm count: {faction.get_building_count('farm')}")
    print(f"Population: {faction.population}")

    productivity = faction.get_building_productivity('farm')
    print(f"Productivity: {productivity:.2f} (expected: 1.00)")

    if abs(productivity - 1.0) < 0.01:
        initial_grain = faction.get_resource('grain')
        sim._produce_resources(delta_time=3600.0)
        final_grain = faction.get_resource('grain')

        expected_gain = 3.0  # Farm produces 3 grain/hour at 100%
        actual_gain = final_grain - initial_grain

        print(f"Initial grain: {initial_grain}")
        print(f"Final grain: {final_grain}")
        print(f"Expected gain: {expected_gain}")
        print(f"Actual gain: {actual_gain}")

        if abs(actual_gain - expected_gain) < 0.1:
            print("✅ PASS: 100% workers = 100% production\n")
            return True
        else:
            print(f"❌ FAIL: Production incorrect\n")
            return False
    else:
        print(f"❌ FAIL: Productivity calculation incorrect\n")
        return False


def test_production_with_half_workers():
    """Test 50% workers = 50% production (linear scaling)"""
    print("=" * 60)
    print("TEST 4: Production With Half Workers")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')
    faction.population = 1  # Farm needs 2, only 1 available = 50%

    print(f"Farm count: {faction.get_building_count('farm')}")
    print(f"Population: {faction.population}")
    print(f"Workers needed: 2, Workers available: 1")

    productivity = faction.get_building_productivity('farm')
    print(f"Productivity: {productivity:.2f} (expected: 0.50)")

    if abs(productivity - 0.5) < 0.01:
        initial_grain = faction.get_resource('grain')
        sim._produce_resources(delta_time=3600.0)
        final_grain = faction.get_resource('grain')

        expected_gain = 1.5  # 3.0 * 0.5 = 1.5 grain/hour
        actual_gain = final_grain - initial_grain

        print(f"Initial grain: {initial_grain}")
        print(f"Final grain: {final_grain}")
        print(f"Expected gain: {expected_gain}")
        print(f"Actual gain: {actual_gain}")

        if abs(actual_gain - expected_gain) < 0.1:
            print("✅ PASS: 50% workers = 50% production\n")
            return True
        else:
            print(f"❌ FAIL: Production incorrect\n")
            return False
    else:
        print(f"❌ FAIL: Productivity calculation incorrect\n")
        return False


def test_production_with_no_workers():
    """Test 0% workers = 0% production"""
    print("=" * 60)
    print("TEST 5: Production With No Workers")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')
    faction.population = 0  # No workers

    print(f"Farm count: {faction.get_building_count('farm')}")
    print(f"Population: {faction.population}")

    productivity = faction.get_building_productivity('farm')
    print(f"Productivity: {productivity:.2f} (expected: 0.00)")

    if productivity == 0.0:
        initial_grain = faction.get_resource('grain')
        sim._produce_resources(delta_time=3600.0)
        final_grain = faction.get_resource('grain')

        print(f"Initial grain: {initial_grain}")
        print(f"Final grain: {final_grain}")

        # No production should occur
        if final_grain == initial_grain:
            print("✅ PASS: 0% workers = 0% production\n")
            return True
        else:
            print(f"❌ FAIL: Production occurred without workers\n")
            return False
    else:
        print(f"❌ FAIL: Productivity should be 0.0\n")
        return False


def test_military_consumes_population():
    """Test that training a unit consumes 1 population"""
    print("=" * 60)
    print("TEST 6: Military Consumes Population")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.population = 10
    faction.add_building('barracks')

    print(f"Initial population: {faction.population}")

    faction.add_resource('wood', 100)
    faction.add_resource('grain', 100)
    faction.add_resource('weapons', 10)

    sim._train_unit(faction, 'soldier')

    print(f"Training started: {len(faction.units_in_training)} units in training")
    print(f"Population after starting training: {faction.population}")

    if len(faction.units_in_training) == 1 and faction.population == 10:
        training_time = UNIT_TRAINING_TIME['soldier'] * 3600.0
        sim._progress_training(delta_time=training_time)

        print(f"Training completed")
        print(f"Population after training completion: {faction.population}")
        print(f"Soldier count: {faction.get_unit_count('soldier')}")

        # Population should be consumed
        if faction.population == 9 and faction.get_unit_count('soldier') == 1:
            print("✅ PASS: Military training consumes population\n")
            return True
        else:
            print(f"❌ FAIL: Population not consumed correctly\n")
            return False
    else:
        print(f"❌ FAIL: Training didn't start correctly\n")
        return False


def test_multiple_buildings_same_type():
    """Test worker distribution across multiple buildings of same type"""
    print("=" * 60)
    print("TEST 7: Multiple Buildings Same Type")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # 3 farms, each needs 2 workers = 6 total
    faction.add_building('farm')
    faction.add_building('farm')
    faction.add_building('farm')

    faction.population = 4  # Only 4 workers available

    print(f"Farm count: {faction.get_building_count('farm')}")
    print(f"Total workers needed: 6 (3 farms × 2 workers)")
    print(f"Workers available: {faction.population}")

    assignments = faction.calculate_worker_assignments()

    print(f"Workers assigned to farms: {assignments.get('farm', 0)}")

    # Should assign 4 workers to farms (partial staffing)
    if assignments.get('farm', 0) == 4:
        # Productivity should be 4/6 = 66.67%
        productivity = faction.get_building_productivity('farm')
        expected = 4.0 / 6.0

        print(f"Productivity: {productivity:.2f} (expected: {expected:.2f})")

        if abs(productivity - expected) < 0.01:
            print("✅ PASS: Multiple buildings distribution correct\n")
            return True
        else:
            print(f"❌ FAIL: Productivity incorrect\n")
            return False
    else:
        print(f"❌ FAIL: Worker assignment incorrect\n")
        return False


def test_worker_redistribution_after_building():
    """Test workers automatically redistribute when new building is added"""
    print("=" * 60)
    print("TEST 8: Worker Redistribution After Building")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('farm')  # Priority 1, needs 2
    faction.population = 2

    print(f"Initial state:")
    print(f"  Buildings: 1 farm")
    print(f"  Population: 2")

    # Initially, farm is fully staffed
    assignments1 = faction.calculate_worker_assignments()
    print(f"  Farm workers: {assignments1.get('farm', 0)}")

    if assignments1.get('farm', 0) == 2:
        # Add lumberyard (priority 2, needs 2)
        faction.add_building('lumberyard')

        print(f"\nAfter adding lumberyard:")
        print(f"  Buildings: 1 farm, 1 lumberyard")

        # Workers should redistribute (farm keeps priority)
        assignments2 = faction.calculate_worker_assignments()
        print(f"  Farm workers: {assignments2.get('farm', 0)}")
        print(f"  Lumberyard workers: {assignments2.get('lumberyard', 0)}")

        if assignments2.get('farm', 0) == 2 and assignments2.get('lumberyard', 0) == 0:
            faction.population = 4

            print(f"\nAfter population increase to 4:")

            # Now both should be staffed
            assignments3 = faction.calculate_worker_assignments()
            print(f"  Farm workers: {assignments3.get('farm', 0)}")
            print(f"  Lumberyard workers: {assignments3.get('lumberyard', 0)}")

            if assignments3.get('farm', 0) == 2 and assignments3.get('lumberyard', 0) == 2:
                print("✅ PASS: Worker redistribution works\n")
                return True
            else:
                print(f"❌ FAIL: Final distribution incorrect\n")
                return False
        else:
            print(f"❌ FAIL: Distribution after building incorrect\n")
            return False
    else:
        print(f"❌ FAIL: Initial distribution incorrect\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WORKER DISTRIBUTION SYSTEM TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("Worker Calculation", test_worker_calculation()))
    results.append(("Worker Priority Allocation", test_worker_priority_allocation()))
    results.append(("Production With Full Workers", test_production_with_full_workers()))
    results.append(("Production With Half Workers", test_production_with_half_workers()))
    results.append(("Production With No Workers", test_production_with_no_workers()))
    results.append(("Military Consumes Population", test_military_consumes_population()))
    results.append(("Multiple Buildings Same Type", test_multiple_buildings_same_type()))
    results.append(("Worker Redistribution", test_worker_redistribution_after_building()))

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
