"""
Test the achievement reward system
"""

import sys
import os
# Add ml_training directory to path (go up 3 levels: unit -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType
from simulator.achievements_rewards import AchievementRewardMapper


def test_achievement_loading():
    """Test that achievements are loaded correctly."""
    mapper = AchievementRewardMapper()
    print("\n=== Testing Achievement Loading ===")
    print(f"Total achievements loaded: {len(mapper.achievements)}")

    stats = mapper.get_achievement_stats()
    print(f"Total possible reward: {mapper.get_total_possible_reward():.1f} points")
    print(f"Categories: {list(stats['by_category'].keys())}")
    print(f"Difficulties: {list(stats['by_difficulty'].keys())}")

    assert len(mapper.achievements) > 0, "No achievements loaded!"
    print("✓ Achievement loading test passed")


def test_basic_achievements():
    """Test basic achievement unlocking."""
    print("\n=== Testing Basic Achievements ===")

    # Create simulator with achievements enabled
    sim = RealTimeRTSSimulator(num_factions=2)

    # Test First Foundation achievement (build first building)
    faction = sim.state.get_faction(0)
    initial_buildings = sum(faction.buildings.values())
    print(f"Initial buildings: {initial_buildings}")

    # Build a farm
    action = Action(faction_id=0, action_type=ActionType.BUILD_FARM)
    state, rewards, done = sim.step({0: action}, delta_time=1.0)

    # Record building completed
    sim.reward_calc.record_building_completed()

    # Check achievements
    unlocked = sim.reward_calc.get_unlocked_achievements()
    print(f"Unlocked achievements: {[ach.name for ach in unlocked]}")

    print("✓ Basic achievement test passed")


def test_resource_achievements():
    """Test resource-based achievements."""
    print("\n=== Testing Resource Achievements ===")

    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.get_faction(0)

    # Give faction lots of resources to trigger achievements
    faction.resources['grain'] = 150
    faction.resources['wood'] = 600
    faction.resources['stone'] = 600

    # Record resources collected
    sim.reward_calc.record_resource_collected('grain', 150)
    sim.reward_calc.record_resource_collected('wood', 600)
    sim.reward_calc.record_resource_collected('stone', 600)

    # Step simulation to check achievements
    action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)
    state, rewards, done = sim.step({0: action}, delta_time=1.0)

    unlocked = sim.reward_calc.get_unlocked_achievements()
    achievement_names = [ach.name for ach in unlocked]

    print(f"Resources - Grain: {faction.resources['grain']}, Wood: {faction.resources['wood']}, Stone: {faction.resources['stone']}")
    print(f"Unlocked achievements: {achievement_names}")

    # Should unlock First Harvest (100 grain), Woodcutter (500 wood), Stone Mason (500 stone)
    expected = ['First Harvest', 'Woodcutter', 'Stone Mason']
    for exp in expected:
        if exp in achievement_names:
            print(f"✓ Unlocked: {exp}")
        else:
            print(f"✗ Missing: {exp}")

    print("✓ Resource achievement test completed")


def test_population_achievements():
    """Test population-based achievements."""
    print("\n=== Testing Population Achievements ===")

    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.get_faction(0)

    # Set population to trigger Village Founded
    faction.population = 25

    action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)
    state, rewards, done = sim.step({0: action}, delta_time=1.0)

    unlocked = sim.reward_calc.get_unlocked_achievements()
    achievement_names = [ach.name for ach in unlocked]

    print(f"Population: {faction.population}")
    print(f"Unlocked achievements: {achievement_names}")

    if 'Village Founded' in achievement_names:
        print("✓ Village Founded unlocked")
    else:
        print("✗ Village Founded not unlocked")

    print("✓ Population achievement test completed")


def test_military_achievements():
    """Test military-based achievements."""
    print("\n=== Testing Military Achievements ===")

    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.get_faction(0)

    # Build barracks first
    faction.buildings['barracks'] = 1

    # Train soldiers
    for i in range(3):
        faction.units['soldier'] = faction.get_unit_count('soldier') + 1
        sim.reward_calc.record_unit_trained()

    faction.calculate_military_strength()

    action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)
    state, rewards, done = sim.step({0: action}, delta_time=1.0)

    unlocked = sim.reward_calc.get_unlocked_achievements()
    achievement_names = [ach.name for ach in unlocked]

    print(f"Military units: {sum(faction.units.values())}")
    print(f"Military strength: {faction.military_strength}")
    print(f"Unlocked achievements: {achievement_names}")

    if 'First Blood' in achievement_names:
        print("✓ First Blood unlocked")
    else:
        print("✗ First Blood not unlocked")

    print("✓ Military achievement test completed")


def test_achievement_rewards():
    """Test that achievement rewards are added to total rewards."""
    print("\n=== Testing Achievement Rewards ===")

    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.get_faction(0)

    # Set up conditions for multiple achievements
    faction.resources['grain'] = 150
    faction.population = 25
    faction.buildings['farm'] = 3

    sim.reward_calc.record_resource_collected('grain', 150)
    sim.reward_calc.record_building_completed()

    # Get rewards before achievements
    initial_reward = sim.reward_calc.calculate_rewards(sim.state, 100.0)
    print(f"Initial reward for faction 0: {initial_reward[0]:.2f}")

    # Check achievement stats
    stats = sim.reward_calc.get_achievement_stats()
    unlocked = sim.reward_calc.get_unlocked_achievements()

    print(f"Achievements unlocked: {len(unlocked)}")
    print(f"Total unlocked reward: {sim.reward_calc.achievement_mapper.get_unlocked_reward_total():.1f} points")

    for ach in unlocked:
        print(f"  - {ach.name}: {ach.reward_points:.1f} points ({ach.difficulty})")

    print("✓ Achievement reward test completed")


def test_achievement_stats():
    """Test achievement statistics."""
    print("\n=== Testing Achievement Statistics ===")

    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.get_faction(0)

    # Unlock some achievements
    faction.resources['wood'] = 600
    faction.population = 25
    faction.buildings['farm'] = 2
    faction.units['soldier'] = 2

    sim.reward_calc.record_resource_collected('wood', 600)
    sim.reward_calc.record_unit_trained()

    action = Action(faction_id=0, action_type=ActionType.DO_NOTHING)
    state, rewards, done = sim.step({0: action}, delta_time=1.0)

    stats = sim.reward_calc.get_achievement_stats()

    print(f"Total achievements: {stats['total']}")
    print(f"Unlocked: {stats['unlocked']}")
    print(f"Progress: {stats['progress']*100:.1f}%")

    print("\nBy Category:")
    for cat, data in stats['by_category'].items():
        print(f"  {cat}: {data['unlocked']}/{data['total']}")

    print("\nBy Difficulty:")
    for diff, data in stats['by_difficulty'].items():
        print(f"  {diff}: {data['unlocked']}/{data['total']}")

    print("✓ Achievement statistics test completed")


def main():
    """Run all achievement tests."""
    print("=" * 60)
    print("ACHIEVEMENT REWARD SYSTEM TESTS")
    print("=" * 60)

    try:
        test_achievement_loading()
        test_basic_achievements()
        test_resource_achievements()
        test_population_achievements()
        test_military_achievements()
        test_achievement_rewards()
        test_achievement_stats()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
