"""
Test Actual Gameplay Progression
Verify that the game actually progresses with buildings, resources, and strategy
"""

import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_training_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, ml_training_dir)

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType
from simulator.opponents import get_opponent


def test_basic_gameplay_progression():
    """Test that a basic game shows actual progression"""
    print("\n" + "="*70)
    print("TESTING BASIC GAMEPLAY PROGRESSION")
    print("="*70)

    sim = RealTimeRTSSimulator(num_factions=2, max_game_time=600.0)
    faction_0 = sim.state.factions[0]
    faction_1 = sim.state.factions[1]

    print(f"\nInitial State:")
    print(f"  Faction 0: Pop={faction_0.population}, Buildings={sum(faction_0.buildings.values())}, "
          f"Resources: wood={faction_0.resources['wood']}, stone={faction_0.resources['stone']}")
    print(f"  Faction 1: Pop={faction_1.population}, Buildings={sum(faction_1.buildings.values())}, "
          f"Resources: wood={faction_1.resources['wood']}, stone={faction_1.resources['stone']}")

    # Test 1: Can faction 0 build a farm?
    print(f"\n--- Test 1: Building a Farm ---")
    initial_buildings = faction_0.get_building_count('farm')
    initial_wood = faction_0.resources['wood']
    initial_stone = faction_0.resources['stone']

    actions = {
        0: Action(faction_id=0, action_type=ActionType.BUILD_FARM),
        1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
    }

    state, rewards, done = sim.step(actions, delta_time=60.0)

    print(f"  Initial wood: {initial_wood}, stone: {initial_stone}")
    print(f"  After action wood: {faction_0.resources['wood']}, stone: {faction_0.resources['stone']}")
    print(f"  Buildings in progress: {len(faction_0.buildings_in_progress)}")
    print(f"  Reward: {rewards.get(0, 0.0)}")

    if len(faction_0.buildings_in_progress) > 0:
        print(f"  ✅ Farm construction started!")
        print(f"  Time remaining: {faction_0.buildings_in_progress[0].turns_remaining}s")
    else:
        print(f"  ❌ Farm construction did NOT start!")
        print(f"  This might indicate a resource or cost issue")

        # Check if faction could afford it
        from simulator.config import BUILDING_COSTS
        farm_cost = BUILDING_COSTS.get('farm', {})
        print(f"  Farm cost: {farm_cost}")
        print(f"  Can afford: {faction_0.can_afford(farm_cost)}")

    # Test 2: Wait for farm to complete
    print(f"\n--- Test 2: Waiting for Farm to Complete ---")

    do_nothing = {
        0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
        1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
    }

    # Wait 3 minutes (farm takes 2 minutes)
    sim.step(do_nothing, delta_time=180.0)

    completed_farms = faction_0.get_building_count('farm')
    print(f"  Farms before: {initial_buildings}")
    print(f"  Farms after: {completed_farms}")
    print(f"  Buildings in progress: {len(faction_0.buildings_in_progress)}")

    if completed_farms > initial_buildings:
        print(f"  ✅ Farm completed successfully!")
    else:
        print(f"  ❌ Farm did NOT complete!")

    # Test 3: Check resource production over time
    print(f"\n--- Test 3: Resource Production Over Time ---")

    initial_grain = faction_0.resources.get('grain', 0)

    # Run for 5 minutes
    for i in range(5):
        sim.step(do_nothing, delta_time=60.0)

    final_grain = faction_0.resources.get('grain', 0)

    print(f"  Initial grain: {initial_grain}")
    print(f"  Final grain (after 5 min): {final_grain}")
    print(f"  Grain produced: {final_grain - initial_grain}")

    if final_grain > initial_grain:
        print(f"  ✅ Resources are being produced!")
    else:
        print(f"  ⚠️  No grain production detected")
        print(f"  Note: This might be expected if workers are not assigned")

    # Test 4: Population growth
    print(f"\n--- Test 4: Population Growth ---")

    initial_pop = faction_0.population

    # Run for 10 more minutes
    for i in range(10):
        sim.step(do_nothing, delta_time=60.0)

    final_pop = faction_0.population

    print(f"  Initial population: {initial_pop}")
    print(f"  Final population: {final_pop}")
    print(f"  Population change: {final_pop - initial_pop}")

    if final_pop > initial_pop:
        print(f"  ✅ Population is growing!")
    elif final_pop < initial_pop:
        print(f"  ⚠️  Population is declining!")
    else:
        print(f"  ℹ️  Population is stable")

    # Test 5: Scripted opponent behavior
    print(f"\n--- Test 5: Scripted Opponent Behavior ---")

    sim2 = RealTimeRTSSimulator(num_factions=2, max_game_time=600.0)
    opponent = get_opponent('normal', faction_id=1)

    print(f"  Testing opponent: {opponent.name}")

    actions_taken = []
    for i in range(10):
        opp_action = opponent.act(sim2.state, sim2.game_time)
        actions_taken.append(opp_action.action_type.name)

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: opp_action
        }
        sim2.step(actions, delta_time=60.0)

    print(f"  Opponent actions: {set(actions_taken)}")
    print(f"  Opponent buildings: {sum(sim2.state.factions[1].buildings.values())}")

    if ActionType.DO_NOTHING.name not in set(actions_taken) or len(set(actions_taken)) > 1:
        print(f"  ✅ Opponent is taking varied actions!")
    else:
        print(f"  ❌ Opponent only doing nothing!")

    print("\n" + "="*70)
    print("GAMEPLAY PROGRESSION TEST COMPLETE")
    print("="*70)


def test_model_action_distribution():
    """Test what actions the trained model is actually choosing"""
    print("\n" + "="*70)
    print("TESTING MODEL ACTION DISTRIBUTION")
    print("="*70)

    try:
        from stable_baselines3 import PPO
        from config.config_loader import load_env_config
        import numpy as np

        # Import custom feature extractor (required for loading model)
        training_dir = os.path.join(ml_training_dir, 'training')
        sys.path.insert(0, training_dir)
        import transformer_extractor

        # Load model
        model_path = "ppo_model/rts_realtime_final"
        print(f"\nLoading model from: {model_path}")
        model = PPO.load(model_path)

        # Load env config
        env_config = load_env_config()
        feature_keys = env_config['environment']['feature_keys']
        max_game_time = env_config['environment']['max_game_time']

        # Create a simple test state
        sim = RealTimeRTSSimulator(num_factions=2)
        faction = sim.state.factions[0]
        opponent = sim.state.factions[1]

        # Build observation
        obs_dict = {
            'time_ratio': min(sim.game_time / max_game_time, 1.0),
            'wood_ratio': min(faction.get_resource('wood') / 1000, 1.0),
            'stone_ratio': min(faction.get_resource('stone') / 1000, 1.0),
            'clay_ratio': min(faction.get_resource('clay') / 1000, 1.0),
            'grain_ratio': min(faction.get_resource('grain') / 1000, 1.0),
            'water_ratio': min(faction.get_resource('water') / 1000, 1.0),
            'food_ratio': min(faction.get_resource('food') / 1000, 1.0),
            'population_ratio': min(faction.population / 100, 1.0),
            'population_capacity_ratio': min(faction.population_capacity / 100, 1.0),
            'farm_count': min(faction.get_building_count('farm') / 10, 1.0),
            'lumberyard_count': min(faction.get_building_count('lumberyard') / 10, 1.0),
            'quarry_count': min(faction.get_building_count('quarry') / 10, 1.0),
            'house_count': min(faction.get_building_count('house') / 20, 1.0),
            'barracks_count': min(faction.get_building_count('barracks') / 5, 1.0),
            'warehouse_count': min(faction.get_building_count('warehouse') / 5, 1.0),
            'mill_count': min(faction.get_building_count('mill') / 5, 1.0),
            'bakery_count': min(faction.get_building_count('bakery') / 5, 1.0),
            'soldier_count': min(faction.get_unit_count('soldier') / 50, 1.0),
            'military_strength': min(faction.military_strength / 500, 1.0),
            'opponent_military_strength': min(opponent.military_strength / 500, 1.0),
            'opponent_warehouse_count': min(opponent.get_building_count('warehouse') / 5, 1.0),
            'opponent_population': min(opponent.population / 100, 1.0),
        }

        observation = np.array([obs_dict[key] for key in feature_keys], dtype=np.float32)

        print(f"\nTest observation shape: {observation.shape}")
        print(f"Test observation sample: {observation[:5]}")

        # Get action from model (sample multiple times)
        print(f"\nSampling 20 actions from model:")
        from simulator.actions import index_to_action

        action_counts = {}
        for i in range(20):
            action_idx, _states = model.predict(observation, deterministic=False)
            action_type = index_to_action(int(action_idx))
            action_name = action_type.name
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        print(f"\nAction distribution (20 samples):")
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {action}: {count} ({count/20*100:.1f}%)")

        if len(action_counts) == 1 and 'DO_NOTHING' in action_counts:
            print(f"\n  ❌ MODEL ONLY PREDICTS DO_NOTHING!")
            print(f"  This indicates a training or inference issue")
        elif action_counts.get('DO_NOTHING', 0) > 15:
            print(f"\n  ⚠️  Model heavily biased toward DO_NOTHING")
        else:
            print(f"\n  ✅ Model shows action diversity")

    except Exception as e:
        print(f"\n❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_basic_gameplay_progression()
    test_model_action_distribution()
