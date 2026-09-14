"""
Test AI improvements: ActionMasker, StateEnhancer, and reward-config invariants
"""
import sys
import os
# Add ml_training directory to path (go up 3 levels: integration -> tests -> ml_training)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType


def test_action_masking_basic():
    """Test that action masking filters to only affordable actions"""
    print("=" * 60)
    print("TEST 1: Action Masking - Basic Filtering")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Faction starts with some resources, can afford some buildings
    print(f"Starting resources:")
    print(f"  Wood: {faction.get_resource('wood')}")
    print(f"  Stone: {faction.get_resource('stone')}")
    print(f"  Grain: {faction.get_resource('grain')}")

    valid_actions = sim.get_valid_actions(faction_id=0)

    print(f"\nValid actions: {len(valid_actions)}")
    for action in valid_actions:
        print(f"  - {action.name}")

    # Should NOT include military actions (no barracks)
    has_train_soldier = ActionType.TRAIN_SOLDIER in valid_actions
    has_attack = ActionType.ATTACK in valid_actions

    # Should include DO_NOTHING (always valid)
    has_do_nothing = ActionType.DO_NOTHING in valid_actions

    # Should only include affordable buildings
    has_some_buildings = any(
        action.name.startswith('BUILD_') for action in valid_actions
    )

    print(f"\nValidation:")
    print(f"  Has TRAIN_SOLDIER: {has_train_soldier} (should be False - no barracks)")
    print(f"  Has ATTACK: {has_attack} (should be False - no military)")
    print(f"  Has DO_NOTHING: {has_do_nothing} (should be True - always valid)")
    print(f"  Has some buildings: {has_some_buildings} (should be True)")

    if not has_train_soldier and not has_attack and has_do_nothing and has_some_buildings:
        print("✅ PASS: Action masking correctly filters actions\n")
        return True
    else:
        print("❌ FAIL: Action masking incorrect\n")
        return False


def test_action_masking_with_barracks():
    """Test that military actions become available with barracks"""
    print("=" * 60)
    print("TEST 2: Action Masking - Military Prerequisites")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Add barracks and ALL resources needed for training
    faction.add_building('barracks')
    faction.add_resource('wood', 100)     # Soldiers need wood
    faction.add_resource('grain', 100)    # Soldiers need grain
    faction.add_resource('weapons', 10)   # Soldiers need weapons (critical!)

    print(f"Barracks count: {faction.get_building_count('barracks')}")
    print(f"Resources:")
    print(f"  Wood: {faction.get_resource('wood')}")
    print(f"  Grain: {faction.get_resource('grain')}")
    print(f"  Weapons: {faction.get_resource('weapons')}")

    valid_actions = sim.get_valid_actions(faction_id=0)

    print(f"\nValid actions: {len(valid_actions)}")

    # Should now include military training
    has_train_soldier = ActionType.TRAIN_SOLDIER in valid_actions
    has_train_archer = ActionType.TRAIN_ARCHER in valid_actions

    # Should NOT include ATTACK (no military yet)
    has_attack = ActionType.ATTACK in valid_actions

    print(f"\nValidation:")
    print(f"  Has TRAIN_SOLDIER: {has_train_soldier} (should be True)")
    print(f"  Has TRAIN_ARCHER: {has_train_archer} (should be True)")
    print(f"  Has ATTACK: {has_attack} (should be False - no units yet)")

    if has_train_soldier and has_train_archer and not has_attack:
        print("✅ PASS: Military actions available with barracks\n")
        return True
    else:
        print("❌ FAIL: Military prerequisites incorrect\n")
        return False


def test_action_masking_with_military():
    """Test that ATTACK becomes available with military units"""
    print("=" * 60)
    print("TEST 3: Action Masking - Attack Available")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_building('barracks')
    faction.units['soldier'] = 5
    faction.calculate_military_strength()

    print(f"Military strength: {faction.military_strength}")

    valid_actions = sim.get_valid_actions(faction_id=0)

    has_attack = ActionType.ATTACK in valid_actions

    print(f"\nValidation:")
    print(f"  Has ATTACK: {has_attack} (should be True)")

    if has_attack:
        print("✅ PASS: ATTACK available with military units\n")
        return True
    else:
        print("❌ FAIL: ATTACK not available despite military\n")
        return False


def test_enhanced_state_affordability():
    """Test that enhanced state includes affordability signals"""
    print("=" * 60)
    print("TEST 4: Enhanced State - Affordability Signals")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    state = sim.get_state_for_faction(faction_id=0, enhanced=True)

    print(f"State features (total: {len(state)}):")

    # Check for new affordability features
    affordability_features = [
        'can_afford_farm',
        'can_afford_house',
        'can_afford_barracks',
        'can_afford_soldier',
        'can_afford_archer',
    ]

    missing = []
    for feature in affordability_features:
        if feature in state:
            print(f"  ✓ {feature}: {state[feature]}")
        else:
            print(f"  ✗ {feature}: MISSING")
            missing.append(feature)

    if not missing:
        print("✅ PASS: All affordability signals present\n")
        return True
    else:
        print(f"❌ FAIL: Missing features: {missing}\n")
        return False


def test_enhanced_state_prerequisites():
    """Test that enhanced state includes prerequisite signals"""
    print("=" * 60)
    print("TEST 5: Enhanced State - Prerequisite Signals")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Initially no special buildings
    state1 = sim.get_state_for_faction(faction_id=0, enhanced=True)

    print("Initial state:")
    print(f"  has_barracks: {state1.get('has_barracks', 'MISSING')}")
    print(f"  has_farm: {state1.get('has_farm', 'MISSING')}")
    print(f"  has_house: {state1.get('has_house', 'MISSING')}")
    print(f"  can_attack: {state1.get('can_attack', 'MISSING')}")

    # Add buildings and military
    faction.add_building('barracks')
    faction.add_building('farm')
    faction.add_building('house')
    faction.units['soldier'] = 3
    faction.calculate_military_strength()

    state2 = sim.get_state_for_faction(faction_id=0, enhanced=True)

    print("\nAfter adding buildings and military:")
    print(f"  has_barracks: {state2.get('has_barracks', 'MISSING')}")
    print(f"  has_farm: {state2.get('has_farm', 'MISSING')}")
    print(f"  has_house: {state2.get('has_house', 'MISSING')}")
    print(f"  can_attack: {state2.get('can_attack', 'MISSING')}")

    prereq_features = ['has_barracks', 'has_farm', 'has_house', 'can_attack']
    all_present = all(feature in state2 for feature in prereq_features)
    values_correct = (
        state2.get('has_barracks') == 1.0 and
        state2.get('has_farm') == 1.0 and
        state2.get('has_house') == 1.0 and
        state2.get('can_attack') == 1.0
    )

    if all_present and values_correct:
        print("✅ PASS: Prerequisite signals correct\n")
        return True
    else:
        print("❌ FAIL: Prerequisite signals incorrect\n")
        return False


def test_enhanced_state_economic_indicators():
    """Test that enhanced state includes economic indicators"""
    print("=" * 60)
    print("TEST 6: Enhanced State - Economic Indicators")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    faction.add_resource('grain', 50)
    faction.population = 10

    state = sim.get_state_for_faction(faction_id=0, enhanced=True)

    economic_features = [
        'food_per_capita',
        'has_surplus_food',
        'can_grow_population',
        'population_ratio',
    ]

    print("Economic indicators:")
    missing = []
    for feature in economic_features:
        if feature in state:
            print(f"  ✓ {feature}: {state[feature]}")
        else:
            print(f"  ✗ {feature}: MISSING")
            missing.append(feature)

    if not missing:
        print("✅ PASS: All economic indicators present\n")
        return True
    else:
        print(f"❌ FAIL: Missing features: {missing}\n")
        return False


def test_enhanced_state_progress_info():
    """Test that enhanced state includes progress information"""
    print("=" * 60)
    print("TEST 7: Enhanced State - Progress Information")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Start some buildings and training
    from simulator.config import BUILDING_COSTS, BUILDING_BUILD_TIME
    from simulator.config import UNIT_TRAINING_COST, UNIT_TRAINING_TIME

    faction.add_resource('wood', 100)
    faction.add_resource('stone', 100)
    faction.add_resource('grain', 100)

    # Start building construction
    faction.buildings_in_progress.append({
        'type': 'farm',
        'completion_time': sim.game_time + BUILDING_BUILD_TIME['farm']
    })

    # Add barracks and start unit training
    faction.add_building('barracks')
    faction.units_in_training.append({
        'type': 'soldier',
        'completion_time': sim.game_time + UNIT_TRAINING_TIME['soldier']
    })

    state = sim.get_state_for_faction(faction_id=0, enhanced=True)

    print("Progress information:")
    print(f"  buildings_in_progress: {state.get('buildings_in_progress', 'MISSING')}")
    print(f"  units_in_training: {state.get('units_in_training', 'MISSING')}")

    has_progress_features = (
        'buildings_in_progress' in state and
        'units_in_training' in state
    )

    # Values should be > 0 since we have things in progress
    values_correct = (
        state.get('buildings_in_progress', 0) > 0 and
        state.get('units_in_training', 0) > 0
    )

    if has_progress_features and values_correct:
        print("✅ PASS: Progress information correct\n")
        return True
    else:
        print("❌ FAIL: Progress information incorrect\n")
        return False


def test_reward_scale_and_delta_keys():
    """Rewards stay on a comparable scale and use the delta-based keys.

    Guards two regressions we actually hit:
    - victory dwarfing continuous rewards (sparse-reward problem)
    - reintroducing absolute-state reward keys instead of delta keys
    """
    print("=" * 60)
    print("TEST 8: Reward config - bounded scale + delta keys")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)
    cfg = sim.reward_config

    victory = cfg.get('victory_defeat', 'victory')
    defeat = cfg.get('victory_defeat', 'defeat')
    print(f"  Victory: {victory}   Defeat: {defeat}")

    # Terminal rewards must stay on the same order of magnitude as shaped rewards
    bounded = abs(victory) <= 200.0 and abs(defeat) <= 200.0

    # Delta-based keys must exist (absolute-state rewards let the agent idle-and-collect)
    delta_keys = (
        cfg.get('economic_progress', 'population_growth', -1) >= 0 and
        cfg.get('economic_progress', 'building_completed', -1) >= 0 and
        cfg.get('military_progress', 'unit_trained', -1) >= 0 and
        cfg.get('military_progress', 'strength_gained', -1) >= 0
    )
    print(f"  Terminal rewards bounded: {bounded}")
    print(f"  Delta reward keys present: {delta_keys}")

    if bounded and delta_keys:
        print("✅ PASS: Reward config sane\n")
        return True
    print("❌ FAIL: Reward config regressed\n")
    return False


def test_action_mask_format():
    """Test that action mask returns correct format for neural networks"""
    print("=" * 60)
    print("TEST 9: Action Mask - Binary Format")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=1)
    faction = sim.state.factions[0]

    # Define all possible actions
    all_actions = list(ActionType)

    mask = sim.get_action_mask(faction_id=0, all_actions=all_actions)

    print(f"Total actions: {len(all_actions)}")
    print(f"Mask length: {len(mask)}")
    print(f"Valid actions (True count): {sum(mask)}")
    print(f"Invalid actions (False count): {len(mask) - sum(mask)}")

    # Mask should be same length as all_actions
    length_correct = len(mask) == len(all_actions)

    # Mask should be list of booleans
    all_booleans = all(isinstance(x, bool) for x in mask)

    # DO_NOTHING should always be valid
    do_nothing_idx = all_actions.index(ActionType.DO_NOTHING)
    do_nothing_valid = mask[do_nothing_idx] == True

    print(f"\nValidation:")
    print(f"  Length matches: {length_correct}")
    print(f"  All booleans: {all_booleans}")
    print(f"  DO_NOTHING valid: {do_nothing_valid}")

    if length_correct and all_booleans and do_nothing_valid:
        print("✅ PASS: Action mask format correct\n")
        return True
    else:
        print("❌ FAIL: Action mask format incorrect\n")
        return False


def test_enhanced_state_opponent_awareness():
    """Test that enhanced state includes opponent information"""
    print("=" * 60)
    print("TEST 10: Enhanced State - Opponent Awareness")
    print("=" * 60)

    sim = RealTimeRTSSimulator(num_factions=2)

    # Set up faction 0 with some military
    faction0 = sim.state.factions[0]
    faction0.units['soldier'] = 10
    faction0.calculate_military_strength()

    # Set up faction 1 with different military
    faction1 = sim.state.factions[1]
    faction1.units['soldier'] = 5
    faction1.calculate_military_strength()

    # Get state from perspective of faction 0
    state = sim.get_state_for_faction(faction_id=0, enhanced=True)

    opponent_features = [
        'opponent_population',
        'opponent_military',
        'military_advantage',
        'population_advantage',
    ]

    print("Opponent awareness features:")
    missing = []
    for feature in opponent_features:
        if feature in state:
            print(f"  ✓ {feature}: {state[feature]}")
        else:
            print(f"  ✗ {feature}: MISSING")
            missing.append(feature)

    # Faction 0 has 10 soldiers vs faction 1's 5, so should have positive military advantage
    has_advantage = state.get('military_advantage', 0) > 0

    print(f"\nValidation:")
    print(f"  All features present: {len(missing) == 0}")
    print(f"  Military advantage > 0: {has_advantage} (faction 0 has more military)")

    if not missing and has_advantage:
        print("✅ PASS: Opponent awareness correct\n")
        return True
    else:
        print(f"❌ FAIL: Opponent awareness incorrect\n")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AI IMPROVEMENTS TESTS")
    print("=" * 60 + "\n")

    results = []

    results.append(("Action Masking - Basic", test_action_masking_basic()))
    results.append(("Action Masking - Barracks", test_action_masking_with_barracks()))
    results.append(("Action Masking - Attack", test_action_masking_with_military()))
    results.append(("Action Mask Format", test_action_mask_format()))

    results.append(("Enhanced State - Affordability", test_enhanced_state_affordability()))
    results.append(("Enhanced State - Prerequisites", test_enhanced_state_prerequisites()))
    results.append(("Enhanced State - Economics", test_enhanced_state_economic_indicators()))
    results.append(("Enhanced State - Progress", test_enhanced_state_progress_info()))
    results.append(("Enhanced State - Opponent", test_enhanced_state_opponent_awareness()))

    # Reward Normalization Tests
    results.append(("Reward Scale & Delta Keys", test_reward_scale_and_delta_keys()))

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
