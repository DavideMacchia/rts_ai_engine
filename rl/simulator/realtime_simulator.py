"""
Real-Time RTS Simulator - Refactored with Manager Pattern
Continuous time-based simulation for real-time strategy games
"""

import numpy as np
from typing import Dict, Tuple, List
from .game_state import GameState, Faction, MarchingArmy
from .actions import Action, ActionType
from .managers import (
    PopulationManager,
    ResourceManager,
    CombatManager,
    BuildingManager,
    TrainingManager,
    RewardCalculator
)
from .ai_helpers import ActionMasker, StateEnhancer
from .managers.combat_manager import army_strength
from .config import ATTACK_TRAVEL_TIME_SECONDS, MARCH_SECONDS_PER_TILE, SETTLEMENT_SEPARATION
from .map import GameMap, generate_map, find_settlement_sites, NoViableSite, best_settlement_site
from .policies import POLICY_ACTIONS
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.reward_loader import get_reward_config


class RealTimeRTSSimulator:
    """Real-time RTS simulator with continuous time - refactored with managers."""

    def __init__(self, num_factions: int = 2, config_path: str = None, max_game_time: float = None,
                 with_map: bool = True):
        self.state = GameState()
        self.game_time = 0.0
        self.max_game_time = max_game_time

        # Load reward config directly (no scaling - config values are authoritative)
        self.reward_config = get_reward_config(config_path)

        # Initialize AI helpers
        self.action_masker = ActionMasker()
        self.state_enhancer = StateEnhancer()

        # Initialize managers
        self.population_mgr = PopulationManager()
        self.resource_mgr = ResourceManager()
        self.combat_mgr = CombatManager()
        self.building_mgr = BuildingManager()
        self.training_mgr = TrainingManager()
        self.reward_calc = RewardCalculator(self.reward_config)

        # Create factions
        for i in range(num_factions):
            faction = Faction(id=i)
            self.state.factions.append(faction)

        if with_map:
            self._found_the_world()

    def _found_the_world(self):
        """Generate the map and settle every faction on it (D27).

        Each faction's capital is founded on a real tile, far enough from its rivals that
        the march between them is a journey. From here on, "how far away is the enemy" is a
        fact about the world rather than a constant in a config file.
        """
        # A fair map is a SYMMETRIC map, and symmetry is only defined against a rival. A SOLO
        # district (the civil tier, D32 — it has no war to be fair about) still gets a world:
        # it needs ground to build on and veins to find. It simply gets the best site on it.
        if len(self.state.factions) == 1:
            self.state.map = generate_map(symmetric=False)
            site = best_settlement_site(self.state.map)
            for district in self.state.factions[0].districts:
                district.center = site
                district._map = self.state.map
                self.state.map.place(0, 'warehouse', *site)
                district.building_tiles[site] = 'warehouse'
            return

        if len(self.state.factions) != 2:
            return

        # Keep drawing worlds until one can actually seat a district that could DEVELOP.
        # A map is cheap; a game decided by its founding plot is not (D31).
        for _attempt in range(40):
            self.state.map = generate_map()
            try:
                sites = find_settlement_sites(self.state.map, len(self.state.factions))
                break
            except NoViableSite:
                continue
        else:
            raise NoViableSite(
                "no viable world in 40 draws — PLOT_REQUIREMENTS are stricter than the "
                "map generator can satisfy"
            )
        for faction, site in zip(self.state.factions, sites):
            for district in faction.districts:
                district.center = site
                district._map = self.state.map
                # the warehouse the district is built around IS its centre
                self.state.map.place(faction.id, 'warehouse', *site)
                district.building_tiles[site] = 'warehouse'

    def march_time(self, attacker: Faction, defender: Faction) -> float:
        """How long the army walks. On a map this is the distance it actually covers; with
        no map it falls back to the flat constant the game used before there was a world."""
        if (self.state.map is None or not attacker.districts or not defender.districts
                or attacker.capital.center is None or defender.capital.center is None):
            return ATTACK_TRAVEL_TIME_SECONDS
        distance = GameMap.distance(attacker.capital.center, defender.capital.center)
        return distance * MARCH_SECONDS_PER_TILE

    def reset(self) -> GameState:
        """Reset game with randomized starting conditions."""
        max_time = self.max_game_time
        with_map = self.state.map is not None
        self.__init__(len(self.state.factions), max_game_time=max_time, with_map=with_map)

        for faction in self.state.factions:
            for district in faction.districts:
                district.resources['wood'] += np.random.randint(-20, 20)
                district.resources['stone'] += np.random.randint(-10, 10)
                district.resources['grain'] += np.random.randint(-5, 5)

                for resource in district.resources:
                    district.resources[resource] = max(0, district.resources[resource])

        self.game_time = 0.0
        return self.state

    def step(self, actions: Dict[int, Action], delta_time: float = 1.0) -> Tuple[GameState, Dict[int, float], bool]:
        """
        Advance simulation by delta_time seconds.

        Returns:
            (state, rewards, done)
        """
        # Execute actions and collect immediate feedback
        action_rewards = {}
        for faction_id, action in actions.items():
            action_rewards[faction_id] = self._execute_action(action)

        # Advance time
        self.game_time += delta_time

        # Advance the armies on the road. One arrives either at the enemy (it fights) or
        # back home (it rejoins the garrison it left).
        if self.state.marching_armies:
            still_marching = []
            for army in self.state.marching_armies:
                if self.game_time < army.arrival_time:
                    still_marching.append(army)
                    continue

                att = self.state.get_faction(army.attacker_id)
                dfd = self.state.get_faction(army.defender_id)

                if army.returning:
                    self._bring_army_home(att, army)
                    continue

                if att is None or dfd is None or army_strength(army.units) <= 0:
                    continue

                self.combat_mgr.execute_attack(att, dfd, attacking_units=army.units)

                # The survivors turn around and walk back. They are still not home, and
                # still cannot defend it, for one more march.
                if army_strength(army.units) > 0 and not att.is_defeated():
                    still_marching.append(MarchingArmy(
                        attacker_id=army.attacker_id,
                        defender_id=army.defender_id,
                        arrival_time=self.game_time + self.march_time(att, dfd),
                        units=army.units,
                        returning=True,
                    ))
            self.state.marching_armies = still_marching

        # Update systems using managers
        for faction in self.state.factions:
            for district in faction.districts:
                district.assign_work(delta_time)
            self.building_mgr.progress_buildings(faction, delta_time)
            self.training_mgr.progress_training(faction, delta_time)
            self.training_mgr.apply_conscription(faction, delta_time)
            self.resource_mgr.produce_resources(faction, delta_time)
            self.resource_mgr.consume_resources(faction, delta_time)
            self.population_mgr.update_population(faction, delta_time)

        # Update derived stats
        for faction in self.state.factions:
            faction.calculate_military_strength()
            faction.calculate_population_capacity()

        # Victory is conquest only. Running out of time ends the EPISODE, not the game:
        # the environment reports it as a truncation. See game_state.check_game_over().
        self.state.check_game_over()

        # Calculate milestone rewards
        milestone_rewards = self.reward_calc.calculate_rewards(self.state, self.game_time)

        # Combine all rewards
        total_rewards = {}
        for faction_id in actions.keys():
            action_r = action_rewards.get(faction_id, 0.0)
            milestone_r = milestone_rewards.get(faction_id, 0.0)
            total_rewards[faction_id] = action_r + milestone_r

        return self.state, total_rewards, self.state.game_over

    def _bring_army_home(self, faction, army: MarchingArmy):
        """The survivors rejoin the garrison. If there is no home left, there is no army."""
        if faction is None or not faction.districts:
            return
        district = faction.capital
        for unit_type, count in army.units.items():
            if count > 0:
                district.add_unit(unit_type, count)

    def _send_army(self, faction) -> Dict[str, int]:
        """Take the garrison out of the districts. What marches is no longer home."""
        marching: Dict[str, int] = {}
        for district in faction.districts:
            for unit_type, count in list(district.units.items()):
                if count > 0:
                    marching[unit_type] = marching.get(unit_type, 0) + count
                    district.units[unit_type] = 0
        return marching

    def is_out_of_time(self) -> bool:
        """The episode's time limit. Not a defeat, not a victory — a truncation."""
        return self.max_game_time is not None and self.game_time >= self.max_game_time

    def _execute_action(self, action: Action) -> float:
        """Execute a single action and return an immediate reward/penalty."""
        faction = self.state.get_faction(action.faction_id)
        if faction is None:
            return self.reward_config.get('action_validity', 'invalid_faction')

        action_building_map = {
            ActionType.BUILD_LUMBERYARD: 'lumberyard',
            ActionType.BUILD_QUARRY: 'quarry',
            ActionType.BUILD_CLAY_PIT: 'clay_pit',
            ActionType.BUILD_MINE: 'mine',
            ActionType.BUILD_WELL: 'well',
            ActionType.BUILD_FARM: 'farm',
            ActionType.BUILD_MILL: 'mill',
            ActionType.BUILD_BAKERY: 'bakery',
            ActionType.BUILD_HUNTING_SHED: 'hunting_shed',
            ActionType.BUILD_CATTLE_SHED: 'cattle_shed',
            ActionType.BUILD_VEGETABLE_GARDEN: 'vegetable_garden',
            ActionType.BUILD_ORCHARD: 'orchard',
            ActionType.BUILD_SCHOOL: 'school',
            ActionType.BUILD_HOUSE: 'house',
            ActionType.BUILD_POTTERY: 'pottery',
            ActionType.BUILD_CARPENTRY: 'carpentry',
            ActionType.BUILD_STONE_CUTTER: 'stone_cutter',
            ActionType.BUILD_BLACKSMITH: 'blacksmith',
            ActionType.BUILD_SMELTER: 'smelter',
            ActionType.BUILD_TAILOR: 'tailor',
            ActionType.BUILD_DORMITORY: 'dormitory',
            ActionType.BUILD_WATCHTOWER: 'watchtower',
            ActionType.BUILD_DEFENSIVE_WALL_WOOD: 'defensive_wall_wood',
            ActionType.BUILD_DEFENSIVE_WALL_STONE: 'defensive_wall_stone',
            ActionType.BUILD_BARRACKS: 'barracks',
        }

        action_unit_map = {
            ActionType.TRAIN_SOLDIER: 'soldier',
            ActionType.TRAIN_MAN_AT_ARMS: 'man_at_arms',
            ActionType.TRAIN_ARCHER: 'archer',
        }

        if action.action_type in POLICY_ACTIONS:
            # Policies are per-district state; with one district that is the capital.
            POLICY_ACTIONS[action.action_type](faction.capital.policies)
            return self.reward_config.get('action_validity', 'valid_action')

        if action.action_type in action_building_map:
            building_type = action_building_map[action.action_type]
            success, message = self.building_mgr.start_building(
                faction, building_type, tile=action.target_tile)
            return self.reward_config.get('action_validity', message)

        elif action.action_type in action_unit_map:
            unit_type = action_unit_map[action.action_type]
            success, message = self.training_mgr.start_training(faction, unit_type)
            return self.reward_config.get('action_validity', message)

        elif action.action_type == ActionType.ATTACK:
            if action.target_faction_id is not None:
                attacker = self.state.get_faction(action.faction_id)
                defender = self.state.get_faction(action.target_faction_id)

                if not attacker or not defender:
                    return self.reward_config.get('action_validity', 'invalid_faction')

                if attacker.military_strength <= 0:
                    return self.reward_config.get('action_validity', 'attack_without_army')

                # An army already on the march cannot be sent again: without this the
                # caller re-orders every step and hundreds of armies arrive in a wave.
                if self.state.is_marching(action.faction_id):
                    return 0.0

                # The army LEAVES: its units come out of the districts and travel with the
                # column, so the attacker is undefended until they come back (D26).
                marching_units = self._send_army(attacker)
                self.state.marching_armies.append(MarchingArmy(
                    attacker_id=action.faction_id,
                    defender_id=action.target_faction_id,
                    arrival_time=self.game_time + self.march_time(attacker, defender),
                    units=marching_units,
                ))
                return 0.0
            else:
                return self.reward_config.get('action_validity', 'attack_without_target')

        elif action.action_type == ActionType.DO_NOTHING:
            return self.reward_config.get('action_validity', 'do_nothing')

        return self.reward_config.get('action_validity', 'invalid_missing_prerequisite')

    def get_state_for_faction(self, faction_id: int, enhanced: bool = True) -> Dict:
        """Get observable state for ML agent (normalized 0-1)."""
        faction = self.state.get_faction(faction_id)
        opponent_id = 1 if faction_id == 0 else 0
        opponent = self.state.get_faction(opponent_id)

        if not faction:
            return {}

        if enhanced:
            return self.state_enhancer.enhance_state(faction, opponent, self.game_time)
        else:
            MAX_RESOURCE = 500.0
            MAX_POPULATION = 50.0
            MAX_BUILDINGS = 10.0
            MAX_UNITS = 50.0

            state = {
                'game_time': min(1.0, self.game_time / 1800.0),
                'wood': min(1.0, faction.get_resource('wood') / MAX_RESOURCE),
                'stone': min(1.0, faction.get_resource('stone') / MAX_RESOURCE),
                'grain': min(1.0, faction.get_resource('grain') / MAX_RESOURCE),
                'water': min(1.0, faction.get_resource('water') / MAX_RESOURCE),
                'population': min(1.0, faction.population / MAX_POPULATION),
                'population_capacity': min(1.0, faction.population_capacity / MAX_POPULATION),
                'military_strength': min(1.0, faction.military_strength / MAX_UNITS),
                'soldier_count': min(1.0, faction.get_unit_count('soldier') / MAX_UNITS),
                'farm_count': min(1.0, faction.get_building_count('farm') / MAX_BUILDINGS),
                'barracks_count': min(1.0, faction.get_building_count('barracks') / MAX_BUILDINGS),
                'house_count': min(1.0, faction.get_building_count('house') / MAX_BUILDINGS),
            }

            if opponent:
                state.update({
                    'opponent_population': min(1.0, opponent.population / MAX_POPULATION),
                    'opponent_military_strength': min(1.0, opponent.military_strength / MAX_UNITS),
                    'opponent_barracks': min(1.0, opponent.get_building_count('barracks') / MAX_BUILDINGS),
                })

            return state

    def get_valid_actions(self, faction_id: int) -> List[ActionType]:
        faction = self.state.get_faction(faction_id)
        if not faction:
            return [ActionType.DO_NOTHING]
        return self.action_masker.get_valid_actions(faction)

    def get_action_mask(self, faction_id: int, all_actions: List[ActionType]) -> List[bool]:
        faction = self.state.get_faction(faction_id)
        if not faction:
            return [action == ActionType.DO_NOTHING for action in all_actions]
        return self.action_masker.get_action_mask(faction, all_actions)
