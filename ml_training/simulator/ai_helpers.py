"""
AI Helper classes for improved learning.
Provides action masking and state enhancement.
"""

from typing import List, Dict
from .game_state import Faction, GameState
from .actions import ActionType
from .config import BUILDING_COSTS, UNIT_TRAINING_COST


class ActionMasker:
    """
    Filters actions to only valid ones based on current game state.
    This dramatically improves learning by preventing wasted exploration.
    """

    def get_valid_actions(self, faction: Faction) -> List[ActionType]:
        valid = []

        economic_buildings = {
            'lumberyard': ActionType.BUILD_LUMBERYARD,
            'quarry': ActionType.BUILD_QUARRY,
            'clay_pit': ActionType.BUILD_CLAY_PIT,
            'mine': ActionType.BUILD_MINE,
            'well': ActionType.BUILD_WELL,
            'farm': ActionType.BUILD_FARM,
            'mill': ActionType.BUILD_MILL,
            'bakery': ActionType.BUILD_BAKERY,
            'hunting_shed': ActionType.BUILD_HUNTING_SHED,
            'cattle_shed': ActionType.BUILD_CATTLE_SHED,
            'house': ActionType.BUILD_HOUSE,
            'pottery': ActionType.BUILD_POTTERY,
            'carpentry': ActionType.BUILD_CARPENTRY,
            'stone_cutter': ActionType.BUILD_STONE_CUTTER,
            'blacksmith': ActionType.BUILD_BLACKSMITH,
        }

        for building_name, action_type in economic_buildings.items():
            if building_name in BUILDING_COSTS:
                if faction.can_afford(BUILDING_COSTS[building_name]):
                    valid.append(action_type)

        if 'barracks' in BUILDING_COSTS:
            if faction.can_afford(BUILDING_COSTS['barracks']):
                valid.append(ActionType.BUILD_BARRACKS)

        if faction.get_building_count('barracks') > 0:
            if 'soldier' in UNIT_TRAINING_COST:
                if faction.can_afford(UNIT_TRAINING_COST['soldier']):
                    valid.append(ActionType.TRAIN_SOLDIER)

            if 'archer' in UNIT_TRAINING_COST:
                if faction.can_afford(UNIT_TRAINING_COST['archer']):
                    valid.append(ActionType.TRAIN_ARCHER)

        if faction.military_strength > 0:
            valid.append(ActionType.ATTACK)

        valid.append(ActionType.DO_NOTHING)

        return valid

    def get_action_mask(self, faction: Faction, all_actions: List[ActionType]) -> List[bool]:
        valid_actions = set(self.get_valid_actions(faction))
        return [action in valid_actions for action in all_actions]


class StateEnhancer:
    """
    Adds derived features to the state representation.
    Normalization constants match env_config.json.
    """

    def __init__(self):
        self.MAX_RESOURCE = 500.0
        self.MAX_POPULATION = 50.0
        self.MAX_BUILDINGS = 10.0
        self.MAX_UNITS = 50.0
        self.MAX_TIME = 1800.0

    def enhance_state(self, faction: Faction, opponent: Faction, game_time: float) -> Dict:
        state = {}

        state['game_time'] = min(1.0, game_time / self.MAX_TIME)

        # Resources
        state['wood'] = min(1.0, faction.get_resource('wood') / self.MAX_RESOURCE)
        state['stone'] = min(1.0, faction.get_resource('stone') / self.MAX_RESOURCE)
        state['grain'] = min(1.0, faction.get_resource('grain') / self.MAX_RESOURCE)
        state['bread'] = min(1.0, faction.get_resource('bread') / self.MAX_RESOURCE)
        state['water'] = min(1.0, faction.get_resource('water') / self.MAX_RESOURCE)

        # Population
        state['population'] = min(1.0, faction.population / self.MAX_POPULATION)
        state['population_capacity'] = min(1.0, faction.population_capacity / self.MAX_POPULATION)
        state['population_ratio'] = faction.population / max(1, faction.population_capacity)

        # Military
        state['military_strength'] = min(1.0, faction.military_strength / self.MAX_UNITS)
        state['soldier_count'] = min(1.0, faction.get_unit_count('soldier') / self.MAX_UNITS)
        state['archer_count'] = min(1.0, faction.get_unit_count('archer') / self.MAX_UNITS)

        # Buildings
        state['farm_count'] = min(1.0, faction.get_building_count('farm') / self.MAX_BUILDINGS)
        state['house_count'] = min(1.0, faction.get_building_count('house') / self.MAX_BUILDINGS)
        state['barracks_count'] = min(1.0, faction.get_building_count('barracks') / self.MAX_BUILDINGS)

        # Affordability signals
        state['can_afford_farm'] = 1.0 if self._can_afford(faction, 'farm') else 0.0
        state['can_afford_house'] = 1.0 if self._can_afford(faction, 'house') else 0.0
        state['can_afford_barracks'] = 1.0 if self._can_afford(faction, 'barracks') else 0.0
        state['can_afford_soldier'] = 1.0 if self._can_afford_unit(faction, 'soldier') else 0.0
        state['can_afford_archer'] = 1.0 if self._can_afford_unit(faction, 'archer') else 0.0

        # Prerequisite signals
        state['has_barracks'] = 1.0 if faction.get_building_count('barracks') > 0 else 0.0
        state['has_farm'] = 1.0 if faction.get_building_count('farm') > 0 else 0.0
        state['has_house'] = 1.0 if faction.get_building_count('house') > 0 else 0.0
        state['can_attack'] = 1.0 if faction.military_strength > 0 else 0.0

        # Economic indicators
        total_food = faction.get_resource('grain') + faction.get_resource('bread')
        food_per_capita = total_food / max(1, faction.population)
        state['food_per_capita'] = min(1.0, food_per_capita / 10.0)
        state['has_surplus_food'] = 1.0 if food_per_capita >= 2.0 else 0.0

        from .config import MIN_FOOD_PER_CAPITA_FOR_GROWTH
        growth_possible = (
            food_per_capita >= MIN_FOOD_PER_CAPITA_FOR_GROWTH and
            faction.population < faction.population_capacity
        )
        state['can_grow_population'] = 1.0 if growth_possible else 0.0

        # Progress information
        state['buildings_in_progress'] = min(1.0, len(faction.buildings_in_progress) / 5.0)
        state['units_in_training'] = min(1.0, len(faction.units_in_training) / 5.0)

        # Opponent state
        if opponent:
            state['opponent_population'] = min(1.0, opponent.population / self.MAX_POPULATION)
            state['opponent_military'] = min(1.0, opponent.military_strength / self.MAX_UNITS)
            state['opponent_barracks'] = 1.0 if opponent.get_building_count('barracks') > 0 else 0.0

            my_strength = faction.military_strength
            their_strength = max(0.1, opponent.military_strength)
            state['military_advantage'] = min(1.0, max(-1.0, (my_strength - their_strength) / their_strength))

            state['population_advantage'] = min(1.0, max(-1.0,
                (faction.population - opponent.population) / max(1, opponent.population)))

        return state

    def _can_afford(self, faction: Faction, building_type: str) -> bool:
        if building_type not in BUILDING_COSTS:
            return False
        return faction.can_afford(BUILDING_COSTS[building_type])

    def _can_afford_unit(self, faction: Faction, unit_type: str) -> bool:
        if unit_type not in UNIT_TRAINING_COST:
            return False
        if faction.get_building_count('barracks') == 0:
            return False
        return faction.can_afford(UNIT_TRAINING_COST[unit_type])
