"""
Available actions matching the game documentation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

class ActionType(Enum):
    """All possible action types - matching your game's buildings."""

    # === RESOURCE BUILDINGS ===
    BUILD_LUMBERYARD = "build_lumberyard"
    BUILD_QUARRY = "build_quarry"
    BUILD_CLAY_PIT = "build_clay_pit"
    BUILD_MINE = "build_mine"
    BUILD_WELL = "build_well"

    # === FOOD PRODUCTION ===
    BUILD_FARM = "build_farm"
    BUILD_MILL = "build_mill"
    BUILD_BAKERY = "build_bakery"
    BUILD_HUNTING_SHED = "build_hunting_shed"
    BUILD_CATTLE_SHED = "build_cattle_shed"
    BUILD_VEGETABLE_GARDEN = "build_vegetable_garden"
    BUILD_ORCHARD = "build_orchard"

    # === HOUSING ===
    BUILD_HOUSE = "build_house"
    BUILD_DORMITORY = "build_dormitory"

    # === EDUCATION (human-capital investment) ===
    BUILD_SCHOOL = "build_school"

    # === WORKSHOPS ===
    BUILD_POTTERY = "build_pottery"
    BUILD_CARPENTRY = "build_carpentry"
    BUILD_STONE_CUTTER = "build_stone_cutter"
    BUILD_BLACKSMITH = "build_blacksmith"
    BUILD_SMELTER = "build_smelter"
    BUILD_TAILOR = "build_tailor"

    # === MILITARY BUILDINGS ===
    BUILD_BARRACKS = "build_barracks"
    BUILD_WATCHTOWER = "build_watchtower"
    BUILD_DEFENSIVE_WALL_WOOD = "build_defensive_wall_wood"
    BUILD_DEFENSIVE_WALL_STONE = "build_defensive_wall_stone"

    # === MILITARY ACTIONS ===
    TRAIN_SOLDIER = "train_soldier"
    TRAIN_MAN_AT_ARMS = "train_man_at_arms"
    TRAIN_ARCHER = "train_archer"
    ATTACK = "attack"

    # === DISTRICT POLICIES (the four levers, see docs/population_and_policies.md) ===
    # Only the district-scoped ones. `block_families` / `cant_leave` belong to the
    # military camp's action space (invariant I4) and there is no camp yet.
    SET_POPULATION_RATE_INCREASE = "set_population_rate_increase"   # L2 incentive
    SET_POPULATION_RATE_MAINTAIN = "set_population_rate_maintain"
    SET_POPULATION_RATE_DECREASE = "set_population_rate_decrease"
    ENABLE_MINIMUM_FOOD = "enable_minimum_food"                     # L1 affordance
    DISABLE_MINIMUM_FOOD = "disable_minimum_food"
    ENABLE_FORCED_CONSCRIPTION = "enable_forced_conscription"       # L4 coercion
    DISABLE_FORCED_CONSCRIPTION = "disable_forced_conscription"

    # === PASSIVE ===
    DO_NOTHING = "do_nothing"


@dataclass
class Action:
    """Represents an action to be taken by a faction."""
    action_type: ActionType
    faction_id: int
    target_faction_id: Optional[int] = None  # For ATTACK action
    # WHERE to build (D28). `None` means "the world decides" — the auto-placement that
    # every caller relied on before the tile head existed, and that the scripted bots and
    # the flat agent still rely on.
    target_tile: Optional[tuple] = None

    def __str__(self):
        if self.action_type == ActionType.ATTACK:
            return f"Faction {self.faction_id}: {self.action_type.value} -> Faction {self.target_faction_id}"
        return f"Faction {self.faction_id}: {self.action_type.value}"


def get_all_action_types():
    """Get list of all action types (useful for ML model output layer)."""
    return list(ActionType)


def action_to_index(action_type: ActionType) -> int:
    """Convert action type to integer index (for neural network)."""
    return list(ActionType).index(action_type)


def index_to_action(index: int) -> ActionType:
    """Convert integer index to action type (from neural network output)."""
    return list(ActionType)[index]



