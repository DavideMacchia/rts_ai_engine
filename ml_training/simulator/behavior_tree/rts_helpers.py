"""
RTS-specific helper functions for behavior trees.
Conditions and actions tailored for the RTS game.
"""

from typing import Optional
import sys
import os

# Import game-specific classes
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..actions import Action as GameAction, ActionType
from ..game_state import Faction
from ..config import BUILDING_COSTS, UNIT_TRAINING_COST


# ============================================================================
# CONDITION HELPERS - Return bool
# ============================================================================

def is_under_attack(context: dict) -> bool:
    """Check if faction is under military threat."""
    faction: Faction = context['faction']
    opponent: Faction = context['opponent']

    if opponent is None:
        return False

    # Under attack if opponent has 2x our military strength
    return opponent.military_strength > faction.military_strength * 2



def can_attack_opponent(context: dict) -> bool:
    """Check if we have enough military to attack."""
    faction: Faction = context['faction']
    opponent: Faction = context['opponent']

    if opponent is None:
        return False

    # Attack if we have ANY military and opponent has none (easy target!)
    if opponent.military_strength == 0 and faction.military_strength > 0:
        return True

    # Attack if we have good military advantage (1.2x or more)
    has_advantage = faction.military_strength > opponent.military_strength * 1.2

    # Attack if opponent is very weak and we have some force
    opponent_weak = opponent.military_strength < 5 and faction.military_strength > 8

    return has_advantage or opponent_weak


def needs_housing(context: dict) -> bool:
    """Check if we need more houses."""
    faction: Faction = context['faction']

    # Need housing if population is near capacity (within 8 population buffer)
    near_capacity = faction.population >= faction.population_capacity - 8

    # OR if we have very low capacity AND enough resources to afford houses
    low_capacity = faction.population_capacity < 30
    has_resources_for_housing = (
        faction.get_resource('wood') >= 60 and
        faction.get_resource('stone') >= 30
    )

    return near_capacity or (low_capacity and has_resources_for_housing)


def has_critically_low_stone(context: dict) -> bool:
    """Check if stone is critically low (emergency situation)."""
    faction: Faction = context['faction']
    # Critical threshold - can't build most buildings
    return faction.get_resource('stone') < 50


def has_critically_low_wood(context: dict) -> bool:
    """Check if wood is critically low (emergency situation)."""
    faction: Faction = context['faction']
    # Critical threshold - can't build most buildings
    return faction.get_resource('wood') < 50


def has_low_wood(context: dict) -> bool:
    """Check if wood is low."""
    faction: Faction = context['faction']
    # More lenient threshold - build lumberyards earlier
    return faction.get_resource('wood') < 300


def has_low_stone(context: dict) -> bool:
    """Check if stone is low."""
    faction: Faction = context['faction']
    # More lenient threshold - build quarries earlier
    return faction.get_resource('stone') < 300


def has_low_food(context: dict) -> bool:
    """Check if EDIBLE food is low. Grain is not food — it must be milled and baked."""
    faction: Faction = context['faction']
    return faction.get_total_food() < 200


def has_few_lumberyards(context: dict) -> bool:
    """Check if we need more lumberyards."""
    faction: Faction = context['faction']
    lumberyards = faction.get_building_count('lumberyard')

    # Build 2-3 lumberyards for steady wood production
    return lumberyards < 3


def has_few_quarries(context: dict) -> bool:
    """Check if we need more quarries."""
    faction: Faction = context['faction']
    quarries = faction.get_building_count('quarry')

    # Build 2-3 quarries for steady stone production (critical resource!)
    return quarries < 3


def has_few_farms(context: dict) -> bool:
    """Check if we need more farms."""
    faction: Faction = context['faction']
    farms = faction.get_building_count('farm')

    # Build 3-4 farms for food production
    return farms < 4


def has_no_barracks(context: dict) -> bool:
    """Check if we don't have a barracks yet."""
    faction: Faction = context['faction']
    return faction.get_building_count('barracks') == 0


def has_barracks(context: dict) -> bool:
    """Check if we have at least one barracks."""
    faction: Faction = context['faction']
    return faction.get_building_count('barracks') > 0


def needs_more_military(context: dict) -> bool:
    """Check if we need to build more military."""
    faction: Faction = context['faction']
    opponent: Faction = context['opponent']

    if opponent is None:
        # Build some military anyway
        return faction.military_strength < 20

    # Build military to match opponent + buffer
    desired_military = max(15, opponent.military_strength * 1.2)
    return faction.military_strength < desired_military





def has_worker_surplus(context: dict) -> bool:
    """Check if we have spare workers available."""
    faction: Faction = context['faction']
    available = faction.get_available_workers()
    needed = faction.get_total_workers_needed()
    # Have surplus if we have at least 3 more workers than needed
    return available >= needed + 3


def has_worker_deficit(context: dict) -> bool:
    """Check if we don't have enough workers for current buildings."""
    faction: Faction = context['faction']
    available = faction.get_available_workers()
    needed = faction.get_total_workers_needed()
    # Deficit if we need more workers than we have
    return available < needed


def has_good_economy(context: dict) -> bool:
    """Check if economy is well developed enough for military."""
    faction: Faction = context['faction']

    # Good economy = basic resource buildings established
    has_lumber = faction.get_building_count('lumberyard') >= 1
    has_quarry = faction.get_building_count('quarry') >= 1
    has_farms = faction.get_building_count('farm') >= 1
    has_workers = faction.get_available_workers() >= 8

    return has_lumber and has_quarry and has_farms and has_workers


# ============================================================================
# ACTION HELPERS - Return GameAction or None
# ============================================================================

def try_build_house(context: dict) -> Optional[GameAction]:
    """Try to build a house."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['house']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_HOUSE)
    return None


def try_build_lumberyard(context: dict) -> Optional[GameAction]:
    """Try to build a lumberyard."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['lumberyard']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_LUMBERYARD)
    return None


def try_build_quarry(context: dict) -> Optional[GameAction]:
    """Try to build a quarry."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['quarry']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_QUARRY)
    return None


def try_build_farm(context: dict) -> Optional[GameAction]:
    """Try to build a farm."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['farm']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_FARM)
    return None


def try_build_well(context: dict) -> Optional[GameAction]:
    """Try to build a well."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['well']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_WELL)
    return None


def try_build_hunting_shed(context: dict) -> Optional[GameAction]:
    """Try to build a hunting shed — meat with no processing chain, the fast food route."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['hunting_shed']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_HUNTING_SHED)
    return None


def try_build_vegetable_garden(context: dict) -> Optional[GameAction]:
    """Try to build a vegetable garden — a cheap, fast second food for the variety bonus."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['vegetable_garden']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_VEGETABLE_GARDEN)
    return None


def try_build_orchard(context: dict) -> Optional[GameAction]:
    """Try to build an orchard — a slow luxury food, mostly for the variety happiness."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['orchard']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_ORCHARD)
    return None


def try_build_school(context: dict) -> Optional[GameAction]:
    """Try to build a school — invests an adult (teacher) to raise everyone's skill."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['school']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_SCHOOL)
    return None


def try_build_barracks(context: dict) -> Optional[GameAction]:
    """Try to build a barracks."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(BUILDING_COSTS['barracks']):
        return GameAction(faction_id=faction_id, action_type=ActionType.BUILD_BARRACKS)
    return None


def _try_build(context: dict, building: str, action: ActionType) -> Optional[GameAction]:
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS[building]):
        return GameAction(faction_id=context['faction_id'], action_type=action)
    return None


def try_build_clay_pit(context: dict):
    return _try_build(context, 'clay_pit', ActionType.BUILD_CLAY_PIT)


def try_build_pottery(context: dict):
    """The potter. Clay -> bricks, and bricks are what the school, the bakery and the
    dormitory are made of: without him the district cannot develop past its first hour."""
    return _try_build(context, 'pottery', ActionType.BUILD_POTTERY)


def try_build_smelter(context: dict):
    return _try_build(context, 'smelter', ActionType.BUILD_SMELTER)


def try_build_stone_cutter(context: dict):
    return _try_build(context, 'stone_cutter', ActionType.BUILD_STONE_CUTTER)


def try_build_tailor(context: dict):
    return _try_build(context, 'tailor', ActionType.BUILD_TAILOR)


def try_build_dormitory(context: dict):
    return _try_build(context, 'dormitory', ActionType.BUILD_DORMITORY)


def try_build_cattle_shed(context: dict):
    return _try_build(context, 'cattle_shed', ActionType.BUILD_CATTLE_SHED)


def try_build_carpentry(context: dict) -> Optional[GameAction]:
    """A carpentry. Since D29 this is not a workshop, it is the ARMOURY: a soldier needs a
    weapon, and the wooden weapon is where every army starts."""
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS['carpentry']):
        return GameAction(faction_id=context['faction_id'],
                          action_type=ActionType.BUILD_CARPENTRY)
    return None


def try_build_blacksmith(context: dict) -> Optional[GameAction]:
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS['blacksmith']):
        return GameAction(faction_id=context['faction_id'],
                          action_type=ActionType.BUILD_BLACKSMITH)
    return None


def try_build_mine(context: dict) -> Optional[GameAction]:
    """Only possible where there is an iron vein — the map decides (D27)."""
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS['mine']):
        return GameAction(faction_id=context['faction_id'], action_type=ActionType.BUILD_MINE)
    return None


def try_train_man_at_arms(context: dict) -> Optional[GameAction]:
    """The same man, in iron. Worth far more than a spearman, because MEN — not weapons —
    are what a district runs out of (D22)."""
    faction: Faction = context['faction']
    if faction.can_afford(UNIT_TRAINING_COST['man_at_arms']):
        return GameAction(faction_id=context['faction_id'],
                          action_type=ActionType.TRAIN_MAN_AT_ARMS)
    return None


def try_train_soldier(context: dict) -> Optional[GameAction]:
    """Try to train a soldier."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(UNIT_TRAINING_COST['soldier']):
        return GameAction(faction_id=faction_id, action_type=ActionType.TRAIN_SOLDIER)
    return None


def try_train_archer(context: dict) -> Optional[GameAction]:
    """Try to train an archer."""
    faction: Faction = context['faction']
    faction_id: int = context['faction_id']

    if faction.can_afford(UNIT_TRAINING_COST['archer']):
        return GameAction(faction_id=faction_id, action_type=ActionType.TRAIN_ARCHER)
    return None


def try_attack_opponent(context: dict) -> Optional[GameAction]:
    """Try to attack the opponent."""
    faction_id: int = context['faction_id']
    opponent: Faction = context['opponent']

    if opponent is None:
        return None

    return GameAction(
        faction_id=faction_id,
        action_type=ActionType.ATTACK,
        target_faction_id=opponent.id
    )


def try_build_mill(context: dict) -> Optional[GameAction]:
    """Try to build a mill.

    Both of these used to check a HARDCODED price (80 wood / 40 stone for the mill, 90/50 for
    the bakery) that had drifted away from the real one in BUILDING_COSTS — and drifted UPWARD,
    so the bot stood there with enough to build and refused. The district hovered at ~46 wood
    against a gate of 80 and never once built a mill in a full undisturbed game: the bread
    chain, the only food that scales, was unreachable because of a stale number in an `if`.
    A build helper must ask the price list, not remember a price.
    """
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS['mill']):
        return GameAction(faction_id=context['faction_id'], action_type=ActionType.BUILD_MILL)
    return None


def try_build_bakery(context: dict) -> Optional[GameAction]:
    """Try to build a bakery. (See `try_build_mill` on hardcoded prices.)"""
    faction: Faction = context['faction']
    if faction.can_afford(BUILDING_COSTS['bakery']):
        return GameAction(faction_id=context['faction_id'], action_type=ActionType.BUILD_BAKERY)
    return None


def do_nothing(context: dict) -> Optional[GameAction]:
    """Return do-nothing action (always succeeds)."""
    faction_id: int = context['faction_id']
    return GameAction(faction_id=faction_id, action_type=ActionType.DO_NOTHING)
