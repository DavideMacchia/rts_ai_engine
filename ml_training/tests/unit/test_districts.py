"""
Tests for the Faction-as-a-set-of-Districts model (roadmap Stage 1).

The single-district path is covered by the rest of the suite plus the BC-policy
regression oracle. What is tested here is the part nothing else exercises: a
faction that actually owns more than one district.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.config import PRODUCTION_RATES
from simulator.game_state import District, Faction, GameState
from simulator.managers import (
    BuildingManager,
    CombatManager,
    PopulationManager,
    ResourceManager,
    TrainingManager,
)


def two_district_faction(faction_id: int = 0) -> Faction:
    """A faction with its starting settlement plus one freshly founded district."""
    home = District(id=0, faction_id=faction_id)
    colony = District.found(id=1, faction_id=faction_id, location=7)
    return Faction(id=faction_id, districts=[home, colony])


# ---------------------------------------------------------------- structure

def test_faction_starts_with_exactly_one_district():
    from simulator.config import STARTING_POPULATION
    faction = Faction(id=0)
    assert len(faction.districts) == 1
    assert faction.districts[0].has_warehouse
    assert faction.population == STARTING_POPULATION


def test_founded_district_gets_a_warehouse_and_nothing_else():
    colony = District.found(id=1, faction_id=0, location=3)
    assert colony.has_warehouse
    assert colony.buildings == {'warehouse': 1}
    assert colony.population == 0
    assert colony.location == 3
    assert all(amount == 0 for amount in colony.resources.values())


def test_add_district_adopts_the_district():
    faction = Faction(id=2)
    colony = faction.add_district(District.found(id=1, faction_id=99))
    assert colony.faction_id == 2
    assert faction.districts == [faction.capital, colony]


# ---------------------------------------------------- single-district identity

def test_single_district_containers_are_write_through():
    """The legacy faction-level API must still mutate the sole district."""
    faction = Faction(id=0)
    assert faction.resources is faction.capital.resources
    assert faction.buildings is faction.capital.buildings
    assert faction.units is faction.capital.units

    faction.resources['wood'] = 123
    faction.population = 7
    faction.military_strength = 4.0

    assert faction.capital.get_resource('wood') == 123
    assert faction.capital.population == 7
    assert faction.capital.military_strength == 4.0


# ------------------------------------------------------------- aggregation

def test_scalars_sum_across_districts():
    faction = two_district_faction()
    faction.districts[0].population = 8
    faction.districts[1].population = 5
    faction.districts[0].military_strength = 3.0
    faction.districts[1].military_strength = 1.5

    assert faction.population == 13
    assert faction.military_strength == 4.5


def test_dicts_sum_across_districts():
    faction = two_district_faction()
    faction.districts[0].add_building('farm')
    faction.districts[1].add_building('farm')
    faction.districts[1].add_unit('soldier', 3)

    # starting settlement has 1 farm, +1 here, +1 in the colony
    assert faction.get_building_count('farm') == 3
    assert faction.get_building_count('warehouse') == 2
    assert faction.get_unit_count('soldier') == 3
    assert sum(faction.buildings.values()) == sum(
        sum(d.buildings.values()) for d in faction.districts
    )


def test_progress_lists_concatenate_across_districts():
    faction = two_district_faction()
    faction.districts[1].resources.update({'wood': 500, 'stone': 500})
    building_mgr = BuildingManager()

    building_mgr.start_building(faction, 'farm', district=faction.districts[1])
    building_mgr.start_building(faction, 'farm', district=faction.districts[1])

    assert len(faction.buildings_in_progress) == 2


# --------------------------------------------- aggregates reject silent writes

def test_multi_district_dict_write_raises():
    faction = two_district_faction()
    with pytest.raises(TypeError, match="read-only aggregate"):
        faction.buildings['farm'] = 4
    with pytest.raises(TypeError, match="read-only aggregate"):
        faction.resources['wood'] = 10


def test_multi_district_scalar_write_raises():
    faction = two_district_faction()
    with pytest.raises(TypeError, match="cannot be assigned"):
        faction.population = 20
    with pytest.raises(TypeError, match="cannot be assigned"):
        faction.military_strength = 5.0


# ------------------------------------------------------------ per-district stock

def test_stock_is_per_district_not_pooled():
    """Two half-full warehouses cannot jointly fund one building (logistics is Stage 4)."""
    faction = two_district_faction()
    faction.districts[0].resources.update({'wood': 15, 'stone': 8})
    faction.districts[1].resources.update({'wood': 15, 'stone': 8})

    farm = {'wood': 20, 'stone': 10}
    assert faction.get_resource('wood') == 30      # the total would cover it
    assert not faction.can_afford(farm)            # but no single district does

    faction.districts[1].resources.update({'wood': 30, 'stone': 20})
    assert faction.can_afford(farm)
    assert faction.district_that_can_afford(farm) is faction.districts[1]


def test_building_is_paid_for_and_raised_by_the_district_that_can_afford_it():
    faction = two_district_faction()
    faction.districts[0].resources.update({'wood': 0, 'stone': 0})
    faction.districts[1].resources.update({'wood': 100, 'stone': 100})

    success, message = BuildingManager().start_building(faction, 'farm')

    assert (success, message) == (True, "valid_action")
    assert len(faction.districts[0].buildings_in_progress) == 0
    assert len(faction.districts[1].buildings_in_progress) == 1
    assert faction.districts[1].get_resource('wood') == 80


def test_building_fails_when_no_single_district_can_pay():
    faction = two_district_faction()
    faction.districts[0].resources.update({'wood': 15, 'stone': 5})
    faction.districts[1].resources.update({'wood': 15, 'stone': 5})

    assert BuildingManager().start_building(faction, 'farm') == (False, "invalid_cant_afford")


# ------------------------------------------------------------------ economy

def test_production_accrues_to_the_producing_district():
    faction = two_district_faction()
    faction.districts[1].add_building('lumberyard')
    faction.districts[1].population = 2  # lumberyard needs 2 workers

    wood_before = faction.districts[0].get_resource('wood')
    # Output now scales with the workers' skill (novices produce at SKILL_FLOOR), so
    # derive the expected amount from the building's effective productivity, not the
    # nominal rate. The invariant under test is the ROUTING: it accrues to district 1.
    productivity = faction.districts[1].get_building_productivity('lumberyard')
    assert productivity > 0
    expected = PRODUCTION_RATES['lumberyard']['wood'] * productivity

    ResourceManager().produce_resources(faction, 3600.0)

    assert faction.districts[1].get_resource('wood') == pytest.approx(expected, rel=0.05)
    assert faction.districts[0].get_resource('wood') == wood_before


def test_starvation_is_local_to_a_district():
    """A district with no food starves while a sibling district, stocked, does not.

    The fed district is not frozen at 10 — over this many simulated hours its people
    age, retire and die. What must hold is that famine stays where the famine is.
    """
    faction = two_district_faction()
    # bread is edible; grain is not — a district on grain alone starves.
    faction.districts[0].resources.update({'grain': 0, 'bread': 0, 'meat': 0})
    faction.districts[0].population = 10
    faction.districts[1].resources.update({'bread': 5000})
    faction.districts[1].population = 10

    np.random.seed(0)
    starved, fed = faction.districts
    fed_start = fed.population
    for _ in range(40):                       # ~1 lifespan; long enough to starve, short
        # This probe drives update_population directly, never assign_work, so every sim would
        # read as unemployed and the fed district would die of the D37 malus rather than of
        # famine. Employ them each step to isolate the thing under test — famine locality.
        for d in faction.districts:
            for s in d.sims:
                s.employed = s.is_adult or s.is_elder
        PopulationManager().update_population(faction, 60.0)  # enough to avoid deep demographic drift

    # The invariant is that famine stays LOCAL: the fed district fares far better than
    # the starved one, which dies out. Absolute counts drift with the age model.
    assert starved.population < fed.population, "famine must stay where the famine is"
    assert fed.population >= fed_start * 0.5, "a stocked district must not collapse"
    assert fed.get_total_food() > 0


def test_food_is_consumed_from_the_eating_district():
    faction = two_district_faction()
    faction.districts[1].resources.update({'bread': 1000})
    faction.districts[1].population = 10
    food_home_before = faction.districts[0].get_total_food()

    ResourceManager().consume_resources(faction, 3600.0)

    assert faction.districts[1].get_resource('bread') < 1000
    assert faction.districts[0].get_total_food() < food_home_before  # it eats too


# ----------------------------------------------------------------- military

def test_a_soldier_needs_a_weapon():
    """Since D29 an army is EQUIPPED, not conjured: no spear in the rack, no spearman —
    however much wood and grain the district is sitting on."""
    faction = two_district_faction()
    faction.districts[1].add_building('barracks')
    faction.districts[1].resources.update({'wood': 500, 'grain': 500, 'wooden_weapon': 0})
    faction.districts[1].population = 5

    success, message = TrainingManager().start_training(faction, 'soldier')

    assert not success
    assert message == 'invalid_cant_afford'
    assert not faction.districts[1].units_in_training


def test_training_happens_where_the_barracks_is():
    faction = two_district_faction()
    faction.districts[1].add_building('barracks')
    faction.districts[1].resources.update({'wood': 100, 'grain': 100, 'wooden_weapon': 5})
    faction.districts[1].population = 5

    success, _ = TrainingManager().start_training(faction, 'soldier')
    assert success
    assert len(faction.districts[1].units_in_training) == 1
    assert len(faction.districts[0].units_in_training) == 0

    TrainingManager().progress_training(faction, 3600.0)

    assert faction.districts[1].get_unit_count('soldier') == 1
    assert faction.districts[1].population == 4      # a worker became a soldier
    assert faction.districts[0].get_unit_count('soldier') == 0


def test_training_without_a_barracks_anywhere_is_a_missing_prerequisite():
    faction = two_district_faction()
    faction.districts[1].resources.update({'wood': 100, 'grain': 100})
    assert TrainingManager().start_training(faction, 'soldier') == (
        False, "invalid_missing_prerequisite"
    )


def test_casualties_are_taken_by_every_district():
    faction = two_district_faction()
    faction.districts[0].add_unit('soldier', 10)
    faction.districts[1].add_unit('soldier', 10)

    CombatManager().apply_casualties(faction, 0.5)

    assert faction.districts[0].get_unit_count('soldier') == 5
    assert faction.districts[1].get_unit_count('soldier') == 5
    assert faction.get_unit_count('soldier') == 10


def test_whole_faction_army_fights_as_one_force():
    attacker = two_district_faction(0)
    defender = two_district_faction(1)
    # neither district alone would win; together they do
    attacker.districts[0].add_unit('soldier', 6)
    attacker.districts[1].add_unit('soldier', 6)
    defender.districts[0].add_unit('soldier', 8)
    attacker.calculate_military_strength()
    defender.calculate_military_strength()

    assert CombatManager().execute_attack(attacker, defender) is True


# -------------------------------------------------------------------- defeat

def test_losing_one_warehouse_is_a_wound_not_death():
    faction = two_district_faction()
    faction.districts[0].remove_building('warehouse')

    assert not faction.districts[0].has_warehouse
    assert faction.districts[1].has_warehouse
    assert not faction.is_defeated()


def test_losing_every_warehouse_is_defeat():
    faction = two_district_faction()
    for district in faction.districts:
        district.remove_building('warehouse')

    assert faction.is_defeated()


def test_game_over_requires_razing_all_enemy_warehouses():
    winner = Faction(id=0)
    loser = two_district_faction(1)
    state = GameState(factions=[winner, loser])

    loser.districts[0].remove_building('warehouse')
    state.check_game_over()
    assert not state.game_over

    loser.districts[1].remove_building('warehouse')
    state.check_game_over()
    assert state.game_over
    assert state.winner == 0
