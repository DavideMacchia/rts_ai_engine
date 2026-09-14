"""
Tests for the four policy levers (docs/population_and_policies.md).

The contract, in order of importance:
  - defaults reproduce the un-governed district exactly (the BC oracle depends on it)
  - every policy has an EFFECT and a COST, never only a cost  (I3)
  - prohibition never masks a vital need; rationing is not a ban (I1)
  - policies act on conditions; conditions still dominate       (I2)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.config import (
    BASE_HAPPINESS,
    BIRTH_CHANCE_PER_TURN,
    CONSCRIPTION_RATE_PER_HOUR,
    HAPPINESS_FERTILITY_FLOOR,
    MINIMUM_FOOD_RATION_FACTOR,
    MIN_HAPPINESS_FOR_GROWTH,
    POLICY_EFFECTS,
)
from simulator.game_state import District, Faction
from simulator.managers import PopulationManager, ResourceManager, TrainingManager
from simulator.policies import DistrictPolicies, PopulationRate

HOUR = 3600.0


def nominal() -> District:
    """A default district: food on the shelves, free housing, no policies.

    The starting stock is bread + meat, i.e. a VARIED diet, which grants the variety
    happiness bonus (+10). These policy/fertility tests reason about the base happiness
    (75), so we collapse the diet to a single food here; the variety bonus has its own
    tests below.

    The meat is MOVED into the bread, not thrown away: these tests want a district that is
    well fed on ONE food, and deleting the meat outright would also halve its calories and
    quietly push it under MIN_FOOD_PER_CAPITA_FOR_GROWTH — testing a starving district
    instead of an ungoverned one.
    """
    district = District(id=0, faction_id=0)
    district.resources['bread'] += district.resources['meat']
    district.resources['meat'] = 0            # single food -> no variety bonus -> 75
    _employ_everyone(district)                # full employment is part of 'nominal' (D37)
    assert district.get_food_variety() == 1
    assert district.get_food_per_capita() >= 2.0
    assert district.population < district.population_capacity
    return district


def _employ_everyone(district: District):
    """Mark every working-age sim as employed, so the unemployment happiness malus (D37) is
    zero. These tests isolate food/policy/variety; unemployment has its own tests below, and a
    fixture that never ran assign_work would otherwise read as 100% idle."""
    for s in district.sims:
        if s.is_adult or s.is_elder:
            s.employed = True


def varied() -> District:
    """A district with a varied diet (bread + meat), so the variety bonus applies."""
    district = District(id=0, faction_id=0)
    _employ_everyone(district)
    assert district.get_food_variety() >= 2
    return district


def well_stocked() -> District:
    """A district that cannot run out of food inside a probe.

    Consumption is derived from the farm's output (see the TIME SCALE block in
    config.py), so the starting stock is deliberately small. A rationing test must
    not be food-limited, or every policy eats the same amount: all of it.
    """
    district = nominal()
    district.resources['bread'] = 10_000     # edible; grain is not food
    return district


# ------------------------------------------------------------ defaults

def test_default_district_is_ungoverned():
    district = nominal()
    assert district.policies.active() == []
    assert district.policies.happiness_cost() == 0.0
    assert district.policies.fertility_modifier() == 1.0
    assert district.policies.ration_factor() == 1.0


def test_default_happiness_is_the_legacy_hardcoded_value():
    """`population_manager` used to hardcode `happiness = 75.0`."""
    assert nominal().calculate_happiness() == BASE_HAPPINESS == 75.0


def test_default_fertility_is_the_legacy_birth_chance():
    """Every conditional factor must be exactly 1.0 at the nominal operating point."""
    district = nominal()
    district.calculate_happiness()
    assert PopulationManager().fertility(district) == pytest.approx(BIRTH_CHANCE_PER_TURN)


# ----------------------------------------------- I3: effect AND cost

@pytest.mark.parametrize("policy", ['minimum_food', 'block_families', 'forced_conscription'])
def test_every_policy_costs_happiness(policy):
    district = nominal()
    setattr(district.policies, policy, True)
    assert district.policies.active() == [policy]
    assert district.policies.happiness_cost() == POLICY_EFFECTS[policy]
    assert district.calculate_happiness() == BASE_HAPPINESS + POLICY_EFFECTS[policy]


def test_no_policy_is_cost_only():
    """The bug this module fixes: a policy that subtracts happiness and does nothing."""
    pop_mgr, res_mgr, train_mgr = PopulationManager(), ResourceManager(), TrainingManager()

    def behaviour(policies: DistrictPolicies):
        district = well_stocked()
        district.policies = policies
        district.calculate_happiness()
        food_before = district.get_total_food()
        res_mgr.consume_district_resources(district, HOUR)
        train_mgr.conscript_district(district, HOUR)
        return (
            round(pop_mgr.fertility(district), 6),
            round(food_before - district.get_total_food(), 6),
            district.population,
            district.get_unit_count('soldier'),
        )

    baseline = behaviour(DistrictPolicies())
    assert behaviour(DistrictPolicies(minimum_food=True)) != baseline
    assert behaviour(DistrictPolicies(block_families=True)) != baseline
    assert behaviour(DistrictPolicies(forced_conscription=True)) != baseline
    assert behaviour(DistrictPolicies(population_rate=PopulationRate.INCREASE)) != baseline


# --------------------------------------- L1 affordance: minimum_food

def _food_eaten(district: District, hours: float = 1.0) -> float:
    before = district.get_total_food()
    ResourceManager().consume_district_resources(district, hours * HOUR)
    return before - district.get_total_food()


def test_rationing_reduces_food_intake():
    plain, rationed = well_stocked(), well_stocked()
    rationed.policies.minimum_food = True

    eaten_plain = _food_eaten(plain)
    eaten_rationed = _food_eaten(rationed)

    assert eaten_plain > 0
    assert eaten_rationed == pytest.approx(eaten_plain * MINIMUM_FOOD_RATION_FACTOR)


def test_rationing_is_not_a_ban_on_eating():
    """Invariant I1: a vital need is constrained, never masked. Zero intake = deadlock."""
    assert 0.0 < DistrictPolicies(minimum_food=True).ration_factor() < 1.0

    district = well_stocked()
    district.policies.minimum_food = True
    assert _food_eaten(district) > 0  # they still ate


def _happiness_factor(happiness: float) -> float:
    ramp = (happiness - MIN_HAPPINESS_FOR_GROWTH) / (BASE_HAPPINESS - MIN_HAPPINESS_FOR_GROWTH)
    ramp = max(0.0, min(1.0, ramp))
    return HAPPINESS_FERTILITY_FLOOR + (1.0 - HAPPINESS_FERTILITY_FLOOR) * ramp


def test_rationing_costs_fertility_through_happiness():
    """The trade-off is real: food lasts longer, but unhappy people have fewer children."""
    district = nominal()
    district.policies.minimum_food = True
    district.calculate_happiness()

    assert district.happiness == 60.0                       # 75 - 15
    expected = BIRTH_CHANCE_PER_TURN * _happiness_factor(60.0)
    assert PopulationManager().fertility(district) == pytest.approx(expected)
    assert 0.0 < expected < BIRTH_CHANCE_PER_TURN


# --------------------------------- L2 incentive: population_rate

def test_population_rate_orders_fertility():
    pop_mgr = PopulationManager()

    def fert(rate):
        district = nominal()
        district.policies.population_rate = rate
        district.calculate_happiness()
        return pop_mgr.fertility(district)

    assert fert(PopulationRate.INCREASE) > fert(PopulationRate.MAINTAIN) > fert(PopulationRate.DECREASE)


def test_conditions_dominate_the_incentive():
    """Invariant I2: an incentive nudges, it does not overwrite. No housing, no births."""
    district = nominal()
    district.policies.population_rate = PopulationRate.INCREASE
    district.population = district.population_capacity   # nowhere to put a child
    district.calculate_happiness()

    assert PopulationManager().fertility(district) == 0.0


def test_incentive_cannot_conjure_births_without_food():
    district = nominal()
    district.policies.population_rate = PopulationRate.INCREASE
    district.resources['grain'] = 0
    district.resources['bread'] = 0
    district.calculate_happiness()

    assert PopulationManager().fertility(district) == 0.0


# ------------------------------- L3 prohibition: block_families

def test_block_families_masks_childbirth_entirely():
    district = nominal()
    district.policies.block_families = True
    district.policies.population_rate = PopulationRate.INCREASE  # cannot override a mask
    district.calculate_happiness()

    assert district.policies.fertility_modifier() == 0.0
    assert PopulationManager().fertility(district) == 0.0


def test_block_families_does_not_stop_them_eating():
    """I1 again: prohibition touches the discretionary, never the vital."""
    district = nominal()
    district.policies.block_families = True
    assert _food_eaten(district) > 0


# ------------------------------ L4 coercion: forced_conscription

def test_conscription_turns_workers_into_soldiers():
    district = nominal()
    district.policies.forced_conscription = True
    adults_before = district.adults

    # Draft exactly 4, whatever the configured rate: derive the duration from it,
    # and stay well short of exhausting the district so the RATE is what we observe.
    target = 4
    duration = HOUR * target / CONSCRIPTION_RATE_PER_HOUR
    assert target < adults_before

    TrainingManager().conscript_district(district, duration)

    assert district.get_unit_count('soldier') == target
    assert district.adults == adults_before - target


def test_conscription_cannot_draft_more_adults_than_exist():
    """The draft empties the district of MEN, and then stops.

    Only men enlist, so the draft cannot exceed them — and it must not keep "drafting"
    once they are gone: the loop used to be gated on `adults`, so it went on adding
    soldiers it could not take anyone for, conjuring an army out of the women.
    """
    district = nominal()
    district.policies.forced_conscription = True
    adults_before = district.adults
    men_before = district.adult_men
    assert men_before > 0, "fixture must have men to draft"

    TrainingManager().conscript_district(district, 10 * HOUR)

    assert district.get_unit_count('soldier') == men_before
    assert district.adult_men == 0
    # the women are still there, still civilians: the point of men-only recruitment
    assert district.adults == adults_before - men_before


def test_conscription_bypasses_barracks_cost_and_training_time():
    """Coercion replaces the choice; it does not go through the volunteer's pipeline."""
    district = nominal()
    district.policies.forced_conscription = True
    wood_before = district.get_resource('wood')

    TrainingManager().conscript_district(district, HOUR)

    assert district.get_building_count('barracks') == 0
    assert district.get_resource('wood') == wood_before
    assert district.units_in_training == []
    assert district.get_unit_count('soldier') > 0


def test_conscription_is_deterministic_and_consumes_no_randomness():
    """Enabling a policy must not perturb any other stochastic draw."""
    district = nominal()
    district.policies.forced_conscription = True

    np.random.seed(7)
    before = np.random.get_state()
    TrainingManager().conscript_district(district, HOUR)
    after = np.random.get_state()

    assert before[1].tolist() == after[1].tolist() and before[2] == after[2]


def test_conscription_cannot_draft_an_empty_district():
    district = nominal()
    district.policies.forced_conscription = True
    district.population = 0

    TrainingManager().conscript_district(district, 10 * HOUR)

    assert district.population == 0
    assert district.get_unit_count('soldier') == 0
    assert district.conscription_progress == 0.0


def test_conscription_pushes_happiness_to_the_growth_floor():
    """The calibration: BASE_HAPPINESS - 25 == MIN_HAPPINESS_FOR_GROWTH.

    A drafted district lands exactly on the floor, where the happiness ramp is spent.
    Fertility is at its minimum, but not zero — HAPPINESS_FERTILITY_FLOOR keeps a way
    back, deliberately, so misery is not an absorbing state during training.
    """
    assert BASE_HAPPINESS + POLICY_EFFECTS['forced_conscription'] == MIN_HAPPINESS_FOR_GROWTH

    district = nominal()
    district.policies.forced_conscription = True
    district.calculate_happiness()

    assert district.happiness == MIN_HAPPINESS_FOR_GROWTH
    fertility = PopulationManager().fertility(district)
    assert fertility == pytest.approx(BIRTH_CHANCE_PER_TURN * HAPPINESS_FERTILITY_FLOOR)
    assert 0.0 < fertility < BIRTH_CHANCE_PER_TURN


def test_only_starvation_and_prohibition_can_zero_fertility():
    """Unhappiness throttles; it never closes the door. Conditions and masks do."""
    pop_mgr = PopulationManager()

    starving = nominal()
    starving.resources['grain'] = 0
    starving.resources['bread'] = 0
    starving.calculate_happiness()
    assert pop_mgr.fertility(starving) == 0.0

    forbidden = nominal()
    forbidden.policies.block_families = True
    forbidden.calculate_happiness()
    assert pop_mgr.fertility(forbidden) == 0.0

    miserable = nominal()
    miserable.happiness = 0.0
    assert pop_mgr.fertility(miserable) > 0.0


# ----------------------------------------------------- food variety

def test_varied_diet_lifts_happiness_by_tiers():
    """Each distinct food beyond the first adds a bonus, capped. This is what makes a
    low-calorie luxury (fruit) worth building — it is a distinct food."""
    from simulator.config import FOOD_VARIETY_BONUS_PER_TYPE, FOOD_VARIETY_BONUS_CAP

    single = nominal()                           # bread only
    assert single.get_food_variety() == 1
    assert single.calculate_happiness() == BASE_HAPPINESS

    two = varied()                               # bread + meat
    assert two.get_food_variety() == 2
    assert two.calculate_happiness() == BASE_HAPPINESS + FOOD_VARIETY_BONUS_PER_TYPE

    four = varied()
    four.resources.update({'vegetable': 100, 'fruit': 100})
    assert four.get_food_variety() == 4
    expected = min(FOOD_VARIETY_BONUS_CAP, 3 * FOOD_VARIETY_BONUS_PER_TYPE)
    assert four.calculate_happiness() == BASE_HAPPINESS + expected


def test_grain_and_flour_are_not_food():
    """The whole point: milling and baking are necessary, not optional."""
    district = nominal()
    district.resources.update({'bread': 0, 'meat': 0, 'grain': 10_000, 'flour': 10_000})
    assert district.get_total_food() == 0
    assert district.get_food_per_capita() == 0.0


# ----------------------------------------------------- happiness

def test_happiness_is_derived_not_integrated():
    """It recovers the instant conditions do, so a past famine cannot haunt the model."""
    district = nominal()
    district.resources['bread'] = 0     # nominal is single-food (bread); this starves it
    assert district.calculate_happiness() == BASE_HAPPINESS - 10.0   # STARVATION_HAPPINESS_LOSS

    district.resources['bread'] = 50    # edible food restored
    assert district.calculate_happiness() == BASE_HAPPINESS


def test_happiness_never_leaves_zero_hundred():
    district = nominal()
    district.policies = DistrictPolicies(minimum_food=True, block_families=True,
                                         forced_conscription=True)
    district.resources['grain'] = 0
    district.resources['bread'] = 0
    assert 0.0 <= district.calculate_happiness() <= 100.0


# ------------------------------------------- per-district scoping

def test_policies_are_per_district():
    faction = Faction(id=0, districts=[District(id=0), District(id=1)])
    faction.districts[0].policies.forced_conscription = True

    TrainingManager().apply_conscription(faction, HOUR)

    assert faction.districts[0].get_unit_count('soldier') > 0
    assert faction.districts[1].get_unit_count('soldier') == 0


def test_faction_happiness_is_population_weighted():
    # Both districts start with bread + meat -> 2 foods -> +5 variety bonus -> 80.
    home = District(id=0)
    colony = District(id=1)
    colony.policies.minimum_food = True          # 80 - 15 = 65
    faction = Faction(id=0, districts=[home, colony])

    home.population, colony.population = 10, 30
    for district in faction.districts:
        _employ_everyone(district)               # isolate from the unemployment malus (D37)
        district.calculate_happiness()

    assert faction.happiness == pytest.approx((80.0 * 10 + 65.0 * 30) / 40)


def test_faction_policies_accessor_is_ambiguous_with_two_districts():
    faction = Faction(id=0, districts=[District(id=0), District(id=1)])
    with pytest.raises(TypeError, match="cannot be assigned|aggregate"):
        _ = faction.policies
