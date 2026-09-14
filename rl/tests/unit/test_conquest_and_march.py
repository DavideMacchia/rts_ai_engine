"""The march, and what it takes to actually TAKE a district.

Two mechanics are under test here, both from D24/D25:

  - conquest is INTENTIONAL: a win that arrives with overwhelming force razes the
    warehouse and takes the district; a narrower win only damages a building. Before
    this, conquest was a lottery — a won battle deleted a RANDOM building at 30%, so
    taking a district meant grinding through every garden until the warehouse came up.

  - an attack is not instantaneous: it MARCHES. An army in transit cannot be re-sent,
    and the attacker commits on what it forecasts at ARRIVAL, not on what it sees at
    departure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.managers.combat_manager import CombatManager
from simulator.game_state import MarchingArmy, UnitInTraining, UnitType
from simulator.actions import Action, ActionType
from simulator.opponents.aggressive import (
    AggressiveBot, predicted_strength_at_arrival, attack_wins_on_arrival, army_is_home,
)
from simulator.config import (
    ATTACK_TRAVEL_TIME_SECONDS, DEFENDER_BONUS, WAREHOUSE_DESTRUCTION_THRESHOLD,
)


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    """Leave the global RNG exactly as we found it.

    The simulator draws from the GLOBAL numpy RNG, and these tests seed it to pin the
    building-damage roll. Without restoring it afterwards, this file silently reseeds
    every test that runs after it (it made an unrelated starvation test fail).
    """
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _two_factions():
    sim = RealTimeRTSSimulator(num_factions=2)
    return sim, sim.state.factions[0], sim.state.factions[1]


def _army(faction, soldiers):
    faction.capital.units['soldier'] = soldiers
    faction.calculate_military_strength()


# --- conquest is intentional ------------------------------------------------

def test_overwhelming_attack_razes_the_warehouse():
    sim, attacker, defender = _two_factions()
    _army(defender, 2)
    # 2 defenders * 1.3 = 2.6 effective; the capture threshold is 2x that
    _army(attacker, int(np.ceil(2 * DEFENDER_BONUS * WAREHOUSE_DESTRUCTION_THRESHOLD)) + 1)

    assert defender.capital.has_warehouse
    assert CombatManager().execute_attack(attacker, defender) is True

    assert not defender.capital.has_warehouse
    assert defender.is_defeated(), "losing the last warehouse is losing the game"


def test_a_narrow_win_does_not_take_the_district():
    """Winning is not conquering: you have to arrive overwhelming."""
    sim, attacker, defender = _two_factions()
    _army(defender, 4)          # 4 * 1.3 = 5.2 effective
    _army(attacker, 6)          # wins (6 > 5.2) but 6 < 2 * 5.2, so no capture
    defender.capital.buildings['lumberyard'] = 3

    np.random.seed(0)
    assert CombatManager().execute_attack(attacker, defender) is True

    assert defender.capital.has_warehouse, "a narrow win must not raze the warehouse"
    assert not defender.is_defeated()


def test_an_undefended_district_falls():
    """No garrison, no district. This is what makes leaving home undefended a real risk."""
    sim, attacker, defender = _two_factions()
    _army(defender, 0)
    _army(attacker, 1)

    assert CombatManager().execute_attack(attacker, defender) is True
    assert defender.is_defeated()


def test_incidental_damage_never_hits_the_warehouse():
    """The random building damage of a narrow win must not be able to end the game:
    conquest is deliberate, so the warehouse only falls to overwhelming force."""
    sim, attacker, defender = _two_factions()
    _army(defender, 4)
    _army(attacker, 6)          # a win, but never a capture
    defender.capital.buildings['lumberyard'] = 1

    for seed in range(40):      # exercise the 30% damage roll many times
        np.random.seed(seed)
        _army(defender, 4)
        _army(attacker, 6)
        CombatManager().execute_attack(attacker, defender)
        assert defender.capital.has_warehouse


# --- the march --------------------------------------------------------------

def test_attack_resolves_on_arrival_not_on_departure():
    sim, attacker, defender = _two_factions()
    _army(attacker, 10)
    _army(defender, 0)
    order = {0: Action(faction_id=0, action_type=ActionType.ATTACK, target_faction_id=1)}

    march = sim.march_time(attacker, defender)      # a distance, not a constant (D27)
    sim.step(order, delta_time=1.0)
    assert sim.state.is_marching(0), "the army is on the road"
    assert not defender.is_defeated(), "nothing has arrived yet"

    sim.step({}, delta_time=march)
    assert defender.is_defeated(), "the army arrived and took the district"


def test_the_army_that_leaves_is_not_home_to_defend():
    """The whole point of D26: marching out is a RISK. The column takes its units with
    it, so the district it left is empty until they walk back."""
    sim, attacker, defender = _two_factions()
    _army(attacker, 10)
    _army(defender, 1)
    order = {0: Action(faction_id=0, action_type=ActionType.ATTACK, target_faction_id=1)}

    march = sim.march_time(attacker, defender)
    sim.step(order, delta_time=1.0)
    attacker.calculate_military_strength()
    assert attacker.military_strength == 0, "the garrison marched out; nobody is holding home"
    assert sim.state.marching_armies[0].units['soldier'] == 10

    # it fights on arrival, then the survivors have to walk all the way back
    sim.step({}, delta_time=march)
    attacker.calculate_military_strength()
    assert attacker.military_strength == 0, "still away: it is walking home"
    assert sim.state.is_marching(0)
    assert sim.state.marching_armies[0].returning

    sim.step({}, delta_time=march)
    attacker.calculate_military_strength()
    assert not sim.state.is_marching(0)
    assert attacker.military_strength > 0, "the survivors rejoined the garrison"


def test_soldiers_trained_during_the_march_do_not_teleport_into_the_battle():
    """Only the column that left does the fighting. Reinforcements raised behind it stay
    behind it — they are home, which is exactly where they are needed."""
    sim, attacker, defender = _two_factions()
    _army(attacker, 3)
    _army(defender, 0)
    order = {0: Action(faction_id=0, action_type=ActionType.ATTACK, target_faction_id=1)}
    sim.step(order, delta_time=1.0)

    # raise a fresh garrison at home while the column is away
    attacker.capital.add_unit('soldier', 5)
    attacker.calculate_military_strength()

    army = sim.state.marching_armies[0]
    assert army.units['soldier'] == 3, "the column is what left, not what exists"


def test_an_army_on_the_march_cannot_be_sent_again():
    sim, attacker, defender = _two_factions()
    _army(attacker, 10)
    _army(defender, 0)
    order = {0: Action(faction_id=0, action_type=ActionType.ATTACK, target_faction_id=1)}

    sim.step(order, delta_time=1.0)
    for _ in range(5):
        sim.step(order, delta_time=1.0)

    assert len(sim.state.marching_armies) == 1, "one army, not a column of copies"


def test_the_bot_does_not_waste_the_march_shouting_attack():
    """While the army marches the tree must fall THROUGH the attack branch, so the bot
    keeps training and building. It used to re-issue ATTACK every step of the journey —
    a no-op that cost it ~12% of all its decisions."""
    sim, faction, opponent = _two_factions()
    _army(faction, 10)
    _army(opponent, 0)
    sim.state.marching_armies.append(
        MarchingArmy(attacker_id=0, defender_id=1, arrival_time=sim.game_time + 90.0))

    context = {'game_state': sim.state, 'faction_id': 0}
    assert army_is_home(context) is False

    bot = AggressiveBot(0, min_attack_time=0.0)
    action = bot.act(sim.state, game_time=5000.0)
    assert action.action_type is not ActionType.ATTACK


# --- committing on the state we will MEET -----------------------------------

def test_forecast_counts_training_that_lands_during_the_march():
    sim, faction, _ = _two_factions()
    _army(faction, 2)
    now = predicted_strength_at_arrival(faction, ATTACK_TRAVEL_TIME_SECONDS)
    assert now == pytest.approx(2.0)

    faction.capital.units_in_training.append(
        UnitInTraining(unit_type=UnitType('soldier'), turns_remaining=10.0))
    faction.capital.units_in_training.append(
        UnitInTraining(unit_type=UnitType('soldier'),
                       turns_remaining=ATTACK_TRAVEL_TIME_SECONDS + 100.0))

    # only the one that LANDS in time counts; the other is still in the barracks
    assert predicted_strength_at_arrival(
        faction, ATTACK_TRAVEL_TIME_SECONDS) == pytest.approx(3.0)


def test_forecast_is_capped_by_the_men_who_can_actually_enlist():
    """A barracks queue longer than the district's men does not become an army: each
    completed soldier consumes an adult man, and only men enlist."""
    sim, faction, _ = _two_factions()
    district = faction.capital
    district.sims = []
    district.add_adults(20)
    men = district.adult_men
    assert men > 0

    for _ in range(men + 5):
        district.units_in_training.append(
            UnitInTraining(unit_type=UnitType('soldier'), turns_remaining=1.0))

    forecast = predicted_strength_at_arrival(faction, ATTACK_TRAVEL_TIME_SECONDS)
    assert forecast == pytest.approx(float(men)), "cannot enlist men you do not have"


def test_bot_will_not_march_into_a_defence_that_beats_it():
    sim, faction, opponent = _two_factions()
    _army(faction, 4)
    _army(opponent, 4)      # 4 * 1.3 = 5.2 > 4: we would arrive and lose

    context = {'faction': faction, 'opponent': opponent, 'attack_safety_margin': 1.0}
    assert attack_wins_on_arrival(context) is False

    _army(faction, 6)       # 6 > 5.2: now it is a win
    assert attack_wins_on_arrival(context) is True
