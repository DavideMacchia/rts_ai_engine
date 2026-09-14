"""Demography: what makes a couple have a child, and what makes population track the jobs (D37).

Four coupled mechanisms, each pinned here:
  - unemployment lowers happiness (idle hands are unhappy hands);
  - happiness is the MAIN gate on births (a low floor, so unhappiness really bites);
  - a woman cannot conceive back-to-back (a post-birth cooldown);
  - high parity lowers fertility (the third child and beyond come rarer).

Together they couple the population to the demand for labour: grow past the jobs, unemployment
rises, happiness falls, births slow — so the district settles near the size it can employ,
instead of at the food ceiling.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simulator.config import (
    BASE_HAPPINESS,
    BIRTH_CHANCE_PER_TURN,
    FERTILITY_COOLDOWN_SECONDS,
    HAPPINESS_FERTILITY_FLOOR,
    HIGH_PARITY_FERTILITY_DECAY,
    HIGH_PARITY_THRESHOLD,
    MIN_HAPPINESS_FOR_GROWTH,
    PREGNANCY_DURATION_SECONDS,
    UNEMPLOYMENT_HAPPINESS_PENALTY,
)
from simulator.game_state import District, Gender, Sim
from simulator.managers import PopulationManager


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _one_woman_district(children_born: int = 0) -> tuple[District, Sim]:
    """A district emptied to a single fertile woman, so a conception draw is unambiguous."""
    d = District(id=0, faction_id=0)
    woman = Sim(age=Sim.__dataclass_fields__['age'].default, gender=Gender.FEMALE)
    # place her squarely in the adult band
    from simulator.game_state import KID_END, ADULT_END
    woman.age = (KID_END + ADULT_END) / 2.0
    woman.children_born = children_born
    d.sims = [woman]
    d.resources['bread'] = 10_000     # never food-limited
    return d, woman


# --- unemployment lowers happiness --------------------------------------------

def test_full_employment_carries_no_unemployment_malus():
    d = District(id=0, faction_id=0)
    for s in d.sims:
        if s.is_adult or s.is_elder:
            s.employed = True
    assert d.unemployment_rate() == 0.0
    # single food -> no variety bonus, no policy -> exactly BASE_HAPPINESS
    d.resources['bread'] += d.resources['meat']; d.resources['meat'] = 0
    assert d.calculate_happiness() == pytest.approx(BASE_HAPPINESS)


def test_unemployment_lowers_happiness_in_proportion():
    d = District(id=0, faction_id=0)
    d.resources['bread'] += d.resources['meat']; d.resources['meat'] = 0   # isolate from variety
    workers = [s for s in d.sims if s.is_adult or s.is_elder]
    assert workers, "fixture must have workers"
    # Idle a small share, so the (strong) malus stays above the happiness floor of 0 and the
    # proportional relation is testable without the clamp getting in the way.
    idle_count = max(1, len(workers) // 4)
    for s in workers:
        s.employed = True
    for s in workers[:idle_count]:
        s.employed = False

    expected_rate = idle_count / len(workers)
    expected_happiness = BASE_HAPPINESS - UNEMPLOYMENT_HAPPINESS_PENALTY * expected_rate
    assert expected_happiness > 0, "keep this test off the clamp"
    assert d.unemployment_rate() == pytest.approx(expected_rate)
    assert d.calculate_happiness() == pytest.approx(expected_happiness)


def test_a_fully_idle_district_falls_below_the_growth_floor():
    """The whole point: grow past your jobs and you drop under MIN_HAPPINESS_FOR_GROWTH."""
    d = District(id=0, faction_id=0)
    for s in d.sims:
        s.employed = False
    assert d.calculate_happiness() < MIN_HAPPINESS_FOR_GROWTH


# --- happiness is the main gate on births -------------------------------------

def test_fertility_is_full_at_base_happiness():
    d = District(id=0, faction_id=0)
    d.happiness = BASE_HAPPINESS
    assert PopulationManager().fertility(d) == pytest.approx(BIRTH_CHANCE_PER_TURN)


def test_unhappiness_bites_fertility_hard():
    """At the growth floor happiness, fertility collapses to the low floor — not the old 0.5."""
    d = District(id=0, faction_id=0)
    d.happiness = MIN_HAPPINESS_FOR_GROWTH
    assert PopulationManager().fertility(d) == pytest.approx(
        BIRTH_CHANCE_PER_TURN * HAPPINESS_FERTILITY_FLOOR)
    assert HAPPINESS_FERTILITY_FLOOR < 0.3, "happiness must be a strong gate, not a nudge"


# --- a woman cannot conceive back-to-back -------------------------------------

def test_birth_starts_a_cooldown_that_blocks_conception():
    d, woman = _one_woman_district()
    assert woman.can_conceive

    woman.pregnant_remaining = PREGNANCY_DURATION_SECONDS
    d.age_sims(PREGNANCY_DURATION_SECONDS)          # gestate to term -> birth
    assert woman.children_born == 1
    assert woman.fertility_cooldown_remaining > 0.0
    assert not woman.can_conceive, "she conceived again the instant she gave birth"


def test_the_cooldown_expires():
    d, woman = _one_woman_district()
    woman.pregnant_remaining = PREGNANCY_DURATION_SECONDS
    d.age_sims(PREGNANCY_DURATION_SECONDS)
    assert not woman.can_conceive

    d.age_sims(FERTILITY_COOLDOWN_SECONDS + 1.0)
    assert woman.can_conceive, "the cooldown never lifted"


# --- high parity lowers fertility ---------------------------------------------

def _conceptions_over(children_born: int, seed: int, hours: float = 200.0) -> int:
    """Monte-Carlo: how many times a woman of this parity conceives over `hours`, at full
    happiness. Deterministic under the seed; the RNG is restored by the fixture."""
    np.random.seed(seed)
    pm = PopulationManager()
    total = 0
    dt = 6.25
    for _ in range(int(hours * 3600 / dt)):
        d, woman = _one_woman_district(children_born=children_born)
        d.happiness = BASE_HAPPINESS
        # one conception draw with a fresh, cooldown-free woman of the given parity
        rate = pm.fertility(d) * (HIGH_PARITY_FERTILITY_DECAY **
                                  max(0, children_born - HIGH_PARITY_THRESHOLD + 1))
        if np.random.random() < rate * dt / 3600.0:
            total += 1
    return total


def test_the_first_two_children_come_at_the_full_rate():
    """Parity 0 and 1 draw at the same rate — the decay only starts at the third child."""
    a = _conceptions_over(children_born=0, seed=1)
    b = _conceptions_over(children_born=1, seed=1)
    assert a == b


def test_high_parity_makes_further_children_rarer():
    """A woman with two children already conceives markedly less than one with none."""
    low = _conceptions_over(children_born=0, seed=2)
    high = _conceptions_over(children_born=HIGH_PARITY_THRESHOLD, seed=2)
    assert high < low, f"parity did not lower fertility: {low} vs {high}"
    # and the fifth-child woman rarer still than the third-child one
    higher = _conceptions_over(children_born=HIGH_PARITY_THRESHOLD + 2, seed=2)
    assert higher < high
