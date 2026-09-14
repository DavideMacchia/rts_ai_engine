"""The SettlerBot and the civil district's objective (D32).

This bot is the TEACHER for the civil tier — the behavioural-cloning expert. A clone can only
ever reproduce its teacher (measured, twice: D28's spatial clone matched the auto-placer to
97% and could not exceed it by construction), so a teacher that starves is not a teacher.
These tests pin what it must never do.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from agents.district.env import RealTimeRTSEnv, GOODS
from simulator.actions import ActionType, action_to_index, index_to_action
from simulator.opponents import get_opponent
from simulator.professions import PROFESSION_TIER, BUILDING_PROFESSION, Profession


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def _play(seed: int, bot_name: str = 'settler'):
    """One full episode of the civil objective, driven by a scripted bot."""
    env = RealTimeRTSEnv(objective='economy')
    obs, _ = env.reset(seed=seed)
    bot = get_opponent(bot_name, faction_id=0)
    faction = env.sim.state.factions[0]

    done, pop_min = False, float('inf')
    while not done:
        mask = env.action_masks()
        action = bot.act(env.sim.state, env.sim.game_time)
        idx = action_to_index(action.action_type)
        if not mask[idx]:
            idx = int(np.where(mask)[0][-1])
        obs, _r, terminated, truncated, _info = env.step(idx)
        done = terminated or truncated
        pop_min = min(pop_min, faction.population)

    score = env.score()
    env.close()
    return faction, pop_min, score


# --- the objective: this tier makes people and goods, and does not fight -------

def test_the_district_can_build_and_arm_but_cannot_attack():
    """The district raises and arms its own soldiers (district-unified design). What it cannot
    do on a solo plot is ATTACK — there is no enemy — so only combat is withheld, not the
    barracks and the training that turn surplus population into an army."""
    from simulator.actions import ActionType

    env = RealTimeRTSEnv(objective='economy')
    env.reset(seed=0)
    mask = env.action_masks()

    # ATTACK is out of scope on a solo plot.
    for i in range(env.action_space_size):
        if index_to_action(i).name == 'ATTACK':
            assert not mask[i], "ATTACK must be withheld: there is no enemy"

    # Barracks and training are IN the action space (they may need resources/a barracks first,
    # but they exist and are not categorically forbidden).
    military_production = {'BUILD_BARRACKS', 'TRAIN_SOLDIER', 'TRAIN_MAN_AT_ARMS'}
    present = {index_to_action(i).name for i in range(env.action_space_size)}
    assert military_production <= present, "the district must be able to build and train military"

    assert env.num_factions == 1, "solo plot: it develops its own territory"
    assert env.opponent is None
    env.close()


def test_a_lone_district_is_not_declared_the_winner_on_step_zero():
    """The conquest check used to look at the single surviving faction and crown it — the
    settlement was pronounced victorious before it had laid a brick, and the episode ended."""
    env = RealTimeRTSEnv(objective='economy')
    env.reset(seed=0)
    _obs, _r, terminated, truncated, _info = env.step(action_to_index(ActionType.DO_NOTHING))

    assert not terminated
    assert not truncated
    env.close()


def test_the_score_is_the_sum_of_what_the_step_rewards_paid_for():
    """The step reward is the DELTA of the score. If the two could disagree, the agent would
    be shaped toward one thing and measured on another."""
    env = RealTimeRTSEnv(objective='economy')
    env.reset(seed=3)

    start = env.score()
    total = 0.0
    for _ in range(200):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        _obs, reward, _t, _tr, _i = env.step(int(valid[-1]))   # DO_NOTHING
        total += reward

    assert env.score() == pytest.approx(start + total, abs=1e-6)
    env.close()


# --- the teacher must be competent --------------------------------------------

def test_the_settler_never_starves_its_district():
    """It kept building workshops with the hands that were feeding it, and died with fourteen
    buildings standing. A settlement that cannot feed the people it makes has not developed —
    it has failed, and a clone of it would fail the same way."""
    for seed in range(6):
        _faction, pop_min, _score = _play(seed)
        assert pop_min > 0, f"seed {seed}: the district starved to death"


def test_the_settler_practises_the_higher_trades():
    """A workshop nobody is qualified to work is a shed. What the civil tier is FOR is careers
    ripening: a clay digger who becomes a potter (D31)."""
    practised = set()
    for seed in range(6):
        faction, _pop_min, _score = _play(seed)
        district = faction.capital
        for building in district.buildings:
            if district.get_building_count(building) <= 0:
                continue
            prof = BUILDING_PROFESSION.get(building, Profession.NONE)
            if PROFESSION_TIER.get(prof, 0) >= 2 and district.get_building_productivity(building) > 0:
                practised.add(prof)

    assert practised, "not one higher trade was ever practised in six settlements"


def test_the_settler_makes_several_different_goods():
    """Breadth, not a mountain of one commodity: a settlement that produces only wood is not a
    settlement, it is a sawmill."""
    for seed in range(4):
        faction, _pop_min, _score = _play(seed)
        made = [g for g in GOODS if faction.get_resource(g) >= 1.0]
        assert len(made) >= 4, f"seed {seed}: only made {made}"


# --- the district climbs to high-grade production and arms its surplus ---------

def test_the_district_climbs_to_high_grade_production():
    """Exploiting the land is only the start; the worth is in REFINING it. A mature district
    must reach grade-3 output — bread, clothes, or the iron kit — not sit on heaps of raw
    grain and water it never processes (higher grade is what the score is FOR)."""
    from agents.district.env import RESOURCE_GRADE

    grade3 = [g for g, grade in RESOURCE_GRADE.items() if grade == 3]
    for seed in range(6):
        faction, _pop_min, _score = _play(seed)
        made = [g for g in grade3 if faction.get_resource(g) >= 1.0]
        assert made, f"seed {seed}: no grade-3 good ever produced (of {grade3})"


def test_the_district_arms_a_standing_army_from_its_surplus():
    """The surplus population is not waste — it becomes soldiers (district-unified design). A
    mature district builds a barracks and fields a real, armed force by the end of the game."""
    for seed in range(6):
        faction, _pop_min, _score = _play(seed)
        soldiers = sum(faction.units.values())
        assert soldiers >= 3, f"seed {seed}: only {soldiers} soldiers — the surplus went to waste"


def test_the_district_does_not_carry_idle_masses():
    """Population must track the jobs, not outrun them (D37).

    Before D37 the district grew to the food ceiling regardless of whether there was work for
    the extra sims, ending with half its workforce idle — mouths that eat, do not produce, and
    drag food per capita down. Now unemployment feeds into happiness and happiness gates births,
    so the population settles near the work it has. Measured on the REAL idle rate, not a slot
    count: open slots sit in L2/L3 workshops only the qualified can fill, so 'jobs >= workers'
    can hold while people are still idle. What must hold is that few actually sit idle — the
    residue of generational turnover and unfinished apprenticeships, not a permanent crowd.

    Averaged over seeds, not asserted per seed: one map can run high (a big surplus the civil
    tier cannot draft into an army), and n=1 is noise here. The baseline this guards against was
    ~56% idle; the bar is a mean well under a third.
    """
    idle = [_play(seed)[0].capital.unemployment_rate() for seed in range(8)]
    mean_idle = sum(idle) / len(idle)
    assert mean_idle <= 0.30, f"mean idle {mean_idle:.0%} across {len(idle)} seeds: {[f'{i:.0%}' for i in idle]}"


# --- the career ladder must actually be climbable (D34) ------------------------

def test_every_producing_building_employs_somebody():
    """A building that produces must cost hands.

    The clay pit and the well had no worker requirement, so `assign_work` skipped them and
    their productivity fell back on the "needs nobody" default of 1.0: they ran at FULL output
    with an empty floor. Free clay was the smaller half. The clay pit is the first rung of the
    potter's ladder, and a rung nobody stands on teaches nobody — the pottery was built in six
    settlements out of six and staffed in none, with not one grain of clay-digging experience
    anywhere in the district.
    """
    from simulator.config import PRODUCTION_RATES, RESOURCE_PROCESSING, WORKERS_NEEDED

    producers = set(PRODUCTION_RATES) | set(RESOURCE_PROCESSING)
    free = sorted(producers - set(WORKERS_NEEDED))
    assert not free, f"these produce but employ nobody: {free}"


def test_the_clay_digger_becomes_the_potter():
    """The whole point of the career ladder, end to end: a man digs clay until he is good at
    it, and then he is the one who can work the pottery — while somebody else takes his pit."""
    practised = set()
    for seed in range(6):
        faction, _pop_min, _score = _play(seed)
        district = faction.capital
        for sim in district.sims:
            if sim.profession is Profession.POTTER:
                practised.add(Profession.POTTER)
            # nobody becomes a potter without having dug clay first
            if sim.profession is Profession.POTTER:
                assert sim.skills.get(Profession.DIGGER, 0.0) >= 0.5, (
                    "a potter who never dug clay: the prerequisite is not being enforced"
                )

    assert Profession.POTTER in practised, (
        "not one potter in six settlements — the ladder from clay digging is broken again"
    )
