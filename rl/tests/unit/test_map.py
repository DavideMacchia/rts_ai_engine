"""The world (D27): one tile grid, shared by every tier.

What is pinned here is what the rest of the simulator is now allowed to assume:
the map is FAIR, geography CONSTRAINS what can be built (a plot has finite room, and a
mine needs a vein) but does NOT tax a building's yield — output is workers x skill alone —
and the march is a distance rather than a constant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.managers.building_manager import BuildingManager
from simulator.map import (
    GameMap, Tile, TileType, DepositType, generate_map, find_settlement_sites, rotate,
    PLACEMENT_YIELD_FLOOR,
)
from simulator.config import (
    MAP_WIDTH, MAP_HEIGHT, DISTRICT_RADIUS, SETTLEMENT_SEPARATION,
    MARCH_SECONDS_PER_TILE, ATTACK_TRAVEL_TIME_SECONDS,
)


@pytest.fixture(autouse=True)
def _isolate_global_rng():
    state = np.random.get_state()
    yield
    np.random.set_state(state)


# --- the map is fair, by construction ---------------------------------------

def test_the_map_is_rotationally_symmetric():
    """The two halves are the SAME ground. A random map hands the first settler the better
    plot (measured: water in 84% of faction 0's plots against 58% of faction 1's), and a
    permanent head-start would quietly poison every measurement taken on this simulator."""
    np.random.seed(7)
    game_map = generate_map()

    for x in range(game_map.width):
        for y in range(game_map.height):
            rx, ry = rotate(game_map.width, game_map.height, x, y)
            here, opposite = game_map.tiles[x][y], game_map.tiles[rx][ry]
            assert here.terrain is opposite.terrain
            assert here.deposit is opposite.deposit
            assert here.richness == pytest.approx(opposite.richness)


def test_both_factions_get_congruent_plots():
    """Same ground, therefore the same options: whatever one faction can seat on its plot, the
    other can seat on its mirror plot. (WHICH free tile the packer picks is not mirror-symmetric
    and no longer matters — placement does not affect yield — so only buildability is congruent,
    not the exact site.)"""
    for seed in range(20):
        np.random.seed(seed)
        sim = RealTimeRTSSimulator(num_factions=2)
        f0, f1 = sim.state.factions

        for building in ('lumberyard', 'quarry', 'farm', 'hunting_shed', 'mine', 'well'):
            site0, _q0 = BuildingManager.find_site(f0.capital, building)
            site1, _q1 = BuildingManager.find_site(f1.capital, building)
            assert (site0 is None) == (site1 is None), building


def test_the_settlements_sit_the_intended_distance_apart():
    """The separation is the length of every march. If the site-picker were free to wander,
    it would silently rewrite the tempo of the whole war."""
    for seed in range(20):
        np.random.seed(seed)
        sim = RealTimeRTSSimulator(num_factions=2)
        f0, f1 = sim.state.factions
        d = GameMap.distance(f0.capital.center, f1.capital.center)
        assert abs(d - SETTLEMENT_SEPARATION) <= 3.0


# --- geography constrains what can exist ------------------------------------

def test_a_mine_needs_a_vein():
    """A hard requirement: no iron in reach, no mine. It fails as an impossible action, not
    as an expensive one — and it fails BEFORE the costs are paid."""
    np.random.seed(3)
    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.factions[0]
    district = faction.capital
    game_map = sim.state.map

    # strip every iron vein out of the plot
    for (x, y), tile in list(game_map.tiles_within(district.center, DISTRICT_RADIUS)):
        if tile.deposit is DepositType.IRON:
            tile.deposit = None

    district.resources['wood'] = 10_000
    district.resources['stone'] = 10_000
    wood_before = district.get_resource('wood')

    ok, message = BuildingManager().start_building(faction, 'mine')

    assert ok is False
    assert message == 'invalid_missing_prerequisite'
    assert district.get_resource('wood') == wood_before, "an impossible build must not charge"


def test_you_cannot_build_on_water_or_a_cliff():
    np.random.seed(1)
    game_map = generate_map()
    for x in range(game_map.width):
        for y in range(game_map.height):
            if game_map.tiles[x][y].terrain in (TileType.WATER, TileType.MOUNTAIN):
                assert game_map.placement_quality('house', x, y) is None


def test_a_well_yields_the_same_wherever_it_is_dug():
    """The water table is under every tile alike. A well beside a lake is not a better well —
    it is the same well. Geography has nothing to say about it until the map has soil types."""
    game_map = GameMap(width=3, height=1, tiles=[
        [Tile(TileType.WATER, 1.0)],
        [Tile(TileType.GRASS, 4.0)],
        [Tile(TileType.SAND, 4.0)],
    ])
    assert game_map.placement_quality('well', 0, 0) is None      # you cannot stand on a lake
    assert game_map.placement_quality('well', 1, 0) == pytest.approx(1.0)
    assert game_map.placement_quality('well', 2, 0) == pytest.approx(1.0)   # inland: identical


# --- where you build changes what you get -----------------------------------

def test_placement_quality_ranks_tiles():
    """`placement_quality` still RANKS tiles — a lumberyard site in the woods scores above one
    on bare sand — because the auto-placer uses it to pick where to build and which building to
    raze first. It just no longer feeds yield (see the test below)."""
    game_map = GameMap(width=3, height=1, tiles=[
        [Tile(TileType.FOREST, 4.0)],
        [Tile(TileType.GRASS, 4.0)],     # at the treeline
        [Tile(TileType.SAND, 4.0)],      # and well away from the woods
    ])
    in_the_woods = game_map.placement_quality('lumberyard', 0, 0)
    at_the_treeline = game_map.placement_quality('lumberyard', 1, 0)
    on_the_sand = game_map.placement_quality('lumberyard', 2, 0)

    assert in_the_woods == pytest.approx(1.0)
    assert on_the_sand == pytest.approx(PLACEMENT_YIELD_FLOOR)
    assert in_the_woods > at_the_treeline > on_the_sand


def test_placement_does_not_change_the_yield():
    """Where a building stands does NOT tax its output: yield is workers x skill alone. The map
    limits a district by how many buildings fit on its plot, not by reducing their yield. So a
    lumberyard on bare sand, far from any tree, produces exactly base x productivity."""
    from simulator.config import PRODUCTION_RATES

    np.random.seed(5)
    sim = RealTimeRTSSimulator(num_factions=2)
    d = sim.state.factions[0].capital
    game_map = sim.state.map

    # the worst possible ground: bare sand with no forest anywhere near it
    cx, cy = d.center
    barren = (cx + 1, cy + 1)
    for (x, y) in [barren] + game_map.neighbors(*barren):
        game_map.tiles[x][y].terrain = TileType.SAND
        game_map.tiles[x][y].deposit = None
    game_map.tiles[barren[0]][barren[1]].building = None

    d.add_building('lumberyard', tile=barren)
    d.assign_work(1.0)                                 # staff it from the population
    prod = d.get_building_productivity('lumberyard')
    assert prod > 0, "the lumberyard must be staffed for this to mean anything"

    before = d.get_resource('wood')
    sim.resource_mgr.produce_district_resources(d, 3600.0)   # one hour
    made = d.get_resource('wood') - before

    expected = PRODUCTION_RATES['lumberyard']['wood'] * prod   # NO placement factor
    assert made == pytest.approx(expected), (
        f"placement must not tax yield: made {made}, expected {expected} (a placement floor "
        f"would have cut it)"
    )


# --- the march is a distance, not a constant --------------------------------

def test_the_march_is_derived_from_the_distance_walked():
    np.random.seed(2)
    sim = RealTimeRTSSimulator(num_factions=2)
    f0, f1 = sim.state.factions

    distance = GameMap.distance(f0.capital.center, f1.capital.center)
    assert sim.march_time(f0, f1) == pytest.approx(distance * MARCH_SECONDS_PER_TILE)

    # and it lands near the constant it replaces: introducing a world must not silently
    # re-balance the war (the old flat value was ATTACK_TRAVEL_TIME_SECONDS)
    assert 0.75 * ATTACK_TRAVEL_TIME_SECONDS <= sim.march_time(f0, f1) <= 1.35 * ATTACK_TRAVEL_TIME_SECONDS


def test_a_mapless_game_still_works():
    """The simulator must still run with no world at all — every test written before the map
    existed is a game without one, and they still have to mean what they meant."""
    sim = RealTimeRTSSimulator(num_factions=2, with_map=False)
    f0, f1 = sim.state.factions

    assert sim.state.map is None
    assert f0.capital.center is None
    assert sim.march_time(f0, f1) == pytest.approx(ATTACK_TRAVEL_TIME_SECONDS)


# --- the agent says WHERE (D28) ---------------------------------------------

def test_the_agent_builds_on_the_tile_it_named():
    """The tile head is not advisory: the building goes where the agent said."""
    from agents.district.env import RealTimeRTSEnv
    from simulator.actions import ActionType, action_to_index

    env = RealTimeRTSEnv(opponent_type='hard', spatial=True)
    env.reset(seed=0)
    masks = env.action_masks()
    legal_tiles = np.where(masks[env.action_space_size:])[0]

    chosen = int(legal_tiles[0])
    env.step(np.array([action_to_index(ActionType.BUILD_LUMBERYARD), chosen]))

    in_progress = env.sim.state.factions[0].capital.buildings_in_progress
    assert len(in_progress) == 1
    assert tuple(in_progress[0].tile) == tuple(env.tile_of(chosen))
    env.close()


def test_naming_impossible_ground_is_refused_and_not_charged():
    """Choosing a tile that cannot carry the building is a wrong DECISION, not a typo to be
    quietly corrected: the world does not re-place it somewhere nicer."""
    np.random.seed(11)
    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.factions[0]
    district = faction.capital
    district.resources['wood'] = 10_000
    district.resources['stone'] = 10_000

    # a tile with no iron under it, and no iron beside it
    barren = next(
        t for t, tile in sim.state.map.tiles_within(district.center, DISTRICT_RADIUS)
        if sim.state.map.placement_quality('mine', *t) is None and tile.is_buildable
    )
    wood_before = district.get_resource('wood')

    ok, message = BuildingManager().start_building(faction, 'mine', tile=barren)

    assert ok is False
    assert message == 'invalid_missing_prerequisite'
    assert district.get_resource('wood') == wood_before
    assert not district.buildings_in_progress


def test_you_may_only_build_on_your_own_land():
    np.random.seed(12)
    sim = RealTimeRTSSimulator(num_factions=2)
    faction = sim.state.factions[0]
    district = faction.capital
    district.resources['wood'] = 10_000

    cx, cy = district.center
    far_away = (cx + DISTRICT_RADIUS + 3, cy)

    ok, message = BuildingManager().start_building(faction, 'house', tile=far_away)
    assert ok is False


# --- the founding plot must be able to DEVELOP (D31) -------------------------

def test_every_founding_plot_can_actually_develop():
    """A settlement is founded once and lives with its ground for the whole game, so a plot
    that cannot run an economy is a game lost before a single decision is made. Viability is
    a REQUIREMENT, not a score: before this, plots ranged from 37 to 168 buildable tiles and
    some had no forest, no mountainside or no clay at all — no timber, no stone, or no bricks,
    and therefore no bakery, no dormitory and no forge, ever.

    Variety in what a plot is GOOD at is the point of a map. Variety in whether it is VIABLE
    is an unearned loss.
    """
    from simulator.map import plot_is_viable, PLOT_REQUIREMENTS, MIN_BUILDABLE_TILES, MIN_CLAY_TILES

    for seed in range(30):
        np.random.seed(seed)
        sim = RealTimeRTSSimulator(num_factions=2)
        for faction in sim.state.factions:
            center = faction.capital.center
            assert plot_is_viable(sim.state.map, *center), f"seed {seed}: unviable founding plot"

            plot = [t for _, t in sim.state.map.tiles_within(center, DISTRICT_RADIUS)]
            from simulator.map import UNBUILDABLE
            # buildable is a fact about the LAND (terrain), not current occupancy — the founding
            # warehouse's footprint must not tip a plot below the floor
            assert sum(1 for t in plot if t.terrain not in UNBUILDABLE) >= MIN_BUILDABLE_TILES
            assert sum(1 for t in plot if t.deposit is DepositType.CLAY) >= MIN_CLAY_TILES
            for terrain, needed in PLOT_REQUIREMENTS.items():
                assert sum(1 for t in plot if t.terrain is terrain) >= needed, terrain


def test_iron_is_a_strategic_difference_not_a_viability_one():
    """Iron must stay SCARCE — that is what makes ground worth fighting over — but its absence
    must never stop a district from developing. Nothing in PLOT_REQUIREMENTS asks for it."""
    from simulator.map import PLOT_REQUIREMENTS
    assert DepositType.IRON not in PLOT_REQUIREMENTS

    with_iron = 0
    for seed in range(30):
        np.random.seed(seed)
        sim = RealTimeRTSSimulator(num_factions=2)
        center = sim.state.factions[0].capital.center
        plot = [t for _, t in sim.state.map.tiles_within(center, DISTRICT_RADIUS)]
        with_iron += any(t.deposit is DepositType.IRON for t in plot)

    assert 0 < with_iron < 30, "iron should be neither guaranteed nor impossible"
