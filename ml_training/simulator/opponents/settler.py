"""SettlerBot — the expert for the CIVIL district tier (D32).

It builds a settlement. It does not fight, it does not raise an army, and it never learns
to: war is the military camp's job, and the camp is a different brain with a different
model (`agents/camp/`).

This replaces `AggressiveBot` as the teacher for this tier: with a conquest to chase, the
trades always rank below the army, and development becomes what the district does with
whatever is left over — which is nothing.

The order below is the order the MATERIALS flow, because that is the only order that works:
- you cannot bake bread without a mill, and cannot build a bakery without bricks;
- you cannot have bricks without a potter, and cannot have a potter without a clay digger
  who has dug long enough to learn the trade (D31);
- so the district's real job is to start the careers early and let them ripen.
"""

from .base import ScriptedOpponent
from ..actions import Action, ActionType
from ..game_state import GameState, Faction
from ..policies import PopulationRate
from ..managers.building_manager import BuildingManager
from ..config import WORKERS_NEEDED
from ..behavior_tree.nodes import Selector, Sequence, Condition, Action as BTAction
from ..behavior_tree.rts_helpers import (
    needs_housing, try_build_house, try_build_dormitory,
    try_build_lumberyard, try_build_quarry, try_build_clay_pit, try_build_mine,
    try_build_farm, try_build_hunting_shed, try_build_vegetable_garden, try_build_orchard,
    try_build_cattle_shed, try_build_mill, try_build_bakery, try_build_well,
    try_build_pottery, try_build_carpentry, try_build_stone_cutter,
    try_build_smelter, try_build_blacksmith, try_build_tailor, try_build_school,
    try_build_barracks, try_train_soldier, try_train_man_at_arms,
    do_nothing,
)

#: What a fully-exploited territory looks like. Ordered: each entry is built only once every
#: entry above it exists, so the careers that feed a trade are always started before it. The
#: extractors are built in NUMBERS, not one apiece — a district with no population cap grows a
#: large workforce, and its worth is in putting that workforce to work turning the land's raw
#: resources into high-grade goods and, from the surplus, an army. The iron chain at the end is
#: gated on the plot actually holding a vein (see `_next_in_plan`).
BUILD_PLAN = [
    # --- the essential ground floor: ONE of each extractor the chain needs. Ground is scarce
    #     (footprints), so the plan is the MINIMAL complete economy — extra pits and food come
    #     later, from whatever land is left over. ---
    ('lumberyard', 2, try_build_lumberyard),        # wood feeds building + weapons
    ('quarry', 1, try_build_quarry),                # stone, and the road to iron
    ('clay_pit', 1, try_build_clay_pit),            # clay, and the road to pottery
    ('farm', 2, try_build_farm),                    # grain for the bread chain AND the soldiers
    ('hunting_shed', 1, try_build_hunting_shed),    # meat while the bread chain goes up
    ('well', 1, try_build_well),                    # water: the bakery needs it

    # --- the bread chain FIRST: bread is the food that scales, so a couple of buildings feed
    #     the whole district and spare the land a sprawl of gardens. ---
    ('pottery', 1, try_build_pottery),              # <- clay digger.  BRICKS (gate the bakery)
    ('mill', 1, try_build_mill),                    # millers -> flour
    ('bakery', 1, try_build_bakery),                # <- miller.  BREAD (grade 3)

    # --- the rest of the trades ---
    ('carpentry', 1, try_build_carpentry),          # <- woodcutter.  ARMS the militia.
    ('stone_cutter', 1, try_build_stone_cutter),    # <- miner.        DRESSED STONE.
    ('cattle_shed', 1, try_build_cattle_shed),      # breeders: meat, leather, wool
    ('school', 1, try_build_school),                # an ACCELERATOR of skill (D31)
    ('tailor', 1, try_build_tailor),                # <- breeder. CLOTHES (grade 3)

    # --- the armoury and the iron kit (iron-gated: only where the plot holds a vein) ---
    ('barracks', 1, try_build_barracks),            # where the surplus becomes soldiers
    ('mine', 1, try_build_mine),                    # <- iron miner (L2, needs stone mining)
    ('smelter', 1, try_build_smelter),              # <- smelter (L3)
    ('blacksmith', 1, try_build_blacksmith),        # IRON WEAPON + ARMOUR (grade 3)
]

#: The iron chain cannot be built on a plot with no vein; listed here so `_next_in_plan` skips
#: it instead of deadlocking on a mine it can never place.
_IRON_CHAIN = {'mine', 'smelter', 'blacksmith'}


def _count(faction: Faction, building: str) -> int:
    """Completed plus under construction — otherwise the bot re-orders all through the build."""
    in_progress = sum(
        1 for b in faction.buildings_in_progress
        if getattr(b.building_type, 'value', b.building_type) == building
    )
    return faction.get_building_count(building) + in_progress


def _has_iron_in_reach(faction: Faction) -> bool:
    """Does the plot hold an iron vein the district could mine? A mine already built counts;
    otherwise ask the map for a free vein site (D27). No vein -> the iron chain is skipped."""
    if _count(faction, 'mine') > 0:
        return True
    tile, _q = BuildingManager.find_site(faction.capital, 'mine')
    return tile is not None


def _next_in_plan(context: dict):
    faction: Faction = context['faction']
    iron = _has_iron_in_reach(faction)
    for building, target, build in BUILD_PLAN:
        if building in _IRON_CHAIN and not iron:
            continue        # no vein on this plot: the iron chain is not buildable, skip it
        if _count(faction, building) < target:
            return building, build
    return None, None


def plan_incomplete(context: dict) -> bool:
    return _next_in_plan(context)[0] is not None


def try_build_next(context: dict):
    """Build the next thing the plan is missing — and if we cannot afford it yet, WAIT for it
    rather than skipping ahead. Skipping is how a district ends up with six workshops it has
    nobody to staff and no bricks to build the seventh."""
    _building, build = _next_in_plan(context)
    return build(context) if build else None


#: A district must know it cannot feed itself BEFORE the larder is empty, so hunger is judged
#: on PRODUCTION against what the people eat — a leading indicator — and never on the stock in
#: the warehouse, which is a lagging one: by the time the shelves are bare, the babies are
#: already born and the hands that should have been hunting are already in the workshops.
FOOD_PRODUCTION_MARGIN = 1.25      # feed the population we will have, not the one we have

#: If the food buildings it already owns are manned worse than this, another one is not the
#: answer — the district is short of hands, not of huts (see `is_hungry`).
FOOD_STAFFING_FLOOR = 0.55

#: What a hungry district can build TO EAT. These three are MEALS.
#:
#: The farm is deliberately NOT on this list: a farm grows grain, and grain is not food (D16)
#: — it must be milled and baked first. A farm answers hunger with a promise, and puts a
#: farmer to work growing something nobody can eat yet. It belongs to the bread chain, in the
#: plan, as the investment it is.
FOOD_BUILDINGS = [
    ('hunting_shed', try_build_hunting_shed),        # meat, immediately
    ('vegetable_garden', try_build_vegetable_garden),# vegetables, immediately
    ('orchard', try_build_orchard),                  # fruit, and a fourth food for the table
]

#: There is deliberately NO cap on food buildings. A cap ("past six, hunger means you are
#: short of farmers, not of huts") is a rule that can silence a famine — the district hits the
#: limit, `is_hungry` goes quiet, and it starves against the wall. If the district needs an
#: eighth hunting shed to live, it builds an eighth one.



def food_production_per_hour(faction: Faction) -> float:
    """What this district actually GROWS and CATCHES in an hour, at its current staffing.

    Counts the edible output of every food building, scaled by who is working it and where it
    stands — an unstaffed hunting shed feeds nobody, and the bot has to know that.
    """
    from ..config import PRODUCTION_RATES, RESOURCE_PROCESSING, EDIBLE_FOODS

    total = 0.0
    for district in faction.districts:
        for building, count in district.buildings.items():
            if count <= 0:
                continue
            productivity = district.get_building_productivity(building)
            if productivity <= 0.0:
                continue
            outputs = (RESOURCE_PROCESSING.get(building, {}).get('output')
                       or PRODUCTION_RATES.get(building, {}))
            for resource, rate in outputs.items():
                if resource in EDIBLE_FOODS:
                    total += rate * count * productivity
    return total


def _food_staffing(faction: Faction) -> float:
    """How well-manned the food buildings it ALREADY has are (0 = nobody in them)."""
    counts, total = 0, 0.0
    for district in faction.districts:
        for building, _b in FOOD_BUILDINGS:
            n = district.get_building_count(building)
            if n > 0:
                counts += n
                total += district.get_building_productivity(building) * n
    return total / counts if counts else 0.0


def is_hungry(context: dict) -> bool:
    """Will this district be able to feed itself? — not: is its larder thin right now.

    And crucially: is another hut even the ANSWER? Every food building needs a hand, and the
    hands come out of the same pool. Past a point the district is not short of huts, it is
    short of HUNTERS — its existing sheds are already half-manned — and building an eighth one
    only spreads the same people thinner. It chased its own tail there for a whole session:
    never starving, and never building a single trade either, because it was always a little
    behind on food and food always came first.

    When the huts it has are under-manned, the way out is not more huts. It is BREAD — the only
    food that scales (one farm feeds forty, against a hunting shed's ten) — and bread is what
    the plan is for. So it stops digging gardens and goes and builds the mill.
    """
    from ..config import POPULATION_FOOD_CONSUMPTION

    faction: Faction = context['faction']
    eaten = POPULATION_FOOD_CONSUMPTION * max(1, faction.population)
    fed = food_production_per_hour(faction) >= eaten * FOOD_PRODUCTION_MARGIN
    if fed:
        return False

    #: Under this, the sheds it owns are not being worked: the shortage is of people, not of
    #: buildings, and one more shed makes it worse rather than better.
    return _food_staffing(faction) >= FOOD_STAFFING_FLOOR


def is_starving(context: dict) -> bool:
    """Truly short of food NOW — production below what the people eat, with no margin. This is
    the only food need that outranks development: on a plot where ground is scarce, sprinkling
    gardens 'to be safe' fills the land before the district ever builds its bread chain or its
    barracks. So it builds food ahead of the plan only when actually going hungry; the planned
    farms and the bread chain feed it the rest of the time."""
    from ..config import POPULATION_FOOD_CONSUMPTION
    faction: Faction = context['faction']
    eaten = POPULATION_FOOD_CONSUMPTION * max(1, faction.population)
    if food_production_per_hour(faction) >= eaten:
        return False
    return _food_staffing(faction) >= FOOD_STAFFING_FLOOR


def try_build_food(context: dict):
    """Answer hunger with the BREAD CHAIN first, then quick food.

    Bread is the food that scales — one bakery feeds far more than a hunting shed — so when the
    district is hungry and the chain is not finished, the way out is to finish it (mill, then
    bakery), not to sprinkle another garden. Sprinkling gardens is exactly how a large
    population chases its own tail on food and never frees a turn to develop. Only if the chain
    cannot be advanced this step does it fall back on a quick, varied food building (D18)."""
    faction: Faction = context['faction']
    if _count(faction, 'bakery') < 1:
        if _count(faction, 'mill') >= 1:
            action = try_build_bakery(context)     # bricks + flour are ready: bake
            if action is not None:
                return action
        elif _count(faction, 'pottery') >= 1:
            action = try_build_mill(context)        # flour, the step before bread
            if action is not None:
                return action
    _building, build = min(FOOD_BUILDINGS, key=lambda bb: _count(faction, bb[0]))
    return build(context)


#: Raw materials that construction and the workshops burn through, and the extractor for each.
#: A plan with a fixed handful of pits was sized for a small settlement; a territory with no
#: population cap grows a workforce that can (and must) exploit far more of the land. When a raw
#: material runs dry it blocks every build that needs it, so the district digs another pit —
#: putting its surplus hands to work — up to a bound that stands in for the plot's extent.
RAW_EXTRACTORS = [
    ('stone', 'quarry', try_build_quarry),
    ('wood', 'lumberyard', try_build_lumberyard),
    ('clay', 'clay_pit', try_build_clay_pit),
]
RAW_LOW = 40                    # below this, a raw material is starving construction
RAW_EXTRACTOR_CAP = 3          # ground is scarce (footprints): a few pits, not a sprawl of them


def _scarce_extractor(context: dict):
    faction: Faction = context['faction']
    for res, building, build in RAW_EXTRACTORS:
        if faction.get_resource(res) < RAW_LOW and _count(faction, building) < RAW_EXTRACTOR_CAP:
            return build
    return None


def try_exploit_more(context: dict):
    build = _scarce_extractor(context)
    return build(context) if build else None


def needs_pro_natalist_policy(context: dict) -> bool:
    faction: Faction = context['faction']
    return faction.capital.policies.population_rate is not PopulationRate.INCREASE


def try_set_pro_natalist_policy(context: dict):
    context['chosen_action'] = Action(
        faction_id=context['faction_id'],
        action_type=ActionType.SET_POPULATION_RATE_INCREASE,
    )
    return context['chosen_action']


# --- military: the surplus population becomes an army -------------------------

def _job_slots(faction: Faction) -> int:
    """Civilian jobs the built economy offers (its full crews)."""
    total = 0
    for d in faction.districts:
        for b in d.buildings:
            if d.get_building_count(b) > 0:
                total += WORKERS_NEEDED.get(b, 1) * d.get_building_count(b)
    return total


def can_field_soldier(context: dict) -> bool:
    """Train a soldier only from the SURPLUS: a district with a barracks whose working-age
    population already exceeds its civilian jobs has idle hands, and idle hands are what an
    army is made of. Also needs a weapon in the rack — the carpentry's spear, or the smith's
    iron kit — because a soldier is EQUIPPED, not conjured (D29)."""
    faction: Faction = context['faction']
    if _count(faction, 'barracks') <= 0:
        return False
    workers = sum(1 for d in faction.districts for s in d.sims if s.is_adult or s.is_elder)
    if workers <= _job_slots(faction):
        return False        # no surplus: everyone the economy needs is still working
    if faction.get_resource('grain') < 5:
        return False        # a soldier is fed on the march: no grain to spare, no muster
    armed = faction.get_resource('wooden_weapon') >= 1 or (
        faction.get_resource('iron_weapon') >= 1 and faction.get_resource('armour') >= 1)
    return armed


def try_field_soldier(context: dict):
    """Arm the best soldier the rack allows: a man-at-arms in iron when the kit is there (worth
    far more), else a spearman."""
    faction: Faction = context['faction']
    if faction.get_resource('iron_weapon') >= 1 and faction.get_resource('armour') >= 1:
        return try_train_man_at_arms(context)
    return try_train_soldier(context)


class SettlerBot(ScriptedOpponent):
    """Develops a territory end to end: exploits its land, climbs to high-grade goods, and
    turns its surplus population into an armed force. Builds, but does not attack."""

    def __init__(self, faction_id: int):
        super().__init__(faction_id)
        self.name = "SettlerBot"
        self.difficulty = "settler"
        self.behavior_tree = self._build_tree()

    def _build_tree(self):
        return Selector(children=[
            # People first, always. Every empty workshop in this district is a person it
            # never had — and a career takes a lifetime to ripen, so the sooner a child is
            # born, the sooner there is a potter.
            Sequence(children=[
                Condition(needs_pro_natalist_policy, "Not Pro-Natalist Yet?"),
                BTAction(try_set_pro_natalist_policy, "Encourage Births"),
            ], name="Population Policy"),

            # ONLY genuine hunger outranks development: on a plot where ground is scarce,
            # building food "to be safe" fills the land before the essential chain ever goes up.
            Sequence(children=[
                Condition(is_starving, "Actually Going Hungry?"),
                BTAction(try_build_food, "Build Food"),
            ], name="Famine Relief"),

            # DEVELOPMENT FIRST, and it RESERVES the ground. While the essential plan is
            # unfinished the district builds its next piece, or — if that piece is blocked for
            # want of a raw material — digs the pit that supplies it, or else WAITS. It never
            # falls through to houses or spare food, because on a bounded plot that sprawl would
            # eat the very ground the plan's bakery, school and barracks still need. Population
            # growth comes after, on whatever land is left.
            Sequence(children=[
                Condition(plan_incomplete, "Settlement Unfinished?"),
                Selector(children=[
                    BTAction(try_build_next, "Build The Next Thing"),
                    BTAction(try_exploit_more, "Dig The Pit It Needs"),
                    BTAction(do_nothing, "Wait For It"),
                ], name="Build, Dig, Or Wait"),
            ], name="Development"),

            # THE SURPLUS BECOMES AN ARMY. With the essentials up, every working-age hand the
            # district holds beyond the jobs its economy offers is drafted and equipped — this
            # is above housing on purpose, so idle hands turn into soldiers rather than into an
            # ever-larger idle crowd. Training draws a man off, which reopens room to grow, so
            # the district oscillates: grow, muster the surplus, grow again.
            Sequence(children=[
                Condition(can_field_soldier, "A Spare Hand And A Weapon?"),
                BTAction(try_field_soldier, "Train & Arm A Soldier"),
            ], name="Muster"),

            # Otherwise the leftover plot grows the population: housing lets it rise, and a
            # little more food carries the larger numbers.
            Sequence(children=[
                Condition(needs_housing, "Needs Housing?"),
                Selector(children=[
                    BTAction(try_build_dormitory, "Build Dormitory"),
                    BTAction(try_build_house, "Build House"),
                ], name="Shelter"),
            ], name="Housing"),

            Sequence(children=[
                Condition(is_hungry, "Larder Thin?"),
                BTAction(try_build_food, "Build Food"),
            ], name="Food Margin"),

            BTAction(do_nothing, "Do Nothing"),
        ], name="SettlerBot Root")

    def act(self, game_state: GameState, game_time: float) -> Action:
        faction = self._get_faction(game_state)
        if not faction:
            return Action(faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)

        context = {
            'game_state': game_state,
            'faction': faction,
            'opponent': self._get_opponent(game_state),
            'faction_id': self.faction_id,
            'game_time': game_time,
            'chosen_action': None,
        }
        self.behavior_tree.tick(context)
        return context.get('chosen_action') or Action(
            faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)
