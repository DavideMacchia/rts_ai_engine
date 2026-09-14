"""
Aggressive difficulty bot - Military-focused rusher using Behavior Trees.

DEPRECATED as a training teacher. The current focus is the CIVIL district tier (D32),
whose teacher is the SettlerBot: it builds a settlement and never fights. This bot is
NOT used to train that tier. It is kept for two reasons only:
  - it drives the combat-oracle tests (tests/unit/test_conquest_and_march.py);
  - it is the starting point for the future MILITARY CAMP tier, which is a separate model.
Do not wire it back into the civil training path. When the camp tier is revived, its
assumptions must be re-checked against the D36 demographics (see design_decisions.md D36).

Strategy:
- Minimal economy (just enough to not starve and afford military)
- Rush barracks early
- Train soldiers continuously
- Attack the opponent whenever it has an army
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import ScriptedOpponent
from ..actions import Action, ActionType
from ..game_state import GameState, Faction

from ..policies import PopulationRate
from ..behavior_tree.nodes import Selector, Sequence, Condition, Action as BTAction
from ..behavior_tree.rts_helpers import (
    needs_housing,
    has_barracks,
    try_build_house,
    try_build_lumberyard,
    try_build_quarry,
    try_build_well,
    try_build_hunting_shed,
    try_build_vegetable_garden,
    try_build_school,
    try_build_barracks,
    try_build_carpentry,
    try_build_blacksmith,
    try_build_mine,
    try_build_clay_pit,
    try_build_pottery,
    try_build_smelter,
    try_build_stone_cutter,
    try_build_tailor,
    try_build_dormitory,
    try_build_cattle_shed,
    try_build_mill,
    try_build_bakery,
    try_train_soldier,
    try_train_man_at_arms,
    try_train_archer,
    try_attack_opponent,
    do_nothing,
)
from ..managers.building_manager import BuildingManager

# Minimal-economy targets: kept small, to avoid overspending before the military.
TARGET_LUMBERYARDS = 4
TARGET_QUARRIES = 2

# Food is hunted, not baked: a rusher cannot afford the 3-building farm->mill->bakery
# sequence before its starting stock runs out (and grain is not edible).
#
# THREE sheds, because a settlement must feed the people it makes whether or not it is at
# war. Fed by two, the district tops out at ~26 people — ~15 adults, against a core economy
# that asks for ~15 workers — so it can never spare the hand for the pottery that would give
# it bread, and bread is the only food that scales. Feed it first and it grows into the hands
# its own workshops need (D30).
TARGET_HUNTING_SHEDS = 3

# The district drinks every hour and the bakery bakes with water: the starting stock runs out
# mid-game. Water is not scarce, it just has to be DUG.
TARGET_WELLS = 1

# The development the district needs before it can man its own trades (D30): clay -> bricks
# gates the school, the bakery and the dormitory.
TARGET_CLAY_PITS = 1
TARGET_POTTERIES = 1
TARGET_MILLS = 1
TARGET_BAKERIES = 1
TARGET_STONE_CUTTERS = 1
TARGET_CATTLE_SHEDS = 1
TARGET_TAILORS = 1

# The armoury — a PREREQUISITE, not a workshop: an army must be equipped, so no carpentry
# means no spears, and no spears means no soldiers (D29).
TARGET_CARPENTRIES = 1

# A cheap second food, for the variety happiness bonus (D15).
TARGET_VEGETABLE_GARDENS = 1

# An adult teaches and the rest of the district gets skilled faster. Measured to pay (D21).
TARGET_SCHOOLS = 1

BARRACKS_WOOD_BUFFER = 60      # wood in hand before committing to barracks

from ..config import (
    ADULT_DURATION_SECONDS as _ADULT_LIFE,
    ATTACK_TRAVEL_TIME_SECONDS,
    DEFENDER_BONUS,
    UNIT_STRENGTH,
)

# Ramping aggression: don't rush-kill the agent before it can defend. Anchored to the time
# scale, not typed in as seconds, so the bot opens hostilities after roughly one generation
# whatever the episode lasts.
MIN_ATTACK_TIME = 1.1 * _ADULT_LIFE

# A floor on what is worth MARCHING at all — not the commit criterion, which is the arrival
# forecast below (D25). A difficulty knob: raising it makes a bot hoard before it moves, which
# is easier to face. At 0 the forecast has the only say, which is what the expert wants.
ATTACK_ARMY_STRENGTH = 0.0

# --- commit against the state we will MEET, not the one we leave (D25) ---
# An attack takes ATTACK_TRAVEL_TIME_SECONDS to land, so departure strength is stale. We
# commit on a forecast of both armies at ARRIVAL.
#
# The forecast counts only what is already PAID FOR — units in the barracks that will land
# inside the window. It deliberately does NOT credit the defender with units it might order
# during our march: that term was built and ablated (D25), and it moved not one decision,
# because the defender is chronically too broke to reinforce. Put it back only if the economy
# ever gets rich enough that a defender CAN raise an army inside a march.
#
# What we demand at arrival, as a multiple of the defender's bonus-adjusted strength. 1.0 =
# "I will win the battle". Demanding the full capture threshold (2.0) measured worse: waiting
# to be overwhelming means not attacking, and a rusher does not give you the time.
ATTACK_SAFETY_MARGIN = 1.0


# ---- Aggressive-specific conditions ----

def _count_with_progress(faction: Faction, building_type: str) -> int:
    """Count completed buildings PLUS those currently under construction.

    Without counting in-progress buildings, the bot spams build orders during
    the multi-step construction delay (in-progress buildings aren't yet counted).
    """
    completed = faction.get_building_count(building_type)
    in_progress = sum(
        1 for b in faction.buildings_in_progress
        if getattr(b.building_type, 'value', b.building_type) == building_type
    )
    return completed + in_progress


def has_attack_army(context: dict) -> bool:
    """Is an attack worth MARCHING at all? (Whether it is worth WINNING is the forecast's
    job — see `attack_wins_on_arrival`.)

    Two gates:
    - enough game time has passed: aggression ramps, so the agent is not one-shot at 300s
      while it still has no defence;
    - we have at least `attack_army_strength`. At the expert's setting (0) this is no gate
      at all and the forecast decides alone; on the easier bots it is the thing that makes
      them easy — they hoard an army instead of spending it.
    """
    faction: Faction = context['faction']
    game_time: float = context['game_time']
    min_time = context.get('min_attack_time', MIN_ATTACK_TIME)
    army_strength = context.get('attack_army_strength', ATTACK_ARMY_STRENGTH)
    return game_time >= min_time and faction.military_strength >= army_strength


def army_is_home(context: dict) -> bool:
    """False while our army is on the road.

    An army in transit cannot be re-committed, so ordering another attack is a no-op that
    COSTS us the decision. Failing this condition lets the tree fall through to training and
    economy, so the army grows while it walks (D26).
    """
    return not context['game_state'].is_marching(context['faction_id'])


def predicted_strength_at_arrival(faction: Faction, horizon: float) -> float:
    """Forecast a faction's military strength `horizon` seconds from now.

    Two terms:
      1. the army it has;
      2. units already IN TRAINING that finish inside the window. These are paid for, so
         they are near-certain — but each still consumes an adult man on completion, and
         men are the binding constraint, since only men enlist. A barracks queue longer
         than the district's men does not become an army.

    Both armies are forecast this way: ours grows while it marches (the march no longer
    eats our decisions, so we keep training), and so does theirs.
    """
    men = faction.adult_men
    strength = faction.military_strength

    landing = sorted(
        (u for u in faction.units_in_training if u.turns_remaining <= horizon),
        key=lambda u: u.turns_remaining,
    )
    for unit in landing:
        if men <= 0:
            break
        men -= 1
        strength += UNIT_STRENGTH.get(unit.unit_type.value, 1.0)

    return strength


def attack_wins_on_arrival(context: dict) -> bool:
    """Commit only if the army would still win the battle it will actually FIGHT.

    Both sides are forecast to the moment of arrival, and the defender keeps its
    DEFENDER_BONUS. Attacking into a predicted loss is not a gamble that sometimes pays: the
    loser takes 40% casualties, so a failed attack burns the army AND the men it was made of,
    and the next attack is further away than the last (D25).
    """
    faction: Faction = context['faction']
    opponent: Faction = context['opponent']
    if opponent is None:
        return False

    horizon = ATTACK_TRAVEL_TIME_SECONDS
    mine = predicted_strength_at_arrival(faction, horizon)
    theirs = predicted_strength_at_arrival(opponent, horizon)

    margin = context.get('attack_safety_margin', ATTACK_SAFETY_MARGIN)
    return mine > DEFENDER_BONUS * theirs * margin


def needs_min_economy(context: dict) -> bool:
    """The CORE: what a district must have before it can fight at all.

    Wood and stone to build with, two foods so it does not starve, and the carpentry —
    which since D29 is the armoury, and without which there are no soldiers.
    """
    faction: Faction = context['faction']
    return (
        _count_with_progress(faction, 'lumberyard') < TARGET_LUMBERYARDS or
        _count_with_progress(faction, 'quarry') < TARGET_QUARRIES or
        _count_with_progress(faction, 'hunting_shed') < TARGET_HUNTING_SHEDS or
        _count_with_progress(faction, 'well') < TARGET_WELLS or
        _count_with_progress(faction, 'vegetable_garden') < TARGET_VEGETABLE_GARDENS or
        _count_with_progress(faction, 'carpentry') < TARGET_CARPENTRIES or
        _count_with_progress(faction, 'school') < TARGET_SCHOOLS
    )


def needs_development(context: dict) -> bool:
    """The TRADES: clay to bricks, grain to bread, hide to cloth, and a school.

    These come AFTER the barracks in priority, not before. Bundled into the core, they
    starved the military entirely — the district spent the whole episode developing, never
    raised a soldier, and every game ended in a timeout. Development is what a district does
    with what is left over once it can defend itself.
    """
    faction: Faction = context['faction']
    return (
        _count_with_progress(faction, 'clay_pit') < TARGET_CLAY_PITS or
        _count_with_progress(faction, 'pottery') < TARGET_POTTERIES or
        _count_with_progress(faction, 'mill') < TARGET_MILLS or
        _count_with_progress(faction, 'bakery') < TARGET_BAKERIES or
        _count_with_progress(faction, 'stone_cutter') < TARGET_STONE_CUTTERS or
        _count_with_progress(faction, 'cattle_shed') < TARGET_CATTLE_SHEDS or
        _count_with_progress(faction, 'tailor') < TARGET_TAILORS or
        _count_with_progress(faction, 'school') < TARGET_SCHOOLS
    )


def try_develop(context: dict):
    """Build the next trade the district is missing, in the order the materials flow."""
    faction: Faction = context['faction']
    if _count_with_progress(faction, 'clay_pit') < TARGET_CLAY_PITS:
        return try_build_clay_pit(context)
    if _count_with_progress(faction, 'pottery') < TARGET_POTTERIES:
        return try_build_pottery(context)
    if _count_with_progress(faction, 'mill') < TARGET_MILLS:
        return try_build_mill(context)
    if _count_with_progress(faction, 'bakery') < TARGET_BAKERIES:
        return try_build_bakery(context)
    if _count_with_progress(faction, 'stone_cutter') < TARGET_STONE_CUTTERS:
        return try_build_stone_cutter(context)
    if _count_with_progress(faction, 'cattle_shed') < TARGET_CATTLE_SHEDS:
        return try_build_cattle_shed(context)
    if _count_with_progress(faction, 'tailor') < TARGET_TAILORS:
        return try_build_tailor(context)
    if _count_with_progress(faction, 'school') < TARGET_SCHOOLS:
        return try_build_school(context)
    return None


# ---- the iron road (D29) ----

def has_iron_in_reach(context: dict) -> bool:
    """Does this district's own ground hold an iron vein it could mine?

    This is the first question in the game whose answer is a fact about the MAP. It is what
    a plot of land is finally worth something FOR: iron does not buy more soldiers — the
    district only has four or five men to spare either way — it makes each of those men
    nearly twice the soldier (D22, D29).
    """
    faction: Faction = context['faction']
    # A mine we already own IS iron in reach. Asking only "is there a free vein site?" made
    # the answer flip to False the moment the mine was built on the plot's only vein — the
    # bot dug the mine and then abandoned the road, never smelting an ounce of what it mined.
    if _count_with_progress(faction, 'mine') > 0:
        return True
    tile, _q = BuildingManager.find_site(faction.capital, 'mine')
    return tile is not None


def needs_iron_chain(context: dict) -> bool:
    """mine -> blacksmith. Two buildings, two workers — and that is already dear when the
    district has five adults to its name."""
    faction: Faction = context['faction']
    return (
        _count_with_progress(faction, 'mine') < 1 or
        _count_with_progress(faction, 'smelter') < 1 or
        _count_with_progress(faction, 'blacksmith') < 1
    )


def try_build_iron_chain(context: dict):
    """Build the next missing link, in the order the ore actually travels."""
    faction: Faction = context['faction']
    if _count_with_progress(faction, 'mine') < 1:
        return try_build_mine(context)
    if _count_with_progress(faction, 'smelter') < 1:
        return try_build_smelter(context)
    if _count_with_progress(faction, 'blacksmith') < 1:
        return try_build_blacksmith(context)
    return None


def try_build_min_economy(context: dict):
    """Build the next missing minimal-economy building.

    Wood and stone first (needed to build anything), then a hunting shed so the
    population has food, then a vegetable garden for a second food (variety bonus).

    """
    faction: Faction = context['faction']
    if _count_with_progress(faction, 'lumberyard') < TARGET_LUMBERYARDS:
        return try_build_lumberyard(context)
    if _count_with_progress(faction, 'quarry') < TARGET_QUARRIES:
        return try_build_quarry(context)
    if _count_with_progress(faction, 'hunting_shed') < TARGET_HUNTING_SHEDS:
        return try_build_hunting_shed(context)
    if _count_with_progress(faction, 'well') < TARGET_WELLS:
        return try_build_well(context)
    if _count_with_progress(faction, 'vegetable_garden') < TARGET_VEGETABLE_GARDENS:
        return try_build_vegetable_garden(context)
    if _count_with_progress(faction, 'carpentry') < TARGET_CARPENTRIES:
        return try_build_carpentry(context)
    # The SCHOOL is core, not a luxury: the skilled trades are gated on SKILL, not on hands
    # (the potter needs a miner's, the baker a miller's — see professions.py). Twenty idle
    # adults and no school still cannot man a pottery.
    if _count_with_progress(faction, 'school') < TARGET_SCHOOLS:
        return try_build_school(context)
    return None


def economy_ready_for_barracks(context: dict) -> bool:
    """Minimal economy is in place and we have a small wood buffer for barracks + units."""
    faction: Faction = context['faction']
    return (not needs_min_economy(context)) and faction.get_resource('wood') >= BARRACKS_WOOD_BUFFER


def needs_barracks(context: dict) -> bool:
    """No barracks built or under construction yet."""
    faction: Faction = context['faction']
    return _count_with_progress(faction, 'barracks') == 0


# ---- policy lever (docs/population_and_policies.md) ----

def needs_pro_natalist_policy(context: dict) -> bool:
    """Set the pro-natalist stance once, on turn one.

    More sims -> more workers -> more soldiers, and the incentive is free (L2, not coercion).
    Ablating this one lever from the trained BC policy collapses its win rate (D15).

    Its value lives in the COMPOSITION, not in the lever: forcing it on the scripted expert
    alone is neutral-to-negative, because the script does not re-tune its build and attack
    timing around the extra mouths, whereas a learned policy does.
    """
    faction: Faction = context['faction']
    return faction.capital.policies.population_rate is not PopulationRate.INCREASE


def try_set_pro_natalist_policy(context: dict):
    context['chosen_action'] = Action(
        faction_id=context['faction_id'],
        action_type=ActionType.SET_POPULATION_RATE_INCREASE,
    )
    return context['chosen_action']


class AggressiveBot(ScriptedOpponent):
    """Military rusher: minimal economy, early barracks, constant soldiers, attacks."""

    def __init__(self, faction_id: int,
                 min_attack_time: float = MIN_ATTACK_TIME,
                 attack_army_strength: float = ATTACK_ARMY_STRENGTH,
                 attack_safety_margin: float = ATTACK_SAFETY_MARGIN):
        super().__init__(faction_id)
        self.name = "AggressiveBot"
        self.difficulty = "aggressive"
        # Difficulty knobs: later attacks + bigger army required = easier for the agent
        self.min_attack_time = min_attack_time
        self.attack_army_strength = attack_army_strength
        # What the arrival forecast must promise before we commit (exposed so it can be
        # ablated rather than assumed — see D25)
        self.attack_safety_margin = attack_safety_margin
        self.behavior_tree = self._build_tree()

    def _build_tree(self):
        return Selector(
            children=[
                # 0. Set the pro-natalist stance once, on turn one. See D15: this single
                # lever, cloned into the BC policy, is worth ~20 conquests out of 40 vs
                # the aggressive bots. The bot conquers in ~3 generations, so the extra
                # workers matter through faster army rebuilds, not long-run demography.
                Sequence(
                    children=[
                        Condition(needs_pro_natalist_policy, "Not Pro-Natalist Yet?"),
                        BTAction(try_set_pro_natalist_policy, "Encourage Births"),
                    ],
                    name="Population Policy"
                ),

                # 1. ATTACK — but only an army that is home, big enough to be worth
                # sending, and forecast to WIN the fight it will find on arrival.
                # If any of those fails we fall through and keep building the army:
                # doing nothing is not the alternative to attacking, growing is.
                Sequence(
                    children=[
                        Condition(army_is_home, "Army At Home?"),
                        Condition(has_attack_army, "Have Army?"),
                        Condition(attack_wins_on_arrival, "Wins On Arrival?"),
                        BTAction(try_attack_opponent, "Attack Opponent"),
                    ],
                    name="Attack Branch"
                ),

                # 2. Minimal economy FIRST (sustainable wood/stone/food before military)
                Sequence(
                    children=[
                        Condition(needs_min_economy, "Needs Minimal Economy?"),
                        BTAction(try_build_min_economy, "Build Minimal Economy"),
                    ],
                    name="Minimal Economy"
                ),

                # 3. Build barracks once economy + wood buffer are ready
                Sequence(
                    children=[
                        Condition(needs_barracks, "No Barracks (incl. in-progress)?"),
                        Condition(economy_ready_for_barracks, "Economy Ready?"),
                        BTAction(try_build_barracks, "Build Barracks"),
                    ],
                    name="Barracks"
                ),

                # 4. THE IRON ROAD (D29). Only where the ground holds a vein — this is the
                # first decision in the game that the MAP answers. Three buildings for a
                # better kit, and the kit is what turns four spare men into a real army.
                Sequence(
                    children=[
                        Condition(has_iron_in_reach, "Iron In Our Ground?"),
                        Condition(needs_iron_chain, "Iron Chain Incomplete?"),
                        BTAction(try_build_iron_chain, "Build Mine/Blacksmith"),
                    ],
                    name="Iron Road"
                ),

                # 5. DEVELOPMENT: the trades. This node MUST sit above training and below the
                # barracks, and both halves are load-bearing. Put it below training and it
                # never fires — training is nearly always affordable, so the tree never falls
                # through and the workshops are never built. Put it above the barracks and the
                # district develops beautifully, then loses to someone who did not.
                Sequence(
                    children=[
                        Condition(needs_development, "Trades Missing?"),
                        BTAction(try_develop, "Develop A Trade"),
                    ],
                    name="Development"
                ),

                # 6. Train. Men-at-arms FIRST whenever the forge has kit for them: men are
                # what we run out of, so every man we spend should be the best soldier we
                # can make of him. A spearman is what we settle for.
                Sequence(
                    children=[
                        Condition(has_barracks, "Has Barracks?"),
                        Selector(
                            children=[
                                BTAction(try_train_man_at_arms, "Train Man-at-Arms"),
                                BTAction(try_train_soldier, "Train Soldier"),
                                BTAction(try_train_archer, "Train Archer"),
                            ],
                            name="Unit Training"
                        ),
                    ],
                    name="Military Training"
                ),

                # 7. Housing. A dormitory holds twice a family house, and the population is
                # now the thing the district is short of — every empty workshop is a person
                # it never housed.
                Sequence(
                    children=[
                        Condition(needs_housing, "Needs Housing?"),
                        Selector(children=[
                            BTAction(try_build_dormitory, "Build Dormitory"),
                            BTAction(try_build_house, "Build House"),
                        ], name="Shelter"),
                    ],
                    name="Housing"
                ),

                # 6. Fallback: idle while wood accumulates for the next unit
                BTAction(do_nothing, "Do Nothing"),
            ],
            name="AggressiveBot Root"
        )

    def act(self, game_state: GameState, game_time: float) -> Action:
        faction = self._get_faction(game_state)
        opponent = self._get_opponent(game_state)

        if not faction:
            return Action(faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)

        context = {
            'game_state': game_state,
            'faction': faction,
            'opponent': opponent,
            'faction_id': self.faction_id,
            'game_time': game_time,
            'min_attack_time': self.min_attack_time,
            'attack_army_strength': self.attack_army_strength,
            'attack_safety_margin': self.attack_safety_margin,
            'chosen_action': None,
        }

        self.behavior_tree.tick(context)

        action = context.get('chosen_action')
        if action is None:
            action = Action(faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)

        return action
