"""
Game configuration for the ML simulator.

Loaded from game_data/ (shared with the Rust game): building costs, build times,
starting resources, unit training data.
Authored here: the time scale, the map, the production chains, and everything derived
from them.

The discipline of this file is: pick an anchor, state a ratio, derive the number.
Never hand-tune a derived number. See docs/design_decisions.md (D13 time, D27 map,
D29 war economy, D30 production circles, D33 district).
"""

from .game_data_loader import (
    load_building_costs,
    load_starting_resources,
    load_building_build_times,
    load_unit_training_data,
    load_production_recipes,      # not used for values — only to guard against drift, below
)
from .professions import SKILL_FLOOR as _NOVICE

# ============================================================================
# TIME SCALE — one anchor, everything else derived (D13)
# ============================================================================
# The simulator is real-time: state advances by `delta_time` seconds, not by turns.
# Buildings must go up fast compared to a life, resource output must be proportioned
# to what a person eats and to what a building costs, and an episode must span several
# generations so the demographic levers have time to pay off — or to present their bill.
# The numbers are uncalibrated (there is no finished game to fit them to), but they are
# ratios, and the ratios are stated.

ADULT_DURATION_SECONDS = 900.0          # THE ANCHOR: mean adult working life

# Demography, as fractions of a working life.
KID_DURATION_SECONDS = 0.25 * ADULT_DURATION_SECONDS
ELDER_DURATION_SECONDS = 0.50 * ADULT_DURATION_SECONDS

# Decision count is the RL budget; game time is the world's clock. They are not
# independent: push generations up while holding decisions fixed and construction
# collapses into a single step — an action that resolves instantly is a purchase, not a
# decision. Both move together, and the invariants at the bottom of this file assert it.
GENERATIONS_PER_EPISODE = 10.0
EPISODE_SECONDS = GENERATIONS_PER_EPISODE * ADULT_DURATION_SECONDS        # 9000 s
DECISIONS_PER_EPISODE = 1440
DECISION_INTERVAL_SECONDS = EPISODE_SECONDS / DECISIONS_PER_EPISODE       # 6.25 s

# A standard building is 5% of a working life. The relative structure of the loaded build
# times is the game designer's; we rescale it as a whole, anchoring on the standard
# building (the farm, 120 s in game_data).
_STANDARD_BUILDING_FRACTION = 0.05
_BUILD_TIME_SCALE = (_STANDARD_BUILDING_FRACTION * ADULT_DURATION_SECONDS) / 120.0

# Vital rates are stated per working life, then converted.
_PER_LIFE_TO_PER_HOUR = 3600.0 / ADULT_DURATION_SECONDS

BIRTHS_PER_LIFE = 7.0                  # fast enough to reach TARGET_DISTRICT_POPULATION
BIRTH_CHANCE_PER_TURN = BIRTHS_PER_LIFE * _PER_LIFE_TO_PER_HOUR

CONSCRIPTS_PER_LIFE = 8.0              # a full mobilisation takes a good part of a life
CONSCRIPTION_RATE_PER_HOUR = CONSCRIPTS_PER_LIFE * _PER_LIFE_TO_PER_HOUR

STARVATION_FRACTION_PER_LIFE = 2.0     # total famine empties a district well inside a life
STARVATION_DEATH_RATE = STARVATION_FRACTION_PER_LIFE * _PER_LIFE_TO_PER_HOUR

PREGNANCY_DURATION_SECONDS = 0.1 * ADULT_DURATION_SECONDS
PREGNANCY_PRODUCTIVITY = 0.5      # a pregnant woman works at half output

# An attack is not instantaneous: the army must MARCH, and it is GONE from home while it
# does (D26). This is the fallback for a game played with no map; with a map the march is
# derived from the distance actually walked (MARCH_SECONDS_PER_TILE).
ATTACK_TRAVEL_TIME_SECONDS = 0.1 * ADULT_DURATION_SECONDS


# ============================================================================
# THE MAP (D27)
# ============================================================================
# The game's map is 300x300; the training map is smaller, and nothing here depends on the
# tile COUNT — only on the ratios. Edit the ratios; the seconds are derived.
MAP_WIDTH = 96
MAP_HEIGHT = 96

#: A district owns the tiles within this radius of its warehouse, and everything it builds
#: must fit in that plot — so the plot is a real constraint on how big it can get. Sized so a
#: fully-developed district (the whole workshop chain, the bread chain, the iron kit, a
#: barracks) plus houses and a standing army all FIT with room to spare, since buildings now
#: occupy rectangular footprints (2x2 to 3x3): at radius 6 the 3x3 compounds could not all be
#: seated, at radius 8 they can. This is the plot's carrying capacity — the real limit on how
#: large a district grows, in place of any population cap.
DISTRICT_RADIUS = 8

#: How far apart rival settlements are founded. Anchored to the district radius: rivals sit
#: ~3 plots apart — close enough that war is live from the start, far enough that neither
#: is inside the other's territory.
SETTLEMENT_SEPARATION = 3.5 * DISTRICT_RADIUS      # ~21 tiles

#: Calibrated so the DEFAULT separation reproduces ATTACK_TRAVEL_TIME_SECONDS: introducing
#: the map must not silently re-balance the game. Tune this ratio, never the seconds.
MARCH_SECONDS_PER_TILE = ATTACK_TRAVEL_TIME_SECONDS / SETTLEMENT_SEPARATION

#: Fraction of eligible tiles carrying a vein. Deposits are what make one plot worth more
#: than another: too many and the choice is free, too few and it is a lottery.
DEPOSIT_DENSITY = 0.25

# Elevation model, mirrored from game_data/map.json. The game's sea_level is 0.0 on
# unnormalised noise; here the noise is normalised to [0, ELEVATION_MAX], so water is the
# bottom fifth of the range — lakes and coasts, not an ocean.
ELEVATION_MAX = 10.0
SEA_LEVEL = 2.0
MOUNTAIN_THRESHOLD = 6.5


# ============================================================================
# THE DISTRICT, ANCHORED (D33)
# ============================================================================
# Food, housing and the workforce are all derived from one target. Tuned one at a time they
# did not add up: a full settlement asked for ~26 worker slots and the district supplied ~18
# adults, a labour deficit no build order can solve.

#: THE ANCHOR: what a mature district IS.
TARGET_DISTRICT_POPULATION = 60

#: Of those, the share that are working-age adults (the rest are children and elders).
ADULT_SHARE = 0.45
TARGET_ADULT_WORKERS = TARGET_DISTRICT_POPULATION * ADULT_SHARE      # ~27 hands

#: Feeding itself must not cost the district more than this share of those hands. Above a
#: fifth is a settlement that exists to eat, with nothing left to become anything.
FOOD_LABOUR_SHARE = 0.20
_FOOD_WORKERS = TARGET_ADULT_WORKERS * FOOD_LABOUR_SHARE            # ~5.4 hands on food

#: How many mouths one worker on food must feed. Computed at NOVICE skill on purpose: a
#: fresh settlement is staffed by novices, and a food supply that only works once everyone
#: is a master is one that starves the district through its first generation.
PEOPLE_FED_PER_FOOD_WORKER = TARGET_DISTRICT_POPULATION / _FOOD_WORKERS / _NOVICE   # ~18

# An elder still works, less well. Mean of person.json's elder_productivity table over the
# civilian professions. Dimensionless, so it needs no anchor.
ELDER_PRODUCTIVITY = 0.45


# ============================================================================
# WORKERS
# ============================================================================
# WORKERS_NEEDED is the MAXIMUM crew, at which a building runs at 100%. WORKERS_MIN is the
# smallest crew that can run it at all. Between the two, output is proportional. A district
# mans all its buildings thinly before fattening any crew (D35) — filling to the maximum in
# priority order emptied the pool before it reached the bottom of the list, so a settlement
# in a forest could run out of wood.
WORKERS_NEEDED = {
    'farm': 2,
    'mill': 1,
    'bakery': 1,
    'lumberyard': 2,
    'quarry': 2,
    'hunting_shed': 1,
    'cattle_shed': 2,
    'pottery': 1,
    'carpentry': 1,
    'stone_cutter': 1,
    'blacksmith': 1,
    'vegetable_garden': 1,
    'orchard': 1,
    'mine': 1,
    'clay_pit': 1,
    'well': 1,
    'smelter': 1,
    'tailor': 1,
}

WORKERS_MIN = {building: 1 for building in WORKERS_NEEDED}

# Lower number = staffed first when hands are short.
BUILDING_PRIORITIES = {
    'farm': 1,
    'hunting_shed': 1,
    'quarry': 2,
    'lumberyard': 3,
    'clay_pit': 3,
    'mine': 5,            # the iron road is an INVESTMENT: staffed only once the
    'well': 3,            # district's own needs are covered
    'mill': 4,
    'bakery': 4,
    'pottery': 3,         # bricks gate the buildings the district needs
    'carpentry': 2,       # the armoury: no spears, no army (D29)
    'stone_cutter': 4,
    'blacksmith': 5,
    'cattle_shed': 4,
    'smelter': 5,
    'tailor': 6,
}


# ============================================================================
# PRODUCTION
# ============================================================================
# A production building must repay its cost in a fraction of a life: a lumberyard costs
# 15 wood and must fund several buildings per generation, not one.
PRODUCTION_SCALE = 5.0
_FARM_GRAIN = 90.0 * PRODUCTION_SCALE

# The bread chain is farm(2) + mill(1) + bakery(1) = 4 hands, so it must feed four workers'
# worth. This sets what a person eats; every other food is a multiple of THAT, so the whole
# table moves together and cannot drift apart again.
_BREAD_CHAIN_WORKERS = WORKERS_NEEDED['farm'] + WORKERS_NEEDED['mill'] + WORKERS_NEEDED['bakery']
PEOPLE_FED_PER_FARM = PEOPLE_FED_PER_FOOD_WORKER * _BREAD_CHAIN_WORKERS
POPULATION_FOOD_CONSUMPTION = _FARM_GRAIN / PEOPLE_FED_PER_FARM      # per person per hour
POPULATION_WATER_CONSUMPTION = 0.6 * POPULATION_FOOD_CONSUMPTION

# Raw producers only — a building here takes no inputs. Processing buildings live in
# RESOURCE_PROCESSING, and the invariant at the bottom forbids a building from being both.
# The extractive rates are the designer's, scaled; the food rates are derived from the
# anchor above, so the garden and the orchard are stated in mouths fed, not in tokens.
PRODUCTION_RATES = {
    'farm': {'grain': _FARM_GRAIN},
    'hunting_shed': {'meat': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER},
    'lumberyard': {'wood': 60.0 * PRODUCTION_SCALE},
    'quarry': {'stone': 30.0 * PRODUCTION_SCALE},
    'clay_pit': {'clay': 30.0 * PRODUCTION_SCALE},
    'mine': {'iron_ore': 15.0 * PRODUCTION_SCALE},
    'well': {'water': 150.0 * PRODUCTION_SCALE},
    # The garden is quicker and cheaper than a shed and feeds fewer; the orchard is a luxury
    # whose real worth is being a FOURTH food on the table (the variety bonus, D18), not its
    # calories.
    'vegetable_garden': {'vegetable': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER * 0.65},
    'orchard': {'fruit': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER * 0.45},
}

# --- the production circles, closed (D30) -----------------------------------
# Recipes are authored HERE, not in game_data/resource.json: every recipe in that file was
# either overridden below or dropped, so loading it was a silent no-op.
#
# Rates are per hour. Every line is a closed circle — something produces it, something
# consumes it. Open lines (produce with no consumer, or require with no producer) are what
# made whole trades unfillable: gold was mined and unspendable, iron was smelted and used by
# nobody, and the bakery cost clay bricks that no building made.
_CHAIN_STAGE_EFFICIENCY = 0.9
_WOOD_PER_WEAPON = 10.0            # what a soldier used to cost directly, in wood
_WOODEN_WEAPONS_PER_HOUR = 20.0
_IRON_KITS_PER_HOUR = 6.0

RESOURCE_PROCESSING = {
    # GRAIN -> FLOUR -> BREAD: the long chain, and the only food that scales.
    'mill': {
        'input': {'grain': _FARM_GRAIN},
        'output': {'flour': _FARM_GRAIN * _CHAIN_STAGE_EFFICIENCY},
    },
    'bakery': {
        'input': {'flour': _FARM_GRAIN * _CHAIN_STAGE_EFFICIENCY, 'water': 0.1 * _FARM_GRAIN},
        'output': {'bread': _FARM_GRAIN * _CHAIN_STAGE_EFFICIENCY ** 2},
    },
    # WOOD -> timber for building, and spears for the militia.
    'carpentry': {
        'input': {'wood': _WOOD_PER_WEAPON * _WOODEN_WEAPONS_PER_HOUR + 120.0},
        'output': {'wooden_weapon': _WOODEN_WEAPONS_PER_HOUR, 'wood_logs': 60.0},
    },
    # CLAY -> BRICKS: the line the D30 audit turned on. The bakery costs bricks, so without
    # a potter, bread — the only food that scales — is unreachable.
    'pottery': {
        'input': {'clay': 120.0},
        'output': {'clay_bricks': 60.0},
    },
    'stone_cutter': {
        'input': {'stone': 120.0},
        'output': {'stone_bricks': 60.0},
    },
    # CATTLE -> meat, and the two textiles. Slower per grain than bread, but it yields meat
    # (variety) and leather (armour) without a mill.
    'cattle_shed': {
        'input': {'grain': 450.0},
        'output': {'meat': 225.0, 'leather': 45.0, 'wool': 45.0},
    },
    # IRON: mine -> smelter -> blacksmith, the three careers the skill table describes.
    'smelter': {
        'input': {'iron_ore': 60.0},
        'output': {'iron_ingots': 30.0},
    },
    'blacksmith': {
        'input': {'iron_ingots': 4.0 * _IRON_KITS_PER_HOUR, 'wood': 2.0 * _IRON_KITS_PER_HOUR},
        'output': {'iron_weapon': _IRON_KITS_PER_HOUR, 'armour': _IRON_KITS_PER_HOUR},
    },
    # WOOL + LEATHER -> CLOTHES: the one derivative the PEOPLE consume rather than the army.
    'tailor': {
        'input': {'wool': 30.0, 'leather': 30.0},
        'output': {'clothes': 20.0},
    },
}


# ============================================================================
# BUILDINGS
# ============================================================================
_BASE_BUILDING_COSTS = load_building_costs()

# Costs not in the JSON, plus the ones D30 re-priced so that buildings COST what the
# workshops make — this is what makes the trades necessary rather than optional. Each was
# chosen so a district can still open on wood and stone alone, and must develop to go on.
_ADDITIONAL_COSTS = {
    'lumberyard': {'wood': 15, 'stone': 5},
    'quarry': {'wood': 15, 'stone': 5},
    'clay_pit': {'wood': 15, 'clay': 10},
    'mine': {'wood': 25, 'stone': 20},
    'well': {'wood': 10, 'stone': 15},
    'hunting_shed': {'wood': 20, 'stone': 10},
    'cattle_shed': {'wood': 30, 'stone': 15},
    'house': {'wood': 25, 'stone': 15},
    'bakery': {'wood': 60, 'stone': 30, 'clay_bricks': 15},
    'barracks': {'wood': 40, 'stone': 30, 'wood_logs': 10},
    'blacksmith': {'wood': 35, 'stone': 30, 'stone_bricks': 10},
    'dormitory': {'wood': 30, 'stone': 20, 'clay_bricks': 10},
    'defensive_wall_wood': {'wood_logs': 20},
    'defensive_wall_stone': {'stone_bricks': 40},
    'watchtower': {'wood': 25, 'stone_bricks': 15},
    # The school costs no bricks, and that is not a discount — it is the bootstrap. The
    # higher trades are gated on SKILL (the potter needs a miner's, the baker a miller's)
    # and the school is what transmits it. Charge it in bricks and the circle has no way in:
    # school needs bricks -> bricks need a potter -> the potter needs what the school teaches.
    'school': {'wood': 30, 'stone': 20},
    'vegetable_garden': {'wood': 15},
    'orchard': {'wood': 20, 'stone': 5},
    'smelter': {'wood': 40, 'stone': 30},
    'tailor': {'wood': 30, 'clay_bricks': 5},
}

BUILDING_COSTS = {**_BASE_BUILDING_COSTS, **_ADDITIONAL_COSTS}

# Loaded in seconds, held in MINUTES, and rescaled onto the time anchor.
BUILDING_BUILD_TIME = {
    building: (seconds / 60.0) * _BUILD_TIME_SCALE
    for building, seconds in load_building_build_times().items()
}
BUILDING_BUILD_TIME['vegetable_garden'] = 0.03 * ADULT_DURATION_SECONDS / 60.0   # cheap, fast
BUILDING_BUILD_TIME['orchard'] = 0.08 * ADULT_DURATION_SECONDS / 60.0            # a slow luxury
BUILDING_BUILD_TIME['smelter'] = 0.05 * ADULT_DURATION_SECONDS / 60.0
BUILDING_BUILD_TIME['tailor'] = 0.05 * ADULT_DURATION_SECONDS / 60.0


# ============================================================================
# UNITS AND THE WAR ECONOMY (D29)
# ============================================================================
# An army is not conjured out of wood and grain: it is EQUIPPED, and a soldier cannot exist
# without his kit. The binding constraint on an army is not throughput but adult MEN, and a
# district has few. So the question that decides a war is not "how many soldiers per hour"
# but "how much army do I get out of each man I can spare" — iron does not give you more
# soldiers, it makes the few you have nearly twice the soldier. That is what a plot with an
# iron vein is worth, and why WHERE you settle can change who wins.
_UNIT_DATA = load_unit_training_data()

UNIT_TRAINING_COST = _UNIT_DATA['costs']
UNIT_STRENGTH = _UNIT_DATA['strength']
UNIT_TRAINING_TIME = {
    unit: (seconds / 60.0) * _BUILD_TIME_SCALE      # loaded in seconds, held in minutes
    for unit, seconds in _UNIT_DATA['times'].items()
}

#: The two kits, as a RATIO — never hand-tune the absolute strength.
_IRON_KIT_ADVANTAGE = 1.8      # a man in mail with an iron sword, against a man with a spear

# The wood that used to sit in the soldier's price tag now sits in his weapon: the district
# pays the same wood per soldier, one step earlier, at the carpenter's.
UNIT_TRAINING_COST['soldier'] = {'grain': 5, 'wooden_weapon': 1}

# The same man, better equipped, and therefore a different soldier.
UNIT_TRAINING_COST['man_at_arms'] = {'grain': 5, 'iron_weapon': 1, 'armour': 1}
UNIT_STRENGTH['man_at_arms'] = UNIT_STRENGTH['soldier'] * _IRON_KIT_ADVANTAGE
UNIT_TRAINING_TIME['man_at_arms'] = UNIT_TRAINING_TIME['soldier']


# ============================================================================
# COMBAT
# ============================================================================
DEFENDER_BONUS = 1.3
COMBAT_CASUALTIES_RATE = 0.4           # the loser's losses
WINNER_CASUALTIES_RATE = 0.1
BUILDING_DAMAGE_CHANCE = 0.3
WAREHOUSE_DESTRUCTION_THRESHOLD = 2.0  # strength ratio needed to destroy a warehouse

# The only victory is conquest: outlast every rival's last warehouse. Running out of time is
# a truncation, not a scored win (D14).


# ============================================================================
# FOOD, HAPPINESS AND POLICY
# ============================================================================
# Grain and flour are intermediates: nobody eats them. The population lives on the END
# products, which is what makes the processing chains necessary rather than a dominated
# curiosity. Fish is deferred to Stage 2 — it needs water bodies (D16).
EDIBLE_FOODS = ['bread', 'meat', 'vegetable', 'fruit']

# Variety is TIERED: each distinct food beyond the first adds a bonus, capped. This is what
# makes the orchard — poor calories, but a distinct food — worth building.
MIN_FOOD_TYPES_FOR_VARIETY = 2
FOOD_VARIETY_BONUS_PER_TYPE = 5.0
FOOD_VARIETY_BONUS_CAP = 15.0          # 4 foods -> +15, leaving room for a 5th (fish)

# Clothes wear out at this rate per person per hour, and a clothed district is a happier one.
# A real sink: without one the tailor would be another dead end.
CLOTHES_CONSUMPTION_PER_CAPITA = 0.15
CLOTHES_HAPPINESS_BONUS = 8.0

MIN_FOOD_PER_CAPITA_FOR_GROWTH = 2.0
MIN_HAPPINESS_FOR_GROWTH = 50.0
STARVATION_HAPPINESS_LOSS = 10.0       # per hour starving

# Unemployment malus (D37): the happiness lost at FULL unemployment, scaled by the actual idle
# share. This is the feedback that couples population to the DEMAND FOR LABOUR — grow past the
# jobs, unemployment rises, happiness falls under MIN_HAPPINESS_FOR_GROWTH, births slow. It must
# DOMINATE the happiness bonuses (varied diet +15, clothes +8): otherwise a well-clothed,
# well-fed district stays happy while half its people sit idle, and nothing checks growth.
# Tuned on measurement (16 seeds): 50 was too weak (pop ran to ~120, 60% idle, wild boom-bust);
# 150 lands the mature district near full employment (~16% idle, the physiological residue of
# generational turnover) and halves the population oscillation. Re-verify alongside the dynamic
# build plan
# (D38), which raises the jobs the population settles against.
UNEMPLOYMENT_HAPPINESS_PENALTY = 150.0

#: Happiness of an un-governed district with food on the shelves. Everything else is a
#: penalty subtracted from it, exactly as the game does (happiness only ever decreases).
BASE_HAPPINESS = 75.0

# The four policy levers (see docs/population_and_policies.md).
POLICY_EFFECTS = {
    'minimum_food': -15.0,
    'siege_mode': -20.0,
    'forced_conscription': -25.0,
    'block_families': -10.0,           # military camp
    'cant_leave': -15.0,               # military camp
}

# INVARIANT, the calibration that makes coercion bite:
#   BASE_HAPPINESS + POLICY_EFFECTS['forced_conscription'] == MIN_HAPPINESS_FOR_GROWTH
# Drafting every adult drives fertility to exactly zero. Preserve this relation if you retune.

#: L1 minimum_food is a ration, not a ban: strictly in (0, 1).
MINIMUM_FOOD_RATION_FACTOR = 0.6

#: How hard unhappiness bites fertility:
#:   FLOOR + (1 - FLOOR) * clamp((happiness - MIN_HAPPINESS_FOR_GROWTH) / span)
#: so it is 1.0 at BASE_HAPPINESS regardless of the floor, and bottoms out at FLOOR, not 0.
#: Happiness is meant to be the MAIN gate on births now (D37), so the floor is low: an unhappy
#: district (from unemployment, coercion or want) breeds far slower, and only recovers when its
#: people are content again. Not zero — a hard zero makes unhappiness an absorbing state that
#: wrecks RL exploration, and only starvation and prohibition should take fertility to exactly 0.
HAPPINESS_FERTILITY_FLOOR = 0.15

#: After a birth, a woman cannot conceive again for this long — a spacing between children, not
#: a back-to-back reproduction. Anchored to the working life (like PREGNANCY_DURATION), so it
#: scales with the time scale rather than being a bare number of seconds.
FERTILITY_COOLDOWN_SECONDS = 0.15 * ADULT_DURATION_SECONDS

#: High parity lowers fertility. The first two children come at the full rate; from the third
#: on (a woman who already has 2), each further conception is multiplied by this factor once
#: more — 1.0, 1.0, DECAY, DECAY², … — so large families get progressively rarer instead of a
#: few women carrying the whole district. (No kinship model yet, so the woman stands in for the
#: couple; see D37.)
HIGH_PARITY_FERTILITY_DECAY = 0.6
HIGH_PARITY_THRESHOLD = 2               # children already born, above which decay kicks in

# L4 forced_conscription drafts adults deterministically (no RNG, so enabling it cannot
# perturb any other stochastic draw). Its rate is CONSCRIPTION_RATE_PER_HOUR, above: a lever
# must be sized against the episode, or it is not a lever.


# ============================================================================
# STARTING CONDITIONS
# ============================================================================
STARTING_RESOURCES = load_starting_resources()

# A district begins with a few spears in the rack. Without them the first soldier waits on a
# carpentry that waits on a lumberyard, and the opening is decided by construction order
# rather than by strategy.
STARTING_RESOURCES['wooden_weapon'] = 4

STARTING_BUILDINGS = {
    'warehouse': 1,      # if it is destroyed, you lose
    'house': 4,
    'farm': 1,           # or the founding population starves before it can build one
}

# Housing is sized so TARGET_DISTRICT_POPULATION fits in a plausible number of buildings, on
# a plot that has room for them. A district capped on beds stops growing, and then it never
# reaches the workforce its own workshops need.
STARTING_POPULATION = 20
STARTING_POPULATION_CAPACITY = 24
POPULATION_CAPACITY_PER_HOUSE = 8
POPULATION_CAPACITY_PER_DORMITORY = 16

# Edible stock to eat from while the first food buildings go up. Grain alone would starve
# them — it is not food.
#
# Sized in LIVES of eating, so it moves with the time scale: two adult working lives' worth of
# food per founding head, split evenly between the two starting foods. That is the runway the
# district has to get a hunting shed up before it is eating into nothing.
_FOUNDING_FOOD_LIVES = 2.0
STARTING_FOOD_PER_CAPITA = (
    POPULATION_FOOD_CONSUMPTION * (ADULT_DURATION_SECONDS / 3600.0) * _FOUNDING_FOOD_LIVES
)
STARTING_FOOD_STOCK = {
    'bread': STARTING_POPULATION * STARTING_FOOD_PER_CAPITA * 0.5,
    'meat': STARTING_POPULATION * STARTING_FOOD_PER_CAPITA * 0.5,
}


# ============================================================================
# REWARD TIMING
# ============================================================================
# env_config.json and reward_config.json carry time values too. Rather than keep two copies
# of the scale in sync by hand (they drifted at every retune), the loaders read these at load
# time and overwrite the JSON's placeholders. config.py is the one source of truth; the JSON
# owns only the non-temporal values (feature_keys, penalties).
#
# Check-points as fractions of the episode: a district with nothing built by 1/6 of the game,
# and no barracks by 1/3, is being warned that it is falling behind.
REWARD_TIMING_THRESHOLDS = {
    'stagnation_check': EPISODE_SECONDS / 6.0,
    'military_check': EPISODE_SECONDS / 3.0,
    'resource_starvation_check': EPISODE_SECONDS / 6.0,
    'max_game_time': EPISODE_SECONDS,
}


# ============================================================================
# INVARIANTS
# ============================================================================

def _decisions_for(seconds: float) -> float:
    return seconds / DECISION_INTERVAL_SECONDS


# An action must span several decisions, or it is not a decision.
_STANDARD_BUILD_DECISIONS = _decisions_for(BUILDING_BUILD_TIME['farm'] * 60.0)
_SOLDIER_DECISIONS = _decisions_for(UNIT_TRAINING_TIME['soldier'] * 60.0)
_KID_DECISIONS = _decisions_for(KID_DURATION_SECONDS)

assert GENERATIONS_PER_EPISODE >= 10.0, "the agent must outlive several generations"
assert _STANDARD_BUILD_DECISIONS >= 5.0, (
    f"a standard building takes {_STANDARD_BUILD_DECISIONS:.1f} decisions; raise "
    f"DECISIONS_PER_EPISODE or the build fraction, or construction is a purchase"
)
assert _SOLDIER_DECISIONS >= 1.5, (
    f"a soldier trains in {_SOLDIER_DECISIONS:.1f} decisions; training must cost time"
)
assert _KID_DECISIONS >= 20.0, (
    f"childhood lasts {_KID_DECISIONS:.1f} decisions; a birth must be felt as an investment"
)

# A building is a raw producer or a processor, never both: ResourceManager tests
# RESOURCE_PROCESSING first, so a building in both would silently ignore its PRODUCTION_RATES.
_shadowed = sorted(set(PRODUCTION_RATES) & set(RESOURCE_PROCESSING))
assert not _shadowed, f"a building cannot be both raw producer and processor: {_shadowed}"

# Anything that produces, employs. A building with no worker requirement is skipped by
# `assign_work`, falls back on the "needs nobody" productivity of 1.0, and runs at FULL
# OUTPUT with an empty floor — and, being unstaffed, teaches its trade to nobody, which
# silently makes the careers above it unreachable (D31).
_FREE_PRODUCERS = sorted(
    (set(PRODUCTION_RATES) | set(RESOURCE_PROCESSING)) - set(WORKERS_NEEDED)
)
assert not _FREE_PRODUCERS, (
    f"these buildings produce but employ nobody, so they would run at full output with an "
    f"empty floor and teach their trade to no one: {_FREE_PRODUCERS}"
)

# The recipes above are AUTHORED here, not loaded from game_data/resource.json, because they
# are derived quantities: a bakery's throughput is *defined as* a farm's grain output minus
# two stages of conversion loss, and JSON can only hold the resulting number, cut loose from
# the farm it came from. The sim also runs a deliberately different economy from the game
# (scaled rates, an anchored 900 s life), so resource.json is the GAME's balance, not ours.
#
# But the game's recipe file must not drift out of sight. config.py used to load it and then
# silently override every line, so editing resource.json changed nothing and reported nothing.
# This guard is what makes that impossible: if the game gains a production line, the sim
# either models it or names it here as deliberately out of scope — it can no longer ignore it
# by accident.
_RECIPES_NOT_MODELLED = {
    'gold_smelting',      # gold has no sink in the sim: a dead resource invites the district
                          # to spend real miners on something it cannot spend (D30)
    'chemist', 'clothes', 'hunting',            # folded into other buildings, or not a tier yet
    'iron_smelting', 'blacksmith_weapons',      # keyed by RECIPE, not by a building that owns
    'leather_armor', 'iron_armor',              # them; the sim's smelter/blacksmith replace them
    'farm',               # a farm takes no input: it is a raw producer, in PRODUCTION_RATES
}
_unmodelled = sorted(set(load_production_recipes()) - set(RESOURCE_PROCESSING) - _RECIPES_NOT_MODELLED)
assert not _unmodelled, (
    f"game_data/resource.json defines recipes the simulator neither models nor waives: "
    f"{_unmodelled}. Add them to RESOURCE_PROCESSING (anchored, not typed in) or list them "
    f"in _RECIPES_NOT_MODELLED with the reason."
)

# A district must be able to STAFF what it builds. If this fails, the settlement is
# arithmetically impossible and no policy, bot or agent can save it.
_SETTLEMENT_SLOTS = sum(
    WORKERS_NEEDED.get(b, 1) for b in (
        'lumberyard', 'lumberyard', 'quarry', 'clay_pit', 'farm', 'vegetable_garden', 'well',
        'carpentry', 'pottery', 'mill', 'stone_cutter', 'cattle_shed', 'school', 'bakery',
        'tailor', 'orchard', 'mine', 'smelter', 'blacksmith', 'hunting_shed',
    )
)
assert _SETTLEMENT_SLOTS <= TARGET_ADULT_WORKERS, (
    f"a full settlement asks for {_SETTLEMENT_SLOTS} workers but a district of "
    f"{TARGET_DISTRICT_POPULATION} has only {TARGET_ADULT_WORKERS:.0f} adults: it cannot staff "
    f"what it builds, and no build order can fix that"
)
