"""
Game configuration for the ML simulator.

Loaded from game_data/ (shared with the Rust game): building costs, build times,
starting resources, unit training data. Authored here: the time scale, the map, the
production chains, and everything derived from them.

Discipline: pick an anchor, state a ratio, derive the number — never hand-tune a derived
value. Full rationale in docs/design_decisions.md (D13 time, D27 map, D29 war economy,
D30 production, D33 district).
"""

from .game_data_loader import (
    load_building_costs,
    load_starting_resources,
    load_building_build_times,
    load_unit_training_data,
    load_production_recipes,      # only to guard against drift (invariants below)
)
from .professions import SKILL_FLOOR as _NOVICE

# --- TIME SCALE: one anchor, everything derived (D13) ---
ADULT_DURATION_SECONDS = 900.0          # THE ANCHOR: mean adult working life

# Demography, as fractions of a working life.
KID_DURATION_SECONDS = 0.25 * ADULT_DURATION_SECONDS
ELDER_DURATION_SECONDS = 0.50 * ADULT_DURATION_SECONDS

# Decisions (the RL budget) and game time move together; the invariants below assert it.
GENERATIONS_PER_EPISODE = 10.0
EPISODE_SECONDS = GENERATIONS_PER_EPISODE * ADULT_DURATION_SECONDS        # 9000 s
DECISIONS_PER_EPISODE = 1440
DECISION_INTERVAL_SECONDS = EPISODE_SECONDS / DECISIONS_PER_EPISODE       # 6.25 s

# A standard building is 5% of a working life; loaded build times are rescaled as a whole,
# anchored on the farm (120 s in game_data).
_STANDARD_BUILDING_FRACTION = 0.05
_BUILD_TIME_SCALE = (_STANDARD_BUILDING_FRACTION * ADULT_DURATION_SECONDS) / 120.0

# Vital rates: stated per working life, then converted.
_PER_LIFE_TO_PER_HOUR = 3600.0 / ADULT_DURATION_SECONDS

BIRTHS_PER_LIFE = 7.0                  # enough to reach TARGET_DISTRICT_POPULATION
BIRTH_CHANCE_PER_TURN = BIRTHS_PER_LIFE * _PER_LIFE_TO_PER_HOUR

CONSCRIPTS_PER_LIFE = 8.0              # a full mobilisation takes a good part of a life
CONSCRIPTION_RATE_PER_HOUR = CONSCRIPTS_PER_LIFE * _PER_LIFE_TO_PER_HOUR

STARVATION_FRACTION_PER_LIFE = 2.0     # total famine empties a district well inside a life
STARVATION_DEATH_RATE = STARVATION_FRACTION_PER_LIFE * _PER_LIFE_TO_PER_HOUR

PREGNANCY_DURATION_SECONDS = 0.1 * ADULT_DURATION_SECONDS
PREGNANCY_PRODUCTIVITY = 0.5      # a pregnant woman works at half output

# Fallback march time for a map-less game; with a map it derives from distance walked (D26).
ATTACK_TRAVEL_TIME_SECONDS = 0.1 * ADULT_DURATION_SECONDS


# --- THE MAP (D27) ---
# Nothing here depends on tile COUNT, only on the ratios. Edit the ratios; seconds derive.
MAP_WIDTH = 96
MAP_HEIGHT = 96

# Territory radius = the plot's carrying capacity, the real size limit in place of a pop cap.
# 8 seats the full build chain + army with rectangular footprints (2x2..3x3); 6 does not.
DISTRICT_RADIUS = 8

# Rivals founded ~3 plots apart: war is live from the start, neither inside the other.
SETTLEMENT_SEPARATION = 3.5 * DISTRICT_RADIUS      # ~21 tiles

# Calibrated so default separation reproduces ATTACK_TRAVEL_TIME_SECONDS. Tune the ratio, never the seconds.
MARCH_SECONDS_PER_TILE = ATTACK_TRAVEL_TIME_SECONDS / SETTLEMENT_SEPARATION

# Fraction of eligible tiles carrying a vein — what makes one plot worth more than another.
DEPOSIT_DENSITY = 0.25

# Elevation mirrored from game_data/map.json, normalised to [0, ELEVATION_MAX]; water is the bottom fifth.
ELEVATION_MAX = 10.0
SEA_LEVEL = 2.0
MOUNTAIN_THRESHOLD = 6.5


# --- THE DISTRICT, ANCHORED (D33) ---
# Food, housing and workforce all derive from one target (tuned separately they did not add up).
TARGET_DISTRICT_POPULATION = 60        # THE ANCHOR: what a mature district IS

ADULT_SHARE = 0.45                     # the rest are children and elders
TARGET_ADULT_WORKERS = TARGET_DISTRICT_POPULATION * ADULT_SHARE      # ~27 hands

FOOD_LABOUR_SHARE = 0.20               # feeding itself must not cost more than this share of hands
_FOOD_WORKERS = TARGET_ADULT_WORKERS * FOOD_LABOUR_SHARE            # ~5.4 hands on food

# Mouths one food worker feeds, computed at NOVICE skill: a fresh settlement is all novices,
# and a food supply that only works at mastery starves it through its first generation.
PEOPLE_FED_PER_FOOD_WORKER = TARGET_DISTRICT_POPULATION / _FOOD_WORKERS / _NOVICE   # ~18

# Mean of person.json elder_productivity over civilian professions. Dimensionless, no anchor.
ELDER_PRODUCTIVITY = 0.45


# --- WORKERS ---
# WORKERS_NEEDED is the maximum crew (100% output); WORKERS_MIN the smallest that runs at
# all; between them output is proportional. A district mans all buildings thinly before
# fattening any crew (D35).
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


# --- PRODUCTION ---
# A production building must repay its cost in a fraction of a life.
PRODUCTION_SCALE = 5.0
_FARM_GRAIN = 90.0 * PRODUCTION_SCALE

# The bread chain (farm 2 + mill 1 + bakery 1 = 4 hands) sets what a person eats; every
# other food is a multiple of THAT, so the whole table moves together.
_BREAD_CHAIN_WORKERS = WORKERS_NEEDED['farm'] + WORKERS_NEEDED['mill'] + WORKERS_NEEDED['bakery']
PEOPLE_FED_PER_FARM = PEOPLE_FED_PER_FOOD_WORKER * _BREAD_CHAIN_WORKERS
POPULATION_FOOD_CONSUMPTION = _FARM_GRAIN / PEOPLE_FED_PER_FARM      # per person per hour
POPULATION_WATER_CONSUMPTION = 0.6 * POPULATION_FOOD_CONSUMPTION

# Raw producers only (no inputs); processors live in RESOURCE_PROCESSING, and an invariant
# below forbids a building from being both. Food rates derive from the anchor above.
PRODUCTION_RATES = {
    'farm': {'grain': _FARM_GRAIN},
    'hunting_shed': {'meat': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER},
    'lumberyard': {'wood': 60.0 * PRODUCTION_SCALE},
    'quarry': {'stone': 30.0 * PRODUCTION_SCALE},
    'clay_pit': {'clay': 30.0 * PRODUCTION_SCALE},
    'mine': {'iron_ore': 15.0 * PRODUCTION_SCALE},
    'well': {'water': 150.0 * PRODUCTION_SCALE},
    # The orchard's real worth is being a FOURTH food (the variety bonus, D18), not calories.
    'vegetable_garden': {'vegetable': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER * 0.65},
    'orchard': {'fruit': POPULATION_FOOD_CONSUMPTION * PEOPLE_FED_PER_FOOD_WORKER * 0.45},
}

# The production circles, closed (D30). Recipes are authored HERE, not loaded from
# game_data/resource.json (see the drift guard at the bottom). Rates per hour; every line is
# a closed circle — something produces it, something consumes it.
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
    # CLAY -> BRICKS: the bakery costs bricks, so without a potter, bread is unreachable (D30).
    'pottery': {
        'input': {'clay': 120.0},
        'output': {'clay_bricks': 60.0},
    },
    'stone_cutter': {
        'input': {'stone': 120.0},
        'output': {'stone_bricks': 60.0},
    },
    # CATTLE -> meat (variety) + leather (armour) + wool, without a mill.
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
    # WOOL + LEATHER -> CLOTHES: the one derivative the PEOPLE consume, not the army.
    'tailor': {
        'input': {'wool': 30.0, 'leather': 30.0},
        'output': {'clothes': 20.0},
    },
}


# --- BUILDINGS ---
_BASE_BUILDING_COSTS = load_building_costs()

# Costs not in the JSON, plus the ones D30 re-priced so buildings COST what the workshops
# make (trades necessary, not optional). Each still lets a district open on wood + stone.
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
    # The school costs no bricks by design: it's the bootstrap. Higher trades are gated on
    # SKILL and the school transmits it, so charging it in bricks would close the circle out.
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


# --- UNITS AND THE WAR ECONOMY (D29) ---
# An army is EQUIPPED, not conjured: the binding constraint is adult MEN, not throughput.
# Iron does not give more soldiers, it makes the few you have nearly twice the soldier —
# which is what a plot with an iron vein is worth, and why WHERE you settle can decide a war.
_UNIT_DATA = load_unit_training_data()

UNIT_TRAINING_COST = _UNIT_DATA['costs']
UNIT_STRENGTH = _UNIT_DATA['strength']
UNIT_TRAINING_TIME = {
    unit: (seconds / 60.0) * _BUILD_TIME_SCALE      # loaded in seconds, held in minutes
    for unit, seconds in _UNIT_DATA['times'].items()
}

_IRON_KIT_ADVANTAGE = 1.8      # mail + iron sword vs a spear, as a RATIO (never hand-tune strength)

# The soldier's wood now sits in his weapon: same wood per soldier, paid one step earlier.
UNIT_TRAINING_COST['soldier'] = {'grain': 5, 'wooden_weapon': 1}

# The same man, better equipped, and therefore a different soldier.
UNIT_TRAINING_COST['man_at_arms'] = {'grain': 5, 'iron_weapon': 1, 'armour': 1}
UNIT_STRENGTH['man_at_arms'] = UNIT_STRENGTH['soldier'] * _IRON_KIT_ADVANTAGE
UNIT_TRAINING_TIME['man_at_arms'] = UNIT_TRAINING_TIME['soldier']


# --- COMBAT ---
DEFENDER_BONUS = 1.3
COMBAT_CASUALTIES_RATE = 0.4           # the loser's losses
WINNER_CASUALTIES_RATE = 0.1
BUILDING_DAMAGE_CHANCE = 0.3
WAREHOUSE_DESTRUCTION_THRESHOLD = 2.0  # strength ratio needed to destroy a warehouse
# The only victory is conquest: outlast every rival's last warehouse; time-out is a truncation (D14).


# --- FOOD, HAPPINESS AND POLICY ---
# Grain and flour are intermediates; the population lives on the END products (what makes
# the chains necessary). Fish is deferred to Stage 2 (needs water bodies, D16).
EDIBLE_FOODS = ['bread', 'meat', 'vegetable', 'fruit']

# Variety is tiered: each distinct food beyond the first adds a capped bonus — what makes
# the orchard (poor calories, distinct food) worth building.
MIN_FOOD_TYPES_FOR_VARIETY = 2
FOOD_VARIETY_BONUS_PER_TYPE = 5.0
FOOD_VARIETY_BONUS_CAP = 15.0          # 4 foods -> +15, leaving room for a 5th (fish)

# Clothes wear out per person per hour; a clothed district is happier. A real sink, or the
# tailor would be a dead end.
CLOTHES_CONSUMPTION_PER_CAPITA = 0.15
CLOTHES_HAPPINESS_BONUS = 8.0

MIN_FOOD_PER_CAPITA_FOR_GROWTH = 2.0
MIN_HAPPINESS_FOR_GROWTH = 50.0
STARVATION_HAPPINESS_LOSS = 10.0       # per hour starving

# Unemployment malus (D37): happiness lost at full unemployment, scaled by idle share. This
# couples population to the DEMAND FOR LABOUR — grow past the jobs and births slow. Must
# dominate the happiness bonuses (diet +15, clothes +8) or nothing checks growth. Tuned on
# 16 seeds: 50 was too weak (pop ~120, 60% idle); 150 lands the mature district near full
# employment (~16% idle) and halves the oscillation. Re-verify alongside the build plan (D38).
UNEMPLOYMENT_HAPPINESS_PENALTY = 150.0

# Happiness of an un-governed district with food on the shelves; everything else is a penalty
# subtracted from it (happiness only ever decreases).
BASE_HAPPINESS = 75.0

# The four policy levers (see docs/population_and_policies.md).
POLICY_EFFECTS = {
    'minimum_food': -15.0,
    'siege_mode': -20.0,
    'forced_conscription': -25.0,
    'block_families': -10.0,           # military camp
    'cant_leave': -15.0,               # military camp
}

# INVARIANT that makes coercion bite:
#   BASE_HAPPINESS + POLICY_EFFECTS['forced_conscription'] == MIN_HAPPINESS_FOR_GROWTH
# Drafting every adult drives fertility to exactly zero. Preserve this if you retune.

MINIMUM_FOOD_RATION_FACTOR = 0.6       # L1 minimum_food is a ration, not a ban: in (0, 1)

# How hard unhappiness bites fertility: FLOOR + (1-FLOOR)*clamp((happiness-MIN)/span), so it
# is 1.0 at BASE_HAPPINESS and bottoms out at FLOOR, not 0. Happiness is the MAIN gate on
# births (D37); the floor is low but non-zero (a hard zero makes an absorbing state that
# wrecks RL exploration — only starvation and prohibition should take fertility to 0).
HAPPINESS_FERTILITY_FLOOR = 0.15

# Post-birth conception cooldown — a spacing between children. Anchored to the working life
# so it scales with the time scale.
FERTILITY_COOLDOWN_SECONDS = 0.15 * ADULT_DURATION_SECONDS

# High parity lowers fertility: the first two children come at full rate; from the third on,
# each further conception is multiplied by DECAY once more (1, 1, DECAY, DECAY², …), so large
# families get progressively rarer. (No kinship model yet; the woman stands in for the couple, D37.)
HIGH_PARITY_FERTILITY_DECAY = 0.6
HIGH_PARITY_THRESHOLD = 2               # children already born, above which decay kicks in

# L4 forced_conscription drafts adults deterministically (no RNG, so it cannot perturb other
# draws) at CONSCRIPTION_RATE_PER_HOUR: a lever must be sized against the episode (D11).


# --- STARTING CONDITIONS ---
STARTING_RESOURCES = load_starting_resources()

# A few spears in the rack, or the first soldier waits on a carpentry that waits on a
# lumberyard, and the opening is decided by build order rather than strategy.
STARTING_RESOURCES['wooden_weapon'] = 4

STARTING_BUILDINGS = {
    'warehouse': 1,      # if it is destroyed, you lose
    'house': 4,
    'farm': 1,           # or the founding population starves before it can build one
}

# Housing sized so TARGET_DISTRICT_POPULATION fits: a district capped on beds stops growing
# and never reaches the workforce its workshops need.
STARTING_POPULATION = 20
STARTING_POPULATION_CAPACITY = 24
POPULATION_CAPACITY_PER_HOUSE = 8
POPULATION_CAPACITY_PER_DORMITORY = 16

# Edible runway while the first food buildings go up (grain alone would starve them — it is
# not food). Sized in LIVES of eating so it moves with the time scale: two working lives'
# worth per founding head, split between the two starting foods.
_FOUNDING_FOOD_LIVES = 2.0
STARTING_FOOD_PER_CAPITA = (
    POPULATION_FOOD_CONSUMPTION * (ADULT_DURATION_SECONDS / 3600.0) * _FOUNDING_FOOD_LIVES
)
STARTING_FOOD_STOCK = {
    'bread': STARTING_POPULATION * STARTING_FOOD_PER_CAPITA * 0.5,
    'meat': STARTING_POPULATION * STARTING_FOOD_PER_CAPITA * 0.5,
}


# --- REWARD TIMING ---
# config.py is the one source of truth for the time scale; the loaders overwrite the JSON's
# temporal placeholders at load time (the JSON owns only non-temporal values). Check-points
# as fractions of the episode: nothing built by 1/6, no barracks by 1/3, is a warning.
REWARD_TIMING_THRESHOLDS = {
    'stagnation_check': EPISODE_SECONDS / 6.0,
    'military_check': EPISODE_SECONDS / 3.0,
    'resource_starvation_check': EPISODE_SECONDS / 6.0,
    'max_game_time': EPISODE_SECONDS,
}


# --- INVARIANTS ---

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

# A building is a raw producer or a processor, never both (ResourceManager tests
# RESOURCE_PROCESSING first, so a building in both would ignore its PRODUCTION_RATES).
_shadowed = sorted(set(PRODUCTION_RATES) & set(RESOURCE_PROCESSING))
assert not _shadowed, f"a building cannot be both raw producer and processor: {_shadowed}"

# Anything that produces, employs. A worker-less producer is skipped by `assign_work`, runs
# at full output with an empty floor, and teaches its trade to nobody — silently making the
# careers above it unreachable (D31).
_FREE_PRODUCERS = sorted(
    (set(PRODUCTION_RATES) | set(RESOURCE_PROCESSING)) - set(WORKERS_NEEDED)
)
assert not _FREE_PRODUCERS, (
    f"these buildings produce but employ nobody, so they would run at full output with an "
    f"empty floor and teach their trade to no one: {_FREE_PRODUCERS}"
)

# Drift guard (D30): the recipes above are authored here, but the game's recipe file must not
# drift out of sight. If the game gains a production line, the sim must model it or name it
# here as out of scope — it can no longer be ignored by accident.
_RECIPES_NOT_MODELLED = {
    'gold_smelting',      # gold has no sink in the sim: a dead resource (D30)
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

# A district must be able to STAFF what it builds, or the settlement is arithmetically
# impossible and no policy, bot or agent can save it.
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
