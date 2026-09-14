"""
Professions and skills for individual sims.

Mirrors the Rust game's `PersonProfession` / skills system (documentation/gameplay/
skills_system.md), restricted to the professions that matter to buildings that exist
in the simulator. A sim learns a profession by doing the work (experience), and higher
tiers need a prerequisite skill — exactly the game's model, aggregated to what the
macro simulator can carry.

Skill tiers (from the game):
  L1  any adult can do it                    farmer, hunter, woodcutter, miner, miller…
  L2  needs 50% of a prerequisite L1 skill   mason (stone), carpenter (wood)…
  L3  needs 50% of a prerequisite L2/other   baker (flouring), blacksmith…
"""

from enum import Enum


class Profession(Enum):
    """Civil professions, taken from `documentation/gameplay/skills_system.md`.

    A CAREER, not a job title. The game's rule is one line, and it is the whole model:

        "Level X can be accessed when getting enough experience on a specific Level X-1
         job, or pairing with another worker with that experience."

    So the way to a potter is to dig clay until you are good at it; the way to a smith is
    to mine stone, then iron, then take up the hammer. Experience on the job is the MAIN
    road (the school only makes the walking faster — D31).
    """
    NONE = "none"              # unskilled: an adult who has not specialised

    # --- Level 1: any adult can do these ---
    FARMER = "farmer"          # farm, vegetable_garden, orchard   ("Food Farming")
    HUNTER = "hunter"          # hunting_shed                      ("Hunting")
    BREEDER = "breeder"        # cattle_shed                       ("Breeding")
    WOODCUTTER = "woodcutter"  # lumberyard                        ("Wood cutting")
    MINER = "miner"            # quarry                            ("Stone Mining")
    DIGGER = "digger"          # clay_pit                          ("Clay digging")
    MILLER = "miller"          # mill                              ("Flouring")

    # --- Level 2: 50% of a specific L1 skill ---
    IRON_MINER = "iron_miner"  # mine        <- 50% Stone Mining   ("Iron mining")
    POTTER = "potter"          # pottery     <- 50% Clay digging
    CARPENTER = "carpenter"    # carpentry   <- 50% Wood cutting
    MASON = "mason"            # stone_cutter<- 50% Stone Mining
    TAILOR = "tailor"          # tailor      <- 50% Breeding

    # --- Level 3: 50% of a specific L2 skill ---
    BAKER = "baker"            # bakery      <- 50% Flouring
    BLACKSMITH = "blacksmith"  # blacksmith  <- 50% Iron mining
    SMELTER = "smelter"        # smelter     <- 50% Iron mining

    # --- Support: does not produce, but speeds everyone's learning ---
    TEACHER = "teacher"        # school


# What profession each building's work needs. Buildings not listed are worked by any
# adult (NONE), or need no workers at all. This is the seam the slot system (step 2b)
# will use: a building's output scales with sims of the RIGHT profession assigned to it.
BUILDING_PROFESSION = {
    'farm': Profession.FARMER,
    'vegetable_garden': Profession.FARMER,
    'orchard': Profession.FARMER,
    'hunting_shed': Profession.HUNTER,
    'cattle_shed': Profession.BREEDER,
    'lumberyard': Profession.WOODCUTTER,
    'quarry': Profession.MINER,
    'mine': Profession.IRON_MINER,
    'clay_pit': Profession.DIGGER,
    'mill': Profession.MILLER,
    'stone_cutter': Profession.MASON,
    'carpentry': Profession.CARPENTER,
    'pottery': Profession.POTTER,
    'bakery': Profession.BAKER,
    'blacksmith': Profession.BLACKSMITH,
    'smelter': Profession.SMELTER,
    'tailor': Profession.TAILOR,
    'school': Profession.TEACHER,
}

# Skill tier of each profession, and its prerequisite (a profession whose skill must be
# at least PREREQUISITE_SKILL_FRACTION before this one can be learned). L1 has none.
PROFESSION_TIER = {
    Profession.FARMER: 1, Profession.HUNTER: 1, Profession.BREEDER: 1,
    Profession.WOODCUTTER: 1, Profession.MINER: 1, Profession.DIGGER: 1,
    Profession.MILLER: 1,
    Profession.IRON_MINER: 2, Profession.POTTER: 2, Profession.CARPENTER: 2,
    Profession.MASON: 2, Profession.TAILOR: 2,
    Profession.BAKER: 3, Profession.BLACKSMITH: 3, Profession.SMELTER: 3,
}

#: Straight from the game's unlock table. Note how deep the forge is: stone -> iron -> smith.
#: THAT is why the blacksmith was never staffed (D29) — it was not a shortage of hands, it
#: was a shortage of CAREER. Nobody in the district had ever mined stone long enough to be
#: allowed near a mine, let alone an anvil.
PROFESSION_PREREQUISITE = {
    Profession.IRON_MINER: Profession.MINER,        # 50% Stone mining
    Profession.POTTER: Profession.DIGGER,           # 50% Clay digging
    Profession.CARPENTER: Profession.WOODCUTTER,    # 50% Wood cutting
    Profession.MASON: Profession.MINER,             # 50% Stone mining
    Profession.TAILOR: Profession.BREEDER,          # 50% Breeding
    Profession.BAKER: Profession.MILLER,            # 50% Flouring
    Profession.BLACKSMITH: Profession.IRON_MINER,   # 50% Iron mining
    Profession.SMELTER: Profession.IRON_MINER,      # 50% Iron mining
}

PREREQUISITE_SKILL_FRACTION = 0.5    # "50% of the prerequisite skill" from the game

# Skill grows by working, from 0 to 1. A sim at skill 1.0 is a master; a NONE sim
# assigned to L1 work is a novice who learns on the job. Anchored to the adult life
# (like everything else, see config.py TIME SCALE): a sim masters a skill by working
# ~this fraction of a working life, so expertise is achievable within a generation but
# not instant, and losing a master hurts.
SKILL_MAX = 1.0
# A sim masters a skill by working ~this fraction of a working life. It must be < 1 life, or
# — given D36's fast generational turnover — no one lives long enough to master anything and
# every workshop is worked by novices (measured: productivity pinned ~0.3, the higher trades
# never staffed). Mentorship and schools still multiply this and still matter; they now make a
# fast walk faster, instead of being the only way to arrive at all.
_SKILL_MASTERY_FRACTION = 0.6

# Transferable expertise: when a qualified sim takes up a higher-tier profession, the
# prerequisite skill it already has gives it a HEAD START in the new one, rather than starting
# from novice. A master clay-digger becomes a competent potter (D31: "the way to a potter is to
# dig clay until you are good at it"). Without this, climbing the ladder is a pure downgrade —
# the sim resets to novice and, given the lifespan, never re-masters — so no one ever moves up
# and the L2/L3 trades sit empty or novice-worked.
SKILL_TRANSFER_FRACTION = 0.5

# A novice is not useless: an adult can do the basic job, just not as well. Output
# scales from this floor (skill 0) to full (skill 1). Kept fairly high so a fresh
# settlement of novices can still feed itself — too low and skill-reduced output starves
# the population before it can learn (a demographic death spiral).
SKILL_FLOOR = 0.6

# Learning beside an experienced coworker is faster ("pairing with another worker with
# that experience", from the game). This is what lets skill accumulate across
# generations instead of every cohort restarting from zero.
MENTORSHIP_MULTIPLIER = 3.0


def skill_growth_per_second() -> float:
    from .config import ADULT_DURATION_SECONDS
    return SKILL_MAX / (_SKILL_MASTERY_FRACTION * ADULT_DURATION_SECONDS)


# A staffed school speeds every learner in the district. The RL lever: spend an adult
# (the teacher, who then produces nothing) and the build cost, to raise the whole
# population's skill faster — an investment in human capital that pays off later and
# reaches the L2/L3 professions (baker, blacksmith) sooner.
SCHOOL_TRAINING_BONUS = 2.0        # each staffed school multiplies learning by up to this


def can_learn(profession: Profession, skills: dict) -> bool:
    """True if a sim with these skills meets the prerequisite for `profession`."""
    prereq = PROFESSION_PREREQUISITE.get(profession)
    if prereq is None:
        return True                  # L1: anyone
    return skills.get(prereq, 0.0) >= PREREQUISITE_SKILL_FRACTION
