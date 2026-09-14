"""
Core data structures representing the game state.

A `Faction` is a container of `District`s (Stage 1 of the roadmap). A district is
a settlement: one warehouse, its own buildings, population, stock and units. All
state lives on the district; the faction exposes aggregate views over its districts.

A faction is defeated when it has lost ALL of its warehouses.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import numpy as np
from .config import STARTING_RESOURCES, STARTING_BUILDINGS, UNIT_STRENGTH
from .config import STARTING_POPULATION, STARTING_POPULATION_CAPACITY
from .config import BASE_HAPPINESS, STARVATION_HAPPINESS_LOSS
from .config import ADULT_DURATION_SECONDS, ELDER_DURATION_SECONDS
from .config import ELDER_PRODUCTIVITY, KID_DURATION_SECONDS
from .config import EDIBLE_FOODS, STARTING_FOOD_STOCK
from .config import MIN_FOOD_TYPES_FOR_VARIETY, FOOD_VARIETY_BONUS_PER_TYPE, FOOD_VARIETY_BONUS_CAP
from .config import CLOTHES_HAPPINESS_BONUS, UNEMPLOYMENT_HAPPINESS_PENALTY
from .policies import DistrictPolicies

class BuildingType(Enum):
    """All building types in the game."""
    WAREHOUSE = "warehouse"
    FARM = "farm"
    LUMBERYARD = "lumberyard"
    QUARRY = "quarry"
    CLAY_PIT = "clay_pit"
    MINE = "mine"
    WELL = "well"
    MILL = "mill"
    BAKERY = "bakery"
    HUNTING_SHED = "hunting_shed"
    CATTLE_SHED = "cattle_shed"
    VEGETABLE_GARDEN = "vegetable_garden"
    ORCHARD = "orchard"
    SCHOOL = "school"
    HOUSE = "house"
    POTTERY = "pottery"
    CARPENTRY = "carpentry"
    STONE_CUTTER = "stone_cutter"
    BLACKSMITH = "blacksmith"
    SMELTER = "smelter"
    TAILOR = "tailor"
    DORMITORY = "dormitory"
    WATCHTOWER = "watchtower"
    DEFENSIVE_WALL_WOOD = "defensive_wall_wood"
    DEFENSIVE_WALL_STONE = "defensive_wall_stone"
    BARRACKS = "barracks"


class UnitType(Enum):
    """All unit types in the game."""
    SOLDIER = "soldier"
    MAN_AT_ARMS = "man_at_arms"
    ARCHER = "archer"
    CAVALRY = "cavalry"


# Nominal age thresholds (seconds). Each sim scales these by its own `life_scale`, so
# a cohort born together does NOT age out in one synchronised wave (see D22).
KID_END = KID_DURATION_SECONDS                                    # kid -> adult
ADULT_END = KID_DURATION_SECONDS + ADULT_DURATION_SECONDS         # adult -> elder
LIFESPAN = ADULT_END + ELDER_DURATION_SECONDS                     # elder -> death

LIFE_SCALE_SIGMA = 0.15          # per-sim spread of life-stage timing (desynchronises waves)


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


@dataclass
class Sim:
    """One individual person.

    Grown as COMPONENTS, mirroring the Rust `Person { core, stats, needs, skills,
    family, … }`, so new per-sim information is added as a field without disturbing the
    rest — the structure scales with the model. Now: age + a personal life-timing scale
    (core), gender, the skills component, and pregnancy state.
    """
    age: float = 0.0                              # seconds lived — the life-stage axis
    life_scale: float = 1.0                       # personal multiplier on age thresholds
    gender: 'Gender' = None                       # MALE / FEMALE
    profession: 'Profession' = None               # current specialisation (None = unskilled)
    skills: Dict['Profession', float] = field(default_factory=dict)  # profession -> [0,1]
    pregnant_remaining: float = 0.0               # seconds left of pregnancy (0 = not pregnant)
    employed: bool = False                        # set each step by assign_work; drives the
                                                  # unemployment happiness malus
    children_born: int = 0                        # this woman's parity — high parity lowers her
                                                  # fertility (no kinship model yet, so the woman
                                                  # stands in for the couple)
    fertility_cooldown_remaining: float = 0.0     # seconds until she can conceive again post-birth
    # future components: family (kinship), needs, stats — add here.

    def __post_init__(self):
        if self.profession is None:
            from .professions import Profession
            self.profession = Profession.NONE
        if self.gender is None:
            self.gender = Gender.MALE if np.random.random() < 0.5 else Gender.FEMALE

    @property
    def is_kid(self) -> bool:
        return self.age < KID_END * self.life_scale

    @property
    def is_adult(self) -> bool:
        return KID_END * self.life_scale <= self.age < ADULT_END * self.life_scale

    @property
    def is_elder(self) -> bool:
        return ADULT_END * self.life_scale <= self.age < LIFESPAN * self.life_scale

    @property
    def is_alive(self) -> bool:
        return self.age < LIFESPAN * self.life_scale

    @property
    def is_pregnant(self) -> bool:
        return self.pregnant_remaining > 0.0

    @property
    def can_conceive(self) -> bool:
        """Adult women who aren't already pregnant and are past their post-birth cooldown."""
        return (self.gender is Gender.FEMALE and self.is_adult
                and not self.is_pregnant and self.fertility_cooldown_remaining <= 0.0)


@dataclass
class BuildingInProgress:
    """Represents a building currently under construction."""
    building_type: BuildingType
    turns_remaining: int
    #: The tile it is being raised on. Reserved when the order is given, not when the work
    #: finishes — otherwise the vein you started digging for could be built over by your own
    #: next order, and the building would complete with nowhere to stand.
    tile: Optional[Tuple[int, int]] = None
    #: Yield multiplier of that tile, fixed at the moment of the decision.
    quality: float = 1.0


@dataclass
class UnitInTraining:
    """Represents a unit currently in training."""
    unit_type: UnitType
    turns_remaining: int


class _AggregateMapping(Mapping):
    """A summed, read-only view over the same dict across several districts.

    Writing through a faction-level aggregate is meaningless once a faction owns
    more than one district (which district's stock did you mean?), so mutation
    raises instead of being silently dropped. Address the district directly.
    """

    __slots__ = ('_data', '_what')

    def __init__(self, data: Dict, what: str):
        self._data = data
        self._what = what

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(self._data)

    def _reject(self, *args, **kwargs):
        raise TypeError(
            f"Faction.{self._what} is a read-only aggregate over "
            f"{len(self._data)} district(s). Write to a specific district: "
            f"faction.districts[i].{self._what}[...] = ..."
        )

    __setitem__ = __delitem__ = _reject


@dataclass
class District:
    """A settlement: one warehouse, its own economy, population and garrison."""
    id: int = 0
    faction_id: int = 0

    # Where the district sits on the region graph. Unused until Stage 2.
    location: Optional[int] = None

    # Resources (per-district stockpile; logistics between districts is Stage 4)
    resources: Dict[str, int] = field(default_factory=dict)

    # Buildings (completed)
    buildings: Dict[str, int] = field(default_factory=dict)

    # Buildings under construction
    buildings_in_progress: List[BuildingInProgress] = field(default_factory=list)

    # --- where this district IS (D27) ---
    # An Area is centred on a tile: a warehouse for a civil district, a military camp for
    # the camp tier (documentation/gameplay/area_system.md). The map does not care which.
    # `None` means "no map" — the sim still runs mapless, which keeps every old test honest.
    center: Optional[Tuple[int, int]] = None
    #: tile -> building_type, for the buildings this district has actually put on the ground.
    building_tiles: Dict[Tuple[int, int], str] = field(default_factory=dict)
    #: The world this district stands in. A back-reference, so that placing or losing a
    #: building keeps the map and the district in step without every caller passing it in.
    _map: Optional['GameMap'] = field(default=None, repr=False, compare=False)

    #: building_type -> how many of its work slots are OPEN. Absent = all of them (D35).
    #: Closing a slot is how a district says "I would rather have this man in the pottery than
    #: as the second woodcutter". It is a real lever, and it belongs to whoever governs the
    #: district — the scripted bot today, the RL agent when it is given the action.
    open_slots: Dict[str, int] = field(default_factory=dict)

    # Population as a list of individuals. Cohort counts are derived from their ages.
    sims: List['Sim'] = field(default_factory=list)
    population_capacity: int = 0

    # Per-building productivity (0-1) set by assign_work() from who staffs each building.
    _productivity: Dict[str, float] = field(default_factory=dict, repr=False)

    # Military
    units: Dict[str, int] = field(default_factory=dict)
    units_in_training: List[UnitInTraining] = field(default_factory=list)

    # The four levers this district steers its population with (see policies.py)
    policies: DistrictPolicies = field(default_factory=DistrictPolicies)

    # Fractional adults drafted so far, so L4 conscription needs no RNG
    conscription_progress: float = 0.0

    # Derived stats (calculated each turn)
    military_strength: float = 0.0
    happiness: float = BASE_HAPPINESS

    def __post_init__(self):
        """Initialize resources after dataclass initialization."""
        # Resources - Initialize ALL resource types
        self.resources = {
            # Starting resources (from config)
            'wood': STARTING_RESOURCES.get('wood', 100),
            'stone': STARTING_RESOURCES.get('stone', 80),
            'clay': STARTING_RESOURCES.get('clay', 60),
            'grain': STARTING_RESOURCES.get('grain', 50),
            'water': STARTING_RESOURCES.get('water', 100),

            # Processed resources (start at 0)
            'flour': 0,
            'bread': 0,
            'meat': 0,
            'vegetable': 0,
            'fruit': 0,
            'leather': 0,
            'wool': 0,
            'clay_bricks': 0,
            'wood_logs': 0,
            'stone_bricks': 0,
            'iron_ore': 0,
            'iron_ingots': 0,

            # War materiel (D29): an army must be EQUIPPED, so its kit is stock like any
            # other — it can be made, hoarded, and spent.
            'wooden_weapon': STARTING_RESOURCES.get('wooden_weapon', 0),
            'iron_weapon': 0,
            'armour': 0,

            # Clothes: the one derivative the PEOPLE consume rather than the army (D30).
            'clothes': 0,
        }

        # Seed edible food so the founders can eat while the food buildings go up.
        # Grain is NOT food (it must be milled and baked, or fed to cattle), so without
        # this the starting settlement would starve before its first bakery.
        for _food, _amount in STARTING_FOOD_STOCK.items():
            self.resources[_food] = self.resources.get(_food, 0) + _amount

        # Initialize starting buildings
        self.buildings = STARTING_BUILDINGS.copy()

        # Initialize starting population as an age pyramid (staggered births/deaths)
        self.seed_founding_population(STARTING_POPULATION)
        # Calculate capacity based on starting buildings
        self.calculate_population_capacity()

    @classmethod
    def found(cls, id: int, faction_id: int, location: Optional[int] = None) -> 'District':
        """A newly founded district: one warehouse, empty stock, no population.

        The plain constructor seeds the *starting* settlement instead, which is
        only correct for the district a faction begins the game with.
        """
        district = cls(id=id, faction_id=faction_id, location=location)
        district.resources = {resource: 0 for resource in district.resources}
        district.buildings = {'warehouse': 1}
        district.population = 0
        district.calculate_population_capacity()
        return district

    @property
    def has_warehouse(self) -> bool:
        """A district is anchored by its warehouse; losing it is losing the district."""
        return self.get_building_count('warehouse') > 0

    # ---- population: a list of individual sims, cohorts derived from age ----

    @property
    def population(self) -> int:
        return len(self.sims)

    @population.setter
    def population(self, value: int):
        """Legacy shim: set the population to `value` working-age adults.

        The whole codebase and test suite writes `population = n`. New adults are
        spread deterministically across the adult age band so they don't all age out
        together (no synchronised die-off), and without consuming the RNG.
        """
        self.sims = []
        self.add_adults(max(0, int(value)))

    @property
    def kids(self) -> int:
        return sum(1 for s in self.sims if s.is_kid)

    @property
    def adults(self) -> int:
        return sum(1 for s in self.sims if s.is_adult)

    @property
    def elders(self) -> int:
        return sum(1 for s in self.sims if s.is_elder)

    @property
    def adult_men(self) -> int:
        """Adults who can be recruited. Only men enlist (see `remove_adult`), so this —
        not `adults` — is the ceiling on how big an army this district can raise."""
        return sum(1 for s in self.sims if s.is_adult and s.gender is Gender.MALE)

    @staticmethod
    def _life_scale() -> float:
        """A per-sim multiplier on life-stage timing, so cohorts don't age out together."""
        return float(np.clip(np.random.normal(1.0, LIFE_SCALE_SIGMA), 0.5, 1.5))

    def add_adults(self, count: int):
        """Seed `count` adults with ages spread across the adult band (deterministic)."""
        for i in range(count):
            frac = (i + 0.5) / count if count else 0.5
            self.sims.append(Sim(age=KID_END + frac * ADULT_DURATION_SECONDS,
                                 life_scale=self._life_scale()))

    def seed_founding_population(self, count: int):
        """Seed a founding population as an age PYRAMID over the whole lifespan.

        Same-age adults retire and die in one wave; a stationary age distribution
        (uniform over [0, LIFESPAN)) plus per-sim life_scale staggers births and deaths
        from turn one — there are always kids maturing as elders die.
        """
        for i in range(count):
            self.sims.append(Sim(age=(i + 0.5) / count * LIFESPAN if count else 0.0,
                                 life_scale=self._life_scale()))

    def add_newborn(self, count: int = 1):
        """A birth. A sim starts at age 0 — not a worker for its (scaled) kid stage."""
        for _ in range(count):
            self.sims.append(Sim(age=0.0, life_scale=self._life_scale()))

    def age_sims(self, delta_time: float):
        """Age everyone one step; advance pregnancies (a birth when one completes); the
        too-old die. Membership in kid/adult/elder is just a function of age now."""
        from .config import FERTILITY_COOLDOWN_SECONDS
        newborns = 0
        for s in self.sims:
            s.age += delta_time
            if s.fertility_cooldown_remaining > 0.0:
                s.fertility_cooldown_remaining -= delta_time
            if s.pregnant_remaining > 0.0:
                s.pregnant_remaining -= delta_time
                if s.pregnant_remaining <= 0.0:
                    s.pregnant_remaining = 0.0
                    s.children_born += 1
                    s.fertility_cooldown_remaining = FERTILITY_COOLDOWN_SECONDS
                    newborns += 1
        if any(not s.is_alive for s in self.sims):
            self.sims = [s for s in self.sims if s.is_alive]
        if newborns:
            self.add_newborn(newborns)

    def working_sims(self) -> List['Sim']:
        """Sims who can work: adults and elders (kids don't)."""
        return [s for s in self.sims if s.is_adult or s.is_elder]

    def assign_work(self, delta_time: float):
        """Match sims to building slots, let them learn, and set each building's
        productivity from WHO works it — head-count and skill, not head-count alone.

        For each building type (food>resource>processing priority), working sims are
        matched to its slots, preferring those who already have (or can learn) its
        profession. Assigned sims adopt the profession and grow the skill — faster if an
        experienced coworker is present (mentorship), which is how skill accumulates
        across generations instead of every cohort restarting from zero. The building's
        productivity is (slots filled / slots) x (mean effective skill of its workers),
        where a novice contributes SKILL_FLOOR and an elder is scaled by ELDER_PRODUCTIVITY.
        """
        from .config import WORKERS_NEEDED, BUILDING_PRIORITIES
        from .professions import (
            BUILDING_PROFESSION, can_learn, Profession,
            SKILL_MAX, SKILL_FLOOR, MENTORSHIP_MULTIPLIER, SCHOOL_TRAINING_BONUS,
            skill_growth_per_second, PROFESSION_PREREQUISITE, SKILL_TRANSFER_FRACTION,
        )

        self._productivity = {}
        unassigned = self.working_sims()
        for s in unassigned:
            s.employed = False        # recomputed below; anyone left unset is unemployed
        if not unassigned:
            return

        # Schools first: each staffed school takes one adult as a teacher (who then
        # produces nothing) and speeds every other learner. The RL lever — invest an
        # adult + the build cost to raise the whole district's skill faster.
        n_schools = self.get_building_count('school')
        teachers = min(n_schools, len(unassigned))
        for s in unassigned[:teachers]:
            s.profession = Profession.TEACHER
            s.employed = True         # a teacher holds a job (produces nothing, but is not idle)
        unassigned = unassigned[teachers:]
        school_multiplier = 1.0 + SCHOOL_TRAINING_BONUS * teachers

        # A LEVEL-2+ building can only be worked by someone qualified for it; a LEVEL-1 job can
        # be worked by ANYONE. They are not competing for the same people, so they are filled
        # from two separate queues (D34):
        #
        #   pass 1 — the qualified go to the workshops only they can work (highest trade first)
        #   pass 2 — everyone else fills the ordinary jobs, food before the rest
        #
        # Filling from one queue starves whichever end goes second. Order matters both ways:
        # the clay pit is the potter's school, and a career ladder needs its bottom rung manned.
        from .professions import PROFESSION_TIER

        from .config import WORKERS_MIN

        def _slots(building: str) -> int:
            """The FULL crew this building type could take — unless the district has closed
            some of its slots (D35)."""
            full = self.get_building_count(building) * WORKERS_NEEDED[building]
            return min(full, self.open_slots.get(building, full))

        def _min_crew(building: str) -> int:
            """The smallest crew that can run it at all. Below this it produces nothing."""
            return min(_slots(building),
                       self.get_building_count(building) * WORKERS_MIN.get(building, 1))

        staffed = {b: [] for b in self.buildings
                   if self.get_building_count(b) > 0 and b in WORKERS_NEEDED}

        def _fill(building: str, only_qualified: bool, up_to=None):
            prof = BUILDING_PROFESSION.get(building, Profession.NONE)
            target = _slots(building) if up_to is None else up_to(building)
            room = target - len(staffed[building])
            if room <= 0:
                return
            # prefer whoever already does this job, then whoever could learn it
            unassigned.sort(key=lambda s: (s.profession != prof, not can_learn(prof, s.skills)))
            for sim in list(unassigned):
                if room <= 0:
                    break
                if prof is not Profession.NONE and not can_learn(prof, sim.skills):
                    continue        # this trade is not open to him yet
                if only_qualified and PROFESSION_TIER.get(prof, 0) < 2:
                    continue
                staffed[building].append(sim)
                unassigned.remove(sim)
                room -= 1

        by_trade = sorted(staffed, key=lambda b: -PROFESSION_TIER.get(
            BUILDING_PROFESSION.get(b, Profession.NONE), 0))
        by_priority = sorted(staffed, key=lambda b: BUILDING_PRIORITIES.get(b, 999))

        # ROUND ONE: a MINIMUM crew in every building, before ANY building gets a second man.
        # Filling to the maximum in priority order empties the pool before it reaches the
        # bottom of the list (D35).
        for building in by_trade:
            _fill(building, only_qualified=True, up_to=_min_crew)
        for building in by_priority:
            _fill(building, only_qualified=False, up_to=_min_crew)

        # ROUND TWO: whoever is left fattens the crews — the trades first, deepest first, then
        # the ordinary work, food first.
        for building in by_trade:
            _fill(building, only_qualified=True)
        for building in by_priority:
            _fill(building, only_qualified=False)

        base_growth = skill_growth_per_second() * delta_time * school_multiplier

        from .config import PREGNANCY_PRODUCTIVITY

        for building, assigned in staffed.items():
            if not assigned:
                continue
            if len(assigned) < _min_crew(building):
                continue        # a crew below its minimum cannot run the building at all
            for sim in assigned:
                sim.employed = True        # in a crew that actually runs: a real job
            prof = BUILDING_PROFESSION.get(building, Profession.NONE)
            slots = _slots(building)

            # Transferable expertise: a sim taking up this trade for the FIRST time, who already
            # has its prerequisite skill, starts partway up rather than as a novice — a master
            # digger is a competent potter. This is what makes climbing the ladder worthwhile.
            prereq = PROFESSION_PREREQUISITE.get(prof)
            if prereq is not None:
                for sim in assigned:
                    if prof not in sim.skills:
                        sim.skills[prof] = SKILL_TRANSFER_FRACTION * sim.skills.get(prereq, 0.0)

            # Mentorship: the most skilled worker present lifts everyone's learning rate. This
            # is how skill survives the generations instead of every cohort starting from zero.
            if prof is not Profession.NONE:
                top = max(s.skills.get(prof, 0.0) for s in assigned)
                for sim in assigned:
                    sim.profession = prof
                    rate = base_growth * (1.0 + MENTORSHIP_MULTIPLIER * top)
                    sim.skills[prof] = min(SKILL_MAX, sim.skills.get(prof, 0.0) + rate)

            # What the building actually yields: how many of its slots are manned, times how
            # good the people in them are. Half a crew is half the output — never all-or-nothing.
            eff = 0.0
            for sim in assigned:
                skill = 1.0 if prof is Profession.NONE else sim.skills.get(prof, 0.0)
                contrib = SKILL_FLOOR + (SKILL_MAX - SKILL_FLOOR) * skill
                if sim.is_elder:
                    contrib *= ELDER_PRODUCTIVITY
                if sim.is_pregnant:
                    contrib *= PREGNANCY_PRODUCTIVITY
                eff += contrib
            self._productivity[building] = min(1.0, eff / slots)

    def remove_adult(self) -> bool:
        """Conscript one adult MAN out of the population (training, conscription).

        Only men are recruited: keeping women civilian preserves the fertile base, so the
        population can regenerate while the army is raised. Without this a rusher converts the
        whole population into soldiers and the district dies with an army standing in it (D22).
        """
        for i, s in enumerate(self.sims):
            if s.is_adult and s.gender is Gender.MALE:
                self.sims.pop(i)
                return True
        return False

    def remove_people(self, count: int):
        """Deaths. The weakest go first: kids, then elders, then adults."""
        remaining = int(count)
        for keep in (Sim.is_kid, Sim.is_elder, Sim.is_adult):
            if remaining <= 0:
                break
            survivors = []
            for s in self.sims:
                if remaining > 0 and keep.fget(s):
                    remaining -= 1
                else:
                    survivors.append(s)
            self.sims = survivors

    def get_resource(self, resource: str) -> int:
        """Get amount of a resource (returns 0 if not found)."""
        return self.resources.get(resource, 0)

    def remove_resource(self, resource: str, amount: int):
        """Remove resource (safely handles missing keys)."""
        if resource in self.resources:
            self.resources[resource] = max(0, self.resources[resource] - amount)

    def add_resource(self, resource: str, amount: int):
        """Add resource (creates key if missing)."""
        if resource not in self.resources:
            self.resources[resource] = 0
        self.resources[resource] += amount

    def can_afford(self, costs: Dict[str, int]) -> bool:
        """Check if this district's stock covers the given costs."""
        for resource, amount in costs.items():
            if self.get_resource(resource) < amount:
                return False
        return True

    def deduct_costs(self, costs: Dict[str, int]) -> bool:
        """
        Deduct costs from resources.
        Returns True if successful, False if can't afford.
        """
        if not self.can_afford(costs):
            return False

        for resource, amount in costs.items():
            self.remove_resource(resource, amount)
        return True

    def get_building_count(self, building_type: str) -> int:
        """Get count of a building type."""
        return self.buildings.get(building_type, 0)

    def add_building(self, building_type: str, tile=None, quality: float = 1.0):
        """Add a completed building. On a map, it stands on a TILE (D27).

        `tile` is optional so that a mapless game — and every test written before the map
        existed — still means exactly what it used to.
        """
        self.buildings[building_type] = self.get_building_count(building_type) + 1
        if tile is not None:
            self.building_tiles[tile] = building_type

    def remove_building(self, building_type: str) -> bool:
        """Remove a building if it exists. On a map, the WORST-placed one is lost first —
        a razed building frees its ground, and what remains is the good land."""
        if self.get_building_count(building_type) <= 0:
            return False

        self.buildings[building_type] -= 1

        mine = [t for t, b in self.building_tiles.items() if b == building_type]
        if mine:
            worst = min(mine, key=lambda t: self._tile_quality(t, building_type))
            del self.building_tiles[worst]
            if self._map is not None:
                self._map.clear(building_type, *worst)      # frees the whole footprint
        return True

    def _tile_quality(self, tile, building_type: str) -> float:
        if self._map is None:
            return 1.0
        q = self._map.placement_quality(building_type, *tile)
        # the tile is occupied BY this building, so quality() reports it unbuildable;
        # recompute as if it were free
        if q is None:
            occupant = self._map.tiles[tile[0]][tile[1]].building
            self._map.tiles[tile[0]][tile[1]].building = None
            q = self._map.placement_quality(building_type, *tile) or 1.0
            self._map.tiles[tile[0]][tile[1]].building = occupant
        return q


    def get_unit_count(self, unit_type: str) -> int:
        """Get count of a unit type."""
        return self.units.get(unit_type, 0)

    def add_unit(self, unit_type: str, count: int = 1):
        """Add units."""
        self.units[unit_type] = self.get_unit_count(unit_type) + count

    def remove_units(self, unit_type: str, count: int) -> bool:
        """Remove units if available."""
        if self.get_unit_count(unit_type) >= count:
            self.units[unit_type] = self.get_unit_count(unit_type) - count
            return True
        return False

    def calculate_military_strength(self) -> float:
        """Calculate total military strength."""
        total = 0.0
        for unit_type, count in self.units.items():
            strength = UNIT_STRENGTH.get(unit_type, 0.0)
            total += count * strength
        self.military_strength = total
        return total

    def get_total_food(self) -> float:
        """Everything the population can eat. Grain and flour are intermediates, not food."""
        return sum(self.get_resource(f) for f in EDIBLE_FOODS)

    def get_food_variety(self) -> int:
        """How many distinct edible foods are actually in stock."""
        return sum(1 for f in EDIBLE_FOODS if self.get_resource(f) > 0)

    def get_mean_skill(self) -> float:
        """Mean best-skill of working sims (0-1): the district's human capital, the
        state the school lever invests in."""
        working = [s for s in self.sims if s.is_adult or s.is_elder]
        if not working:
            return 0.0
        return sum(max(s.skills.values()) if s.skills else 0.0 for s in working) / len(working)

    def get_food_per_capita(self) -> float:
        return self.get_total_food() / max(1, self.population)

    def unemployment_rate(self) -> float:
        """Share of working-age sims with no job (0-1), as set by the last assign_work.

        Kids are excluded — they are not expected to work. With no working-age sims the
        rate is 0 (an empty district is not 'unemployed')."""
        working = [s for s in self.sims if s.is_adult or s.is_elder]
        if not working:
            return 0.0
        idle = sum(1 for s in working if not s.employed)
        return idle / len(working)

    def calculate_happiness(self) -> float:
        """Recompute happiness from current conditions and active policies.

        Derived, not integrated: happiness is a pure function of the district's
        present state, so it recovers the instant the conditions do. The game
        integrates it per-sim (`update_stats_from_needs`); this is the aggregate
        shadow of that, and the two are reconciled by the calibration protocol in
        docs/population_and_policies.md §7.

        Only penalties, never bonuses — the same shape as the game's model.
        """
        happiness = BASE_HAPPINESS

        if self.get_total_food() <= 0:
            happiness -= STARVATION_HAPPINESS_LOSS

        # A varied diet lifts happiness, TIERED: each distinct food beyond the first
        # adds a bonus, capped. This is the reward for running several food sources
        # rather than the single cheapest one — and what makes a low-calorie luxury
        # like fruit worth building.
        variety = self.get_food_variety()
        if variety >= MIN_FOOD_TYPES_FOR_VARIETY:
            happiness += min(FOOD_VARIETY_BONUS_CAP, (variety - 1) * FOOD_VARIETY_BONUS_PER_TYPE)

        # Clothes on people's backs (D30). The tailor's line ends HERE, in the population,
        # not in a warehouse: it is the one derivative the district makes for itself rather
        # than for its army. Without a sink like this, wool and leather were scenery.
        if self.get_resource('clothes') > 0:
            happiness += CLOTHES_HAPPINESS_BONUS

        # Idle hands are unhappy hands. The malus is proportional to the share of working-age
        # sims with no job, so a district that grows past the work it has to offer feels it —
        # this is what couples the population back to the DEMAND FOR LABOUR (D37): unemployment
        # lowers happiness, and happiness is the main gate on births.
        happiness -= UNEMPLOYMENT_HAPPINESS_PENALTY * self.unemployment_rate()

        # happiness_cost() is non-positive; a policy always costs
        happiness += self.policies.happiness_cost()

        self.happiness = max(0.0, min(100.0, happiness))
        return self.happiness

    def calculate_population_capacity(self):
        """Calculate total population capacity from buildings."""
        from simulator.config import POPULATION_CAPACITY_PER_HOUSE, POPULATION_CAPACITY_PER_DORMITORY, \
            STARTING_POPULATION_CAPACITY

        base_capacity = STARTING_POPULATION_CAPACITY
        house_capacity = self.get_building_count('house') * POPULATION_CAPACITY_PER_HOUSE
        dormitory_capacity = self.get_building_count('dormitory') * POPULATION_CAPACITY_PER_DORMITORY

        self.population_capacity = base_capacity + house_capacity + dormitory_capacity
        return self.population_capacity

    def get_available_workers(self) -> float:
        """Effective labour force: civilian adults, plus elders at reduced output.

        Kids do not work at all. Soldiers are NOT subtracted here: enlisting already
        removed them from `adults` (see TrainingManager). The old formula was
        `population - military`, which both counted every head as a working adult AND
        charged each soldier twice — so an agent with 7 soldiers out of 10 people had
        *zero* workers and no economy at all. Pregnant women work at reduced output.
        """
        from .config import PREGNANCY_PRODUCTIVITY
        labour = 0.0
        for s in self.sims:
            if s.is_adult:
                labour += PREGNANCY_PRODUCTIVITY if s.is_pregnant else 1.0
            elif s.is_elder:
                labour += ELDER_PRODUCTIVITY
        return labour

    def get_total_workers_needed(self) -> int:
        """Calculate total workers needed for all buildings"""
        from simulator.config import WORKERS_NEEDED
        total = 0
        for building_type, count in self.buildings.items():
            if building_type in WORKERS_NEEDED:
                total += WORKERS_NEEDED[building_type] * count
        return total

    def calculate_worker_assignments(self) -> Dict[str, int]:
        """
        Distribute workers by priority: food > resource > processing
        Returns: {building_type: workers_assigned}
        """
        from simulator.config import WORKERS_NEEDED, BUILDING_PRIORITIES

        available_workers = self.get_available_workers()
        assignments = {}

        # Build priority queue: (priority, building_type, count, workers_per)
        building_demands = []
        for building_type, count in self.buildings.items():
            if count > 0 and building_type in WORKERS_NEEDED:
                priority = BUILDING_PRIORITIES.get(building_type, 999)
                workers_per = WORKERS_NEEDED[building_type]
                building_demands.append((priority, building_type, count, workers_per))

        # Sort by priority (lower = higher priority)
        building_demands.sort(key=lambda x: x[0])

        # Distribute workers
        remaining_workers = available_workers
        for priority, building_type, count, workers_per in building_demands:
            total_needed = count * workers_per
            assigned = min(total_needed, remaining_workers)
            assignments[building_type] = assigned
            remaining_workers -= assigned
            if remaining_workers <= 0:
                break

        return assignments

    def get_building_productivity(self, building_type: str) -> float:
        """Productivity (0-1), set by the last assign_work() from WHO staffs this building.

        Head-count AND skill: a building run by masters outproduces one run by novices.
        Buildings that need no workers run at 100%. A building with no assigned workers
        produces nothing. `assign_work()` runs each step before production, so this is
        always fresh during simulation; call it first if you query outside the loop.
        """
        from simulator.config import WORKERS_NEEDED

        if self.get_building_count(building_type) == 0:
            return 0.0
        if building_type not in WORKERS_NEEDED:
            return 1.0
        if not self._productivity:                # assign_work not run yet (e.g. a unit test)
            self.assign_work(0.0)
        return self._productivity.get(building_type, 0.0)


@dataclass
class Faction:
    """One player/AI faction: a container of districts.

    Every attribute is an aggregate over `districts`. INVARIANT: while a faction holds a
    single district, the aggregates ARE that district's own containers, by identity — not
    copies. Callers mutate them in place, so returning a copy here silently drops writes.
    """
    id: int
    districts: List['District'] = field(default_factory=list)

    def __post_init__(self):
        if not self.districts:
            self.districts = [District(id=0, faction_id=self.id)]
        for district in self.districts:
            district.faction_id = self.id

    # ---- district access -------------------------------------------------

    @property
    def capital(self) -> 'District':
        """The district that faction-level writes are routed to."""
        return self.districts[0]

    def get_district(self, district_id: int) -> Optional['District']:
        for district in self.districts:
            if district.id == district_id:
                return district
        return None

    def add_district(self, district: 'District') -> 'District':
        """Found a district. Each district is seeded by exactly one warehouse."""
        district.faction_id = self.id
        self.districts.append(district)
        return district

    def _sole(self, what: str) -> 'District':
        """The single district, for legacy faction-level writes."""
        if len(self.districts) != 1:
            raise TypeError(
                f"Faction.{what} is an aggregate over {len(self.districts)} "
                f"districts and cannot be assigned. Write to a specific "
                f"district: faction.districts[i].{what} = ..."
            )
        return self.districts[0]

    # ---- aggregate containers -------------------------------------------
    # With one district these ARE the district's containers, so in-place writes
    # (faction.resources['wood'] += 10) land on the district.

    @property
    def resources(self) -> Dict[str, int]:
        if len(self.districts) == 1:
            return self.districts[0].resources
        return _AggregateMapping(self._sum_dicts('resources'), 'resources')

    @property
    def buildings(self) -> Dict[str, int]:
        if len(self.districts) == 1:
            return self.districts[0].buildings
        return _AggregateMapping(self._sum_dicts('buildings'), 'buildings')

    @property
    def units(self) -> Dict[str, int]:
        if len(self.districts) == 1:
            return self.districts[0].units
        return _AggregateMapping(self._sum_dicts('units'), 'units')

    @property
    def buildings_in_progress(self) -> List[BuildingInProgress]:
        if len(self.districts) == 1:
            return self.districts[0].buildings_in_progress
        return [b for d in self.districts for b in d.buildings_in_progress]

    @property
    def units_in_training(self) -> List[UnitInTraining]:
        if len(self.districts) == 1:
            return self.districts[0].units_in_training
        return [u for d in self.districts for u in d.units_in_training]

    def _sum_dicts(self, attr: str) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for district in self.districts:
            for key, value in getattr(district, attr).items():
                totals[key] = totals.get(key, 0) + value
        return totals

    # ---- aggregate scalars ----------------------------------------------

    @property
    def population(self) -> int:
        return sum(d.population for d in self.districts)

    @population.setter
    def population(self, value: int):
        self._sole('population').population = value

    @property
    def kids(self) -> int:
        return sum(d.kids for d in self.districts)

    @property
    def adults(self) -> int:
        return sum(d.adults for d in self.districts)

    @property
    def elders(self) -> int:
        return sum(d.elders for d in self.districts)

    @property
    def adult_men(self) -> int:
        return sum(d.adult_men for d in self.districts)

    @property
    def population_capacity(self) -> int:
        return sum(d.population_capacity for d in self.districts)

    @population_capacity.setter
    def population_capacity(self, value: int):
        self._sole('population_capacity').population_capacity = value

    @property
    def military_strength(self) -> float:
        return sum(d.military_strength for d in self.districts)

    @military_strength.setter
    def military_strength(self, value: float):
        self._sole('military_strength').military_strength = value

    @property
    def happiness(self) -> float:
        """Population-weighted mean across districts (a policy hurts more where more live)."""
        people = self.population
        if people <= 0:
            return sum(d.happiness for d in self.districts) / len(self.districts)
        return sum(d.happiness * d.population for d in self.districts) / people

    @happiness.setter
    def happiness(self, value: float):
        self._sole('happiness').happiness = value

    @property
    def policies(self) -> DistrictPolicies:
        """Policies are per-district; there is no faction-wide policy object."""
        return self._sole('policies').policies

    # ---- aggregate reads -------------------------------------------------

    def get_resource(self, resource: str) -> int:
        """Total stock across districts (they are not pooled; Stage 4 adds transfers)."""
        return sum(d.get_resource(resource) for d in self.districts)

    def get_building_count(self, building_type: str) -> int:
        return sum(d.get_building_count(building_type) for d in self.districts)

    def get_unit_count(self, unit_type: str) -> int:
        return sum(d.get_unit_count(unit_type) for d in self.districts)

    def get_total_food(self) -> float:
        return sum(d.get_total_food() for d in self.districts)

    def get_food_variety(self) -> int:
        """Distinct edible foods in stock anywhere in the faction."""
        return sum(
            1 for f in EDIBLE_FOODS
            if any(d.get_resource(f) > 0 for d in self.districts)
        )

    def can_afford(self, costs: Dict[str, int]) -> bool:
        """True if SOME district can pay from its own stock."""
        return any(d.can_afford(costs) for d in self.districts)

    def district_that_can_afford(self, costs: Dict[str, int]) -> Optional['District']:
        """First district whose own stock covers `costs`."""
        for district in self.districts:
            if district.can_afford(costs):
                return district
        return None

    def get_available_workers(self) -> float:
        return sum(d.get_available_workers() for d in self.districts)

    def get_total_workers_needed(self) -> int:
        return sum(d.get_total_workers_needed() for d in self.districts)

    def calculate_worker_assignments(self) -> Dict[str, int]:
        """Worker assignments summed across districts (each staffs its own buildings)."""
        totals: Dict[str, int] = {}
        for district in self.districts:
            for building_type, assigned in district.calculate_worker_assignments().items():
                totals[building_type] = totals.get(building_type, 0) + assigned
        return totals

    def get_building_productivity(self, building_type: str) -> float:
        """Staffing ratio for this building type across every district that has one."""
        from simulator.config import WORKERS_NEEDED

        if self.get_building_count(building_type) == 0:
            return 0.0
        if building_type not in WORKERS_NEEDED:
            return 1.0

        needed = sum(
            d.get_building_count(building_type) * WORKERS_NEEDED[building_type]
            for d in self.districts
        )
        if needed == 0:
            return 1.0

        assigned = sum(
            d.calculate_worker_assignments().get(building_type, 0)
            for d in self.districts
        )
        return min(1.0, assigned / needed)

    # ---- faction-level writes (routed to the capital) ---------------------

    def add_resource(self, resource: str, amount: int):
        self.capital.add_resource(resource, amount)

    def remove_resource(self, resource: str, amount: int):
        self.capital.remove_resource(resource, amount)

    def deduct_costs(self, costs: Dict[str, int]) -> bool:
        """Pay from the first district that can afford it."""
        district = self.district_that_can_afford(costs)
        if district is None:
            return False
        return district.deduct_costs(costs)

    def add_building(self, building_type: str):
        self.capital.add_building(building_type)

    def remove_building(self, building_type: str) -> bool:
        for district in self.districts:
            if district.remove_building(building_type):
                return True
        return False

    def add_unit(self, unit_type: str, count: int = 1):
        self.capital.add_unit(unit_type, count)

    def remove_units(self, unit_type: str, count: int) -> bool:
        for district in self.districts:
            if district.remove_units(unit_type, count):
                return True
        return False

    # ---- derived stats ----------------------------------------------------

    def calculate_military_strength(self) -> float:
        return sum(d.calculate_military_strength() for d in self.districts)

    def calculate_population_capacity(self) -> int:
        return sum(d.calculate_population_capacity() for d in self.districts)

    def is_defeated(self) -> bool:
        """Defeated once every warehouse is gone (i.e. every district has fallen)."""
        return not any(d.has_warehouse for d in self.districts)


@dataclass
class MarchingArmy:
    """A column of units on the road: it fights when it ARRIVES, and it is NOT AT HOME.

    Placeholder for real map movement (there is no spatial layer yet — D23): the distance
    is one fixed march, not a path. Two things follow, and they are the point:

    - an attack is not instantaneous, so an attacker must commit against the state it will
      MEET, not the one it left (D25);
    - the army that leaves is GONE from the district that raised it (D26). It carries its
      units with it, so it cannot defend home while it is away, and the soldiers trained
      behind it do not teleport to the battle. Leaving is a risk, and a raid on an army
      that is out is a real move.

    `returning` marks the survivors walking back: they fight nothing, they just come home.
    """
    attacker_id: int
    defender_id: int
    arrival_time: float
    units: Dict[str, int] = field(default_factory=dict)
    returning: bool = False


@dataclass
class GameState:
    """Represents the complete game state.

    Real-time: there is no turn counter (state advances by `delta_time` seconds). The
    old `turn` field was never read or incremented after the turn-based scorer was
    removed, so it is gone.
    """
    factions: List[Faction] = field(default_factory=list)
    game_over: bool = False
    winner: Optional[int] = None

    # The world every tier shares: one tile grid (D27). `None` = a mapless game, which the
    # simulator still supports so that the old, non-spatial tests keep meaning something.
    map: Optional['GameMap'] = None

    # Armies in transit. This is STATE, not simulator bookkeeping: a march is visible
    # (an opponent sees the column coming), and a bot must be able to ask "is my army
    # already out?" before ordering another attack.
    marching_armies: List['MarchingArmy'] = field(default_factory=list)

    def is_marching(self, faction_id: int) -> bool:
        """True while this faction's army is on the road and cannot be re-committed."""
        return any(a.attacker_id == faction_id for a in self.marching_armies)

    def get_faction(self, faction_id: int) -> Optional[Faction]:
        """Get a faction by ID."""
        for faction in self.factions:
            if faction.id == faction_id:
                return faction
        return None

    def check_game_over(self):
        """The only victory is conquest: outlast every rival's last warehouse.

        There is no scored victory. Running out of time is not a result — it is the
        absence of one, and the environment reports it as a truncation so that the
        value function bootstraps instead of learning that the world ends.

        Do not reintroduce a development score to break the tie: it rewards an army that is
        never spent, and the agent learns to hoard rather than to conquer (D12/D13).
        """
        # A district playing ALONE (the civil tier, D32) has no war to win. Without this the
        # conquest check looks at the single surviving faction, declares it the winner, and
        # ends the episode on step zero — the settlement was pronounced victorious before it
        # had laid a single brick.
        if len(self.factions) < 2:
            return

        alive_factions = [f for f in self.factions if not f.is_defeated()]

        if len(alive_factions) == 1:
            self.game_over = True
            self.winner = alive_factions[0].id
        elif len(alive_factions) == 0:
            # Mutual annihilation: a draw, and nobody is paid for it.
            self.game_over = True
            self.winner = None
