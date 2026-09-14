"""The world: one tile grid, shared by every tier. See docs/design_decisions.md D27.

Mirrors the Rust game (`src/resources/map.rs`, `src/components/map/tile.rs`,
`game_data/map.json`): a grid of `TileType` with an elevation field. The game's map is
300x300; the training map is smaller, because what has to be faithful is the RATIO between a
district's radius and the distance between rivals, not the tile count (D13).

Deposits do not exist in the Rust game yet — its `spawners/resources/veins.rs` is entirely
commented out. The sim defines them first and the game can follow.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    MAP_WIDTH, MAP_HEIGHT, DISTRICT_RADIUS, DEPOSIT_DENSITY, SETTLEMENT_SEPARATION,
    SEA_LEVEL, MOUNTAIN_THRESHOLD, ELEVATION_MAX,
)


class NoViableSite(RuntimeError):
    """This map cannot seat a district that could develop. Generate another one."""


class TileType(Enum):
    """Mirrors `TileType` in the Rust game (src/components/map/tile.rs)."""
    GRASS = 'grass'
    WATER = 'water'
    MOUNTAIN = 'mountain'
    FOREST = 'forest'
    SAND = 'sand'
    SNOW = 'snow'
    ROAD = 'road'
    EMPTY = 'empty'


class DepositType(Enum):
    """Resource veins.

    GOLD is deliberately absent: nothing produces or consumes it, so a gold vein would invite
    the district to spend real miners on something it cannot spend. It comes back the day
    there is something to buy with it (D30).
    """
    TREE = 'tree'
    STONE = 'stone'
    CLAY = 'clay'
    IRON = 'iron'


#: Terrain you cannot put a building on at all.
UNBUILDABLE = {TileType.WATER, TileType.MOUNTAIN}

#: Which terrain each deposit can appear on.
DEPOSIT_TERRAIN = {
    DepositType.TREE: {TileType.FOREST},
    DepositType.STONE: {TileType.MOUNTAIN},
    DepositType.IRON: {TileType.MOUNTAIN},
    DepositType.CLAY: {TileType.SAND, TileType.GRASS},
}

#: What a building wants, and whether it is a HARD requirement.
#: (deposit_or_terrain, required) — `required` means the building cannot exist without it.
#: A building with no entry here can go on any buildable tile: it does not care where it is.
#:
#: The split is deliberate. Everyday production hangs off TERRAIN, which is abundant, so a
#: district on decent land can run a normal economy: geography constrains it without taxing
#: it. DEPOSITS are for what is scarce and strategic — iron is somewhere specific or it is
#: nowhere, and that is what makes one plot worth fighting over.
#:
#: A WELL is deliberately absent: it is dug into the ground, the water table is under every
#: tile alike, so it yields the same everywhere.
BUILDING_NEEDS: Dict[str, Tuple[object, bool]] = {
    'lumberyard':     (TileType.FOREST,   False),   # you fell trees in a forest
    'quarry':         (TileType.MOUNTAIN, False),   # you cut stone at the mountainside
    'clay_pit':       (DepositType.CLAY,  False),
    'mine':           (DepositType.IRON,  True),    # no vein, no mine — a real prerequisite
    'farm':           (TileType.GRASS,    False),
    'vegetable_garden': (TileType.GRASS,  False),
    'orchard':        (TileType.GRASS,    False),
    'hunting_shed':   (TileType.FOREST,   False),   # you hunt where the game is
    'cattle_shed':    (TileType.GRASS,    False),
}

#: Ore comes in seams, woods come in stands. These shape the CLUSTERS: how big a seam tends
#: to be, and how eagerly it spreads. Uniformly sprinkled deposits would make every plot
#: statistically identical, and then WHERE could not matter.
_CLUSTER_SIZE = 14.0        # mean tiles per seam
_CLUSTER_SPREAD = 0.6       # chance a seam reaches into a neighbouring tile

#: The SHAPE of the world, as proportions. These stay fixed when the map grows.
_WATER_SHARE = 0.14         # lakes and coast
_MOUNTAIN_SHARE = 0.16      # ridges (unbuildable, but you quarry and mine at their foot)
_SNOW_SHARE = 0.03          # the highest peaks
_FOREST_SHARE = 0.28        # woods: timber, game
_SAND_SHARE = 0.12          # dry ground: poor for crops, good for clay

#: Yield of a building placed with NO access to what it wants. Placement no longer taxes
#: yield (yield is workers x skill), but this floor still RANKS candidate tiles for the
#: auto-placer and picks which building to raze first.
PLACEMENT_YIELD_FLOOR = 0.5

#: How much ground each building occupies, as a (w, h) rectangle of tiles. Straight from the
#: game's size classes (game_data/building.json `area_size`: small 2x2, medium 3x3, warehouse
#: 3x3). EVERY building takes more than one tile, so a plot holds only a bounded number of
#: them — which is what makes MAP SPACE the real limit on how large a district can grow, rather
#: than a labour or food equilibrium. A district must therefore choose what to fit on its land.
_AREA_SIZE = {'small': (2, 2), 'medium': (3, 3), 'large': (4, 4), 'warehouse': (3, 3)}
_FOOTPRINT_CLASS = {
    'warehouse': 'warehouse',
    # the big compounds: a farm's fields, a bakery, a stockyard, the school and the barracks
    'farm': 'medium', 'bakery': 'medium', 'cattle_shed': 'medium', 'blacksmith': 'medium',
    'smelter': 'medium', 'school': 'medium', 'barracks': 'medium', 'dormitory': 'medium',
    # everything else (extractors, workshops, houses, gardens) is a small 2x2 plot
}


def footprint(building_type: str) -> Tuple[int, int]:
    """The (w, h) tile rectangle a building occupies."""
    return _AREA_SIZE[_FOOTPRINT_CLASS.get(building_type, 'small')]


@dataclass
class Tile:
    terrain: TileType
    elevation: float
    deposit: Optional[DepositType] = None
    richness: float = 0.0
    #: (faction_id, building_type) of whatever stands here, or None.
    building: Optional[Tuple[int, str]] = None

    @property
    def is_buildable(self) -> bool:
        return self.terrain not in UNBUILDABLE and self.building is None


@dataclass
class GameMap:
    """A grid of tiles. Coordinates are (x, y), origin top-left, as in the game."""
    width: int
    height: int
    tiles: List[List[Tile]] = field(default_factory=list)

    # ---- basic access ----

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def at(self, x: int, y: int) -> Optional[Tile]:
        return self.tiles[x][y] if self.in_bounds(x, y) else None

    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Four-connected, like `Map::get_neighbors` in Rust."""
        return [
            (x + dx, y + dy)
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0))
            if self.in_bounds(x + dx, y + dy)
        ]

    def tiles_within(self, center: Tuple[int, int], radius: int):
        """Every tile within `radius` (Chebyshev) of a centre — a district's plot."""
        cx, cy = center
        for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
            for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
                yield (x, y), self.tiles[x][y]

    @staticmethod
    def distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance in tiles. This is what a march has to cover."""
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # ---- placement ----

    def placement_quality(self, building_type: str, x: int, y: int) -> Optional[float]:
        """How good is this tile for this building?

        Returns a yield multiplier in [PLACEMENT_YIELD_FLOOR, 1.0], or None if the building
        cannot legally stand here at all: an unbuildable or occupied tile, or a hard
        requirement the tile cannot meet (a mine with no vein).

        Quality comes from what the tile SITS ON and what it TOUCHES: a lumberyard on a rich
        tree vein is at 1.0, one merely next to the forest is worth less, one on bare grass is
        at the floor.
        """
        tile = self.at(x, y)
        if tile is None or not tile.is_buildable:
            return None

        need = BUILDING_NEEDS.get(building_type)
        if need is None:
            return 1.0          # this building does not care where it stands

        wanted, required = need
        best = 0.0

        if isinstance(wanted, DepositType):
            # A vein is worked from on top of it; richness is how much is down there.
            if tile.deposit is wanted:
                best = max(best, 0.7 + 0.3 * tile.richness)
            for nx, ny in self.neighbors(x, y):
                n = self.tiles[nx][ny]
                if n.deposit is wanted:
                    best = max(best, 0.8 * (0.7 + 0.3 * n.richness))
        elif wanted in UNBUILDABLE:
            # You cannot stand ON water or on a cliff — you build BESIDE it. Being next to
            # it IS the ideal placement for a well or a quarry, not a compromise.
            if any(self.tiles[nx][ny].terrain is wanted for nx, ny in self.neighbors(x, y)):
                best = 1.0
        else:
            # Ordinary ground: be on it, or at worst beside it.
            if tile.terrain is wanted:
                best = 1.0
            elif any(self.tiles[nx][ny].terrain is wanted for nx, ny in self.neighbors(x, y)):
                best = 0.8

        if best <= 0.0:
            return None if required else PLACEMENT_YIELD_FLOOR

        return PLACEMENT_YIELD_FLOOR + (1.0 - PLACEMENT_YIELD_FLOOR) * min(1.0, best)

    def footprint_cells(self, building_type: str, x: int, y: int):
        """The tiles a building occupies when its top-left corner is at (x, y)."""
        w, h = footprint(building_type)
        return [(x + dx, y + dy) for dx in range(w) for dy in range(h)]

    def footprint_fits(self, building_type: str, x: int, y: int,
                       center: Optional[Tuple[int, int]] = None,
                       radius: int = DISTRICT_RADIUS) -> bool:
        """Can this building's whole rectangle stand with its corner at (x, y)? Every tile it
        would cover must be on the map, inside the plot, buildable and free. A building with a
        hard deposit requirement (a mine) also needs that deposit UNDER its footprint."""
        cells = self.footprint_cells(building_type, x, y)
        for cx, cy in cells:
            if not self.in_bounds(cx, cy):
                return False
            if center is not None and max(abs(cx - center[0]), abs(cy - center[1])) > radius:
                return False
            if not self.tiles[cx][cy].is_buildable:
                return False
        need = BUILDING_NEEDS.get(building_type)
        if need is not None:
            wanted, required = need
            if required and isinstance(wanted, DepositType):
                if not any(self.tiles[cx][cy].deposit is wanted for cx, cy in cells):
                    return False
        return True

    def best_tile_for(self, building_type: str, center: Tuple[int, int],
                      radius: int = DISTRICT_RADIUS) -> Optional[Tuple[int, int]]:
        """A free SITE (top-left corner) whose footprint fits in the district's plot, chosen to
        PACK the plot rather than scatter across it.

        Buildings now cover rectangles, so the plot holds only a bounded number of them — and if
        they were dropped on the best-looking tiles they would leave the land full of gaps too
        small for the next 3x3 workshop, wasting ground the district needs. So sites are filled
        in a fixed row-major order (bottom-left first): buildings cluster and the free space
        stays CONTIGUOUS, leaving room for the big compounds. When no rectangle fits, this
        returns None and the build cannot happen — that is how map space bounds the district.
        """
        # Pack by SIZE, from opposite corners: the big compounds (3x3+) fill from the top-left,
        # the small 2x2 lots from the bottom-right. Segregating the two sizes keeps the big
        # buildings' end of the plot free of the 2x2 gaps that would otherwise leave no room for
        # a 3x3 — the classic mixed-size fragmentation. Each side still packs tightly.
        w, h = footprint(building_type)
        big = max(w, h) >= 3
        best_key = None
        best = None
        for (x, y), _tile in self.tiles_within(center, radius):
            if not self.footprint_fits(building_type, x, y, center, radius):
                continue
            key = (y, x) if big else (-y, -x)    # big: top-left first; small: bottom-right first
            if best_key is None or key < best_key:
                best, best_key = (x, y), key
        return best

    def place(self, faction_id: int, building_type: str, x: int, y: int):
        """Occupy the building's whole footprint, anchored at (x, y)."""
        for cx, cy in self.footprint_cells(building_type, x, y):
            self.tiles[cx][cy].building = (faction_id, building_type)

    def clear(self, building_type: str, x: int, y: int):
        """Free the whole footprint of the building anchored at (x, y)."""
        for cx, cy in self.footprint_cells(building_type, x, y):
            if self.in_bounds(cx, cy):
                self.tiles[cx][cy].building = None


# --- generation ---------------------------------------------------------------

def _value_noise(width: int, height: int, octaves: int = 4) -> np.ndarray:
    """Smooth noise in [0,1]. Coherent, so terrain comes in BLOBS.

    Independent per-tile draws produce salt-and-pepper: forest that is never a forest, lakes
    one tile wide. A few upsampled low-res grids added together give woods, ridges and lakes,
    which is what makes WHERE a real question.
    """
    acc = np.zeros((width, height), dtype=np.float64)
    amplitude, total = 1.0, 0.0
    for o in range(octaves):
        res = max(2, int(2 ** (o + 1)))
        coarse = np.random.random((res, res))
        # nearest-neighbour upsample to full size
        xi = (np.arange(width) * res // max(1, width)).clip(0, res - 1)
        yi = (np.arange(height) * res // max(1, height)).clip(0, res - 1)
        acc += amplitude * coarse[np.ix_(xi, yi)]
        total += amplitude
        amplitude *= 0.5
    noise = acc / total
    # box-blur once to soften the upsampling steps
    padded = np.pad(noise, 1, mode='edge')
    smooth = sum(
        padded[1 + dx: 1 + dx + width, 1 + dy: 1 + dy + height]
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
    ) / 9.0
    lo, hi = smooth.min(), smooth.max()
    return (smooth - lo) / (hi - lo + 1e-9)


def rotate(width: int, height: int, x: int, y: int) -> Tuple[int, int]:
    """The point opposite (x, y) through the centre of the map."""
    return width - 1 - x, height - 1 - y


def generate_map(width: int = MAP_WIDTH, height: int = MAP_HEIGHT,
                 symmetric: bool = True) -> GameMap:
    """Generate a world. Draws from the GLOBAL numpy RNG, so `env.reset(seed)` fixes it.

    Terrain follows the game's elevation model (`game_data/map.json`): water at or below sea
    level, mountains above the mountain threshold, and in between the land is forest or sand
    or grass depending on moisture.

    The map is rotationally symmetric so that the two starts are the SAME ground and a mirror
    match is a coin flip by construction rather than by hope — on a random map, whoever
    settles first takes the better ground, and that head-start would silently poison every
    measurement made on this simulator (D27).
    """
    elevation = _value_noise(width, height) * ELEVATION_MAX
    moisture = _value_noise(width, height)

    if symmetric:
        # Fold the NOISE, before anything is decided from it. Folding the finished tiles
        # instead would make the quantile cuts below a lie: they would be computed over a
        # world that is then half thrown away, so the map's real proportions would depend on
        # which half won.
        elevation = _fold(elevation)
        moisture = _fold(moisture)

    # Cut by QUANTILE, not by an absolute threshold on the noise: smoothed noise does not keep
    # its distribution when the grid changes size, so a fixed threshold silently rewrites the
    # world the day the map grows. Quantiles hold the world's PROPORTIONS fixed (D13).
    sea = np.quantile(elevation, _WATER_SHARE)
    mountain = np.quantile(elevation, 1.0 - _MOUNTAIN_SHARE)
    snow = np.quantile(elevation, 1.0 - _SNOW_SHARE)
    wet = np.quantile(moisture, 1.0 - _FOREST_SHARE)
    dry = np.quantile(moisture, _SAND_SHARE)

    tiles: List[List[Tile]] = []
    for x in range(width):
        column = []
        for y in range(height):
            e = float(elevation[x, y])
            m = float(moisture[x, y])

            if e <= sea:
                terrain = TileType.WATER
            elif e >= snow:
                terrain = TileType.SNOW
            elif e >= mountain:
                terrain = TileType.MOUNTAIN
            elif m >= wet:
                terrain = TileType.FOREST
            elif m <= dry:
                terrain = TileType.SAND
            else:
                terrain = TileType.GRASS

            column.append(Tile(terrain=terrain, elevation=e))
        tiles.append(column)

    game_map = GameMap(width=width, height=height, tiles=tiles)
    _scatter_deposits(game_map)
    if symmetric:
        _mirror(game_map)     # the seams are scattered at random: fold them too
    return game_map


def _fold(field: np.ndarray) -> np.ndarray:
    """Make a field rotationally symmetric: the second half becomes the first, turned 180°."""
    folded = field.copy()
    w, h = field.shape
    for y in range(h):
        if 2 * y >= h:
            folded[:, y] = field[::-1, h - 1 - y]
    return folded


def _mirror(game_map: GameMap):
    """Copy each tile of the first half onto the point opposite it, so the two halves are
    the same ground. Deposits and richness travel with the terrain: a vein one player can
    reach, the other can reach too."""
    w, h = game_map.width, game_map.height
    for x in range(w):
        for y in range(h):
            if 2 * y >= h:          # source half: the top; the bottom is its reflection
                continue
            src = game_map.tiles[x][y]
            rx, ry = rotate(w, h, x, y)
            game_map.tiles[rx][ry] = Tile(
                terrain=src.terrain,
                elevation=src.elevation,
                deposit=src.deposit,
                richness=src.richness,
            )


def _scatter_deposits(game_map: GameMap):
    """Seed veins in CLUSTERS, not one tile at a time.

    Sprinkling single deposits uniformly makes every plot interchangeable — the same expected
    amount of everything everywhere — and then WHERE you settle cannot matter, whatever the
    placement rules say. Clustering is what creates land that is rich in one thing and poor in
    another, worth walking to or worth fighting over.
    """
    n_seeds = max(1, int(DEPOSIT_DENSITY * game_map.width * game_map.height / _CLUSTER_SIZE))
    for _ in range(n_seeds):
        x = int(np.random.randint(0, game_map.width))
        y = int(np.random.randint(0, game_map.height))
        candidates = [d for d, terr in DEPOSIT_TERRAIN.items()
                      if game_map.tiles[x][y].terrain in terr]
        if not candidates:
            continue
        # iron is the scarce one: it is what an army is made of
        weights = np.array([0.35 if d is DepositType.IRON else 1.0 for d in candidates])
        deposit = candidates[int(np.random.choice(len(candidates), p=weights / weights.sum()))]
        richness = float(np.clip(np.random.normal(0.7, 0.2), 0.2, 1.0))

        # Grow the seam to a BOUNDED size. Spreading by probability alone is a percolation
        # process: past its threshold it does not make a seam, it floods the continent (D27).
        budget = max(1, int(np.random.exponential(_CLUSTER_SIZE)))
        frontier = [(x, y)]
        seen = {(x, y)}
        while frontier and budget > 0:
            cx, cy = frontier.pop(0)
            tile = game_map.tiles[cx][cy]
            if tile.terrain not in DEPOSIT_TERRAIN[deposit] or tile.deposit is not None:
                continue
            tile.deposit = deposit
            tile.richness = richness * float(np.clip(np.random.normal(1.0, 0.15), 0.4, 1.0))
            budget -= 1
            for nx, ny in game_map.neighbors(cx, cy):
                if (nx, ny) not in seen and np.random.random() < _CLUSTER_SPREAD:
                    seen.add((nx, ny))
                    frontier.append((nx, ny))


#: What a plot MUST have for a district to be able to develop on it at all — a hard
#: requirement, not a preference: a settlement is founded once and lives with its ground for
#: the whole game, so a plot that cannot run an economy is a game lost before the first
#: decision (D27).
#:
#: Water is deliberately absent: a well is dug, and the water table is under every tile alike.
PLOT_REQUIREMENTS = {
    TileType.FOREST: 10,     # timber and game
    TileType.GRASS: 20,      # grain, gardens, cattle
    TileType.MOUNTAIN: 4,    # a quarry cuts into a mountainside
}
MIN_BUILDABLE_TILES = 90     # room for the ~20 buildings of a full district
MIN_CLAY_TILES = 3           # clay -> bricks, and bricks gate the bakery and the dormitory


def plot_is_viable(game_map: GameMap, x: int, y: int) -> bool:
    """Can a district founded here develop at all? (Not: is it a GOOD plot — is it a plot.)

    Buildability is judged on the TERRAIN, not on current occupancy: the warehouse the founding
    itself places must not tip a plot from viable to unviable, and neither should anything the
    district goes on to build. It is a fact about the LAND."""
    plot = [t for _, t in game_map.tiles_within((x, y), DISTRICT_RADIUS)]
    if sum(1 for t in plot if t.terrain not in UNBUILDABLE) < MIN_BUILDABLE_TILES:
        return False
    if sum(1 for t in plot if t.deposit is DepositType.CLAY) < MIN_CLAY_TILES:
        return False
    for terrain, needed in PLOT_REQUIREMENTS.items():
        if sum(1 for t in plot if t.terrain is terrain) < needed:
            return False
    return True


def _plot_score(game_map: GameMap, x: int, y: int) -> float:
    """How good is it to LIVE here: room to build, and the terrain the everyday economy hangs
    off (wood, stone, grass, water)."""
    plot = [t for _, t in game_map.tiles_within((x, y), DISTRICT_RADIUS)]
    buildable = sum(1 for t in plot if t.is_buildable)
    deposits = sum(1 for t in plot if t.deposit)
    terrains = {t.terrain for t in plot}
    # Terrain variety must DOMINATE: a plot's worth is what it can feed and build, not how many
    # empty tiles it counts. Weighted level with the rest, size swamped it (D27).
    variety = sum(
        25.0 for needed in (TileType.FOREST, TileType.MOUNTAIN, TileType.GRASS, TileType.WATER)
        if needed in terrains
    )
    return variety + 2.0 * deposits + 0.25 * buildable


def best_settlement_site(game_map: GameMap) -> Tuple[int, int]:
    """The best ground on the map, for a district with no rival to be fair to (D32)."""
    best, best_score = None, -1e9
    for x in range(DISTRICT_RADIUS, game_map.width - DISTRICT_RADIUS):
        for y in range(DISTRICT_RADIUS, game_map.height - DISTRICT_RADIUS):
            if not game_map.tiles[x][y].is_buildable:
                continue
            if not plot_is_viable(game_map, x, y):
                continue
            score = _plot_score(game_map, x, y)
            if score > best_score:
                best, best_score = (x, y), score
    if best is None:
        raise NoViableSite("no plot on this map can support a district")
    return best


def find_settlement_sites(game_map: GameMap, n_factions: int,
                          separation: float = None) -> List[Tuple[int, int]]:
    """Where everyone settles — chosen TOGETHER, so nobody gets the better ground.

    On a symmetric map the fair answer is forced: pick one good site, and give every other
    faction the point opposite it. The plots are then congruent, and the distance between
    them (which is the length of every march) falls out of the geometry rather than out of
    a config constant.

    Only two factions are handled — a symmetric 3-player start needs 3-fold symmetry in the
    map generator, and there is no third player yet.
    """
    if separation is None:
        separation = SETTLEMENT_SEPARATION
    if n_factions != 2:
        raise NotImplementedError(
            "symmetric starts are defined for 2 factions; a 3rd needs 3-fold map symmetry"
        )

    w, h = game_map.width, game_map.height
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    # Rotating a site through the centre puts the rival exactly 2r away, so a site must sit
    # half the intended separation from the middle of the map.
    target_radius = separation / 2.0

    best, best_score = None, -1e9
    for x in range(DISTRICT_RADIUS, w - DISTRICT_RADIUS):
        for y in range(DISTRICT_RADIUS, h // 2):        # search our half only
            if not game_map.tiles[x][y].is_buildable:
                continue
            drift = abs(GameMap.distance((x, y), center) - target_radius)
            if drift > 1.5:                              # keep the march the length it should be
                continue
            if not plot_is_viable(game_map, x, y):       # it must be able to DEVELOP here
                continue
            score = _plot_score(game_map, x, y)
            if score > best_score:
                best, best_score = (x, y), score

    if best is None:
        # A fact about the MAP, not about the game: the caller generates another one.
        raise NoViableSite("no viable, symmetric pair of plots at the required separation")
    return [best, rotate(w, h, *best)]
