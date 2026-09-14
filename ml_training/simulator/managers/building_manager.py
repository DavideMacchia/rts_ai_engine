"""
Building management for the RTS simulator.
Handles building construction and progression.

Buildings belong to a district and are paid for out of that district's own stock.
"""

from typing import Optional

from ..game_state import District, Faction, BuildingInProgress, BuildingType
from ..config import BUILDING_COSTS, BUILDING_BUILD_TIME


class BuildingManager:
    """Manages building construction."""

    def progress_buildings(self, faction: Faction, delta_time: float):
        """
        Progress building construction across every district.

        Args:
            faction: The faction to progress buildings for
            delta_time: Time elapsed in seconds
        """
        for district in faction.districts:
            self.progress_district_buildings(district, delta_time)

    def progress_district_buildings(self, district: District, delta_time: float):
        """
        Progress building construction.

        Args:
            district: The district to progress buildings for
            delta_time: Time elapsed in seconds
        """
        completed = []

        for i, building in enumerate(district.buildings_in_progress):
            building.turns_remaining -= delta_time

            if building.turns_remaining <= 0:
                building_type = building.building_type.value
                district.add_building(building_type, tile=building.tile,
                                      quality=building.quality)
                completed.append(i)

        for i in reversed(completed):
            district.buildings_in_progress.pop(i)

    def start_building(
        self,
        faction: Faction,
        building_type: str,
        district: Optional[District] = None,
        tile: Optional[tuple] = None,
    ) -> tuple[bool, str]:
        """
        Start building construction in a district.

        Args:
            faction: The faction to build for
            building_type: Type of building to construct
            district: Where to build. Defaults to the first district that can pay.

        Returns:
            tuple[bool, str]: (success, message)
        """
        if building_type not in BUILDING_COSTS:
            return False, "invalid_missing_prerequisite"

        costs = BUILDING_COSTS[building_type]

        if district is None:
            district = faction.district_that_can_afford(costs)
        elif not district.can_afford(costs):
            district = None

        if district is None:
            return False, "invalid_cant_afford"

        # Where does it GO? On a map, a building needs ground: a plot tile that is free, and
        # that can meet what the building requires. A mine with no iron in reach, or a well
        # with no water, is not an expensive choice — it is an impossible one, and it fails
        # BEFORE the costs are paid.
        if tile is not None:
            # The CALLER chose the ground (D28). It gets no help and no correction: naming a
            # tile that cannot carry the building is a wrong decision, not a typo to be
            # quietly fixed, and the agent has to feel it.
            quality = self.site_quality(district, building_type, tile)
            if quality is None:
                return False, "invalid_missing_prerequisite"
        else:
            tile, quality = self.find_site(district, building_type)
            if district.center is not None and district._map is not None and tile is None:
                return False, "invalid_missing_prerequisite"

        if not district.deduct_costs(costs):
            return False, "invalid_cant_afford"

        if tile is not None:
            # Reserve the ground now. The site is occupied for as long as the work lasts.
            district._map.place(district.faction_id, building_type, *tile)

        build_time_minutes = BUILDING_BUILD_TIME.get(building_type, 2)
        build_time_seconds = build_time_minutes * 60.0

        building_obj = BuildingInProgress(
            building_type=BuildingType(building_type),
            turns_remaining=build_time_seconds,
            tile=tile,
            quality=quality,
        )
        district.buildings_in_progress.append(building_obj)

        return True, "valid_action"

    @staticmethod
    def site_quality(district: District, building_type: str, tile: tuple):
        """What a tile the caller NAMED (the building's top-left corner) is worth, or None if
        the building cannot stand there: its footprint would run off the map or the plot, cover
        occupied ground, or miss a deposit it requires."""
        if district.center is None or district._map is None:
            return 1.0      # a game with no world cannot argue about where things go

        from ..config import DISTRICT_RADIUS
        if not district._map.footprint_fits(building_type, tile[0], tile[1],
                                            district.center, DISTRICT_RADIUS):
            return None
        return district._map.placement_quality(building_type, *tile)

    @staticmethod
    def find_site(district: District, building_type: str):
        """The ground this building would stand on, and what that ground is worth.

        Until the agent gets `BUILD(type, tile)`, the world places the building in the best
        site the district has. So the map is currently a CONSTRAINT (it can forbid a mine)
        and not yet an ADVANTAGE (it cannot yet be exploited badly) — which is deliberate:
        it keeps the economy neutral, so that when the pointer action arrives its value can
        be measured against this baseline instead of being confounded with it.
        """
        if district.center is None or district._map is None:
            return None, 1.0
        tile = district._map.best_tile_for(building_type, district.center)
        if tile is None:
            return None, 1.0
        return tile, district._map.placement_quality(building_type, *tile) or 1.0
