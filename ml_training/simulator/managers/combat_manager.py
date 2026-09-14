"""
Combat management for the RTS simulator.
Handles attacks and casualties.

This is the combat ORACLE: a swappable placeholder standing in for real combat.
Callers must depend only on "army X vs army Y -> outcome", never on the internals.

Armies are still resolved at faction scale (every district's garrison fights as one
force). Losses, however, land on the districts that own the units and the buildings.
"""

import numpy as np
from typing import Dict, Optional
from ..game_state import Faction
from .. import config as cfg
from ..config import (
    DEFENDER_BONUS,
    COMBAT_CASUALTIES_RATE,
    WINNER_CASUALTIES_RATE,
    BUILDING_DAMAGE_CHANCE,
    UNIT_STRENGTH,
)


def army_strength(units: Dict[str, int]) -> float:
    """Strength of a column of units — an army that is not in a district right now."""
    return sum(UNIT_STRENGTH.get(u, 1.0) * n for u, n in units.items())


def apply_casualties_to_army(units: Dict[str, int], rate: float):
    """Casualties on a marching column. Same rule as a garrison's, applied in place."""
    for unit_type in list(units.keys()):
        count = units[unit_type]
        if count > 0:
            units[unit_type] = max(0, count - max(1, int(count * rate)))


class CombatManager:
    """Manages combat between factions."""

    def execute_attack(self, attacker: Faction, defender: Faction,
                       attacking_units: Optional[dict] = None) -> bool:
        """
        Execute combat between two factions.

        Args:
            attacker: The attacking faction
            defender: The defending faction
            attacking_units: the army that actually SHOWED UP — the column that marched
                (D26). Casualties land on it, in place. When omitted, the attacker's whole
                standing army fights, which is only meaningful for a battle at home.

        Returns:
            bool: True if attacker wins, False if defender wins
        """
        marched = attacking_units is not None
        attacker_strength = (
            army_strength(attacking_units) if marched else attacker.military_strength
        )
        if attacker_strength <= 0:
            return False  # Cannot attack without army

        # The defender fights with what is HOME. An army that marched out is not here to
        # hold the walls — that is the whole risk of leaving (D26).
        defender_strength = defender.military_strength * DEFENDER_BONUS

        if attacker_strength > defender_strength:
            # Attacker wins
            if marched:
                apply_casualties_to_army(attacking_units, WINNER_CASUALTIES_RATE)
            else:
                self.apply_casualties(attacker, WINNER_CASUALTIES_RATE)
            self.apply_casualties(defender, COMBAT_CASUALTIES_RATE)

            if attacker_strength >= cfg.WAREHOUSE_DESTRUCTION_THRESHOLD * defender_strength:
                # OVERWHELMING: the district is taken, not merely raided. Its warehouse falls,
                # and with the last warehouse the faction falls.
                #
                # This threshold is what makes conquest something you can INTEND (D24): if a
                # won battle only ever deleted a random building, the SIZE of the winning army
                # would not matter, only the NUMBER of wins — and raiding with a token army
                # would strictly dominate committing a real one.
                self._raze_warehouse(defender)
            elif np.random.random() < BUILDING_DAMAGE_CHANCE:
                # A win that is not overwhelming still hurts: it costs the defender a
                # building, but not the district.
                defender_buildings = [
                    (district, building_type)
                    for district in defender.districts
                    for building_type, count in district.buildings.items()
                    if count > 0 and building_type != 'warehouse'
                ]
                if defender_buildings:
                    district, building_type = defender_buildings[
                        np.random.randint(len(defender_buildings))
                    ]
                    district.buildings[building_type] -= 1

            return True
        else:
            # Defender wins
            self.apply_casualties(defender, WINNER_CASUALTIES_RATE)
            if marched:
                apply_casualties_to_army(attacking_units, COMBAT_CASUALTIES_RATE)
            else:
                self.apply_casualties(attacker, COMBAT_CASUALTIES_RATE)
            return False

    def _raze_warehouse(self, defender: Faction):
        """Take one district: the weakest defended one loses its warehouse.

        With a single district (Stage 1) this is simply "the faction falls". The
        district-by-district form is what Stage 2 needs, so it is written that way now.
        """
        standing = [d for d in defender.districts if d.has_warehouse]
        if not standing:
            return
        district = min(standing, key=lambda d: d.military_strength)
        district.buildings['warehouse'] -= 1

    def apply_casualties(self, faction: Faction, rate: float):
        """
        Apply casualties to every district's garrison.

        Args:
            faction: The faction to apply casualties to
            rate: Casualty rate (0.0 to 1.0)
        """
        for district in faction.districts:
            for unit_type in list(district.units.keys()):
                count = district.units[unit_type]
                if count > 0:
                    # Ensure at least 1 casualty if unit exists and rate > 0
                    casualties = max(1, int(count * rate))
                    district.units[unit_type] = max(0, count - casualties)
                else:
                    district.units[unit_type] = 0
