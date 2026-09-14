"""
Training management for the RTS simulator.
Handles unit training and progression.

Units are trained in a district that has a barracks, cost that district's stock,
and are drawn from that district's population.
"""

from typing import Optional

from ..game_state import District, Faction, UnitInTraining, UnitType
from ..config import CONSCRIPTION_RATE_PER_HOUR, UNIT_TRAINING_COST, UNIT_TRAINING_TIME


class TrainingManager:
    """Manages unit training and conscription."""

    def progress_training(self, faction: Faction, delta_time: float):
        """
        Progress unit training across every district.

        Args:
            faction: The faction to progress training for
            delta_time: Time elapsed in seconds
        """
        for district in faction.districts:
            self.progress_district_training(district, delta_time)

    def progress_district_training(self, district: District, delta_time: float):
        """
        Progress unit training.

        Args:
            district: The district to progress training for
            delta_time: Time elapsed in seconds
        """
        completed = []

        for i, unit in enumerate(district.units_in_training):
            unit.turns_remaining -= delta_time

            if unit.turns_remaining <= 0:
                unit_type = unit.unit_type.value

                # Consume an adult (a worker becomes a soldier). Kids cannot enlist.
                if district.remove_adult():
                    district.add_unit(unit_type)
                completed.append(i)

        for i in reversed(completed):
            district.units_in_training.pop(i)

    def apply_conscription(self, faction: Faction, delta_time: float):
        """Draft adults into soldiers wherever `forced_conscription` is in force."""
        for district in faction.districts:
            self.conscript_district(district, delta_time)

    def conscript_district(self, district: District, delta_time: float):
        """Lever L4 (coercion): turn workers into soldiers, against their preference.

        Coercion does not multiply a utility — it replaces the choice. So it bypasses
        the barracks, the cost and the training time that a volunteer would go
        through, and it is priced accordingly: -25 happiness, the steepest penalty in
        POLICY_EFFECTS, which alone drives fertility to zero.

        Deterministic (an accumulator, not a coin flip) so that switching the policy
        on cannot perturb any other stochastic draw in the simulation.
        """
        if not district.policies.forced_conscription:
            return

        district.conscription_progress += CONSCRIPTION_RATE_PER_HOUR * delta_time / 3600.0

        # Only men are drafted (`remove_adult`). The draft must be gated on the men, not
        # on the adults: gated on adults, it kept "drafting" a district whose men were
        # already gone — `remove_adult()` failed, but the soldier was added anyway, so
        # coercion conjured an army out of the women it could not actually take (measured:
        # 320 soldiers from 7 adults). A soldier is always a man who stopped being a worker.
        while district.conscription_progress >= 1.0 and district.adult_men > 0:
            if not district.remove_adult():
                break
            district.add_unit('soldier')
            district.conscription_progress -= 1.0

        # No men left to draft; don't bank credit against a future generation
        if district.adult_men <= 0:
            district.conscription_progress = 0.0

    def start_training(
        self,
        faction: Faction,
        unit_type: str,
        district: Optional[District] = None,
    ) -> tuple[bool, str]:
        """
        Start unit training in a district with a barracks.

        Args:
            faction: The faction to train for
            unit_type: Type of unit to train
            district: Where to train. Defaults to the first district with a
                barracks that can pay.

        Returns:
            tuple[bool, str]: (success, message)
        """
        # Check if barracks exists
        candidates = [
            d for d in ([district] if district is not None else faction.districts)
            if d.get_building_count('barracks') > 0
        ]
        if not candidates:
            return False, "invalid_missing_prerequisite"

        if unit_type not in UNIT_TRAINING_COST:
            return False, "invalid_missing_prerequisite"

        costs = UNIT_TRAINING_COST[unit_type]
        district = next((d for d in candidates if d.can_afford(costs)), None)
        if district is None:
            return False, "invalid_cant_afford"

        if not district.deduct_costs(costs):
            return False, "invalid_cant_afford"

        # UNIT_TRAINING_TIME is in minutes, convert to seconds
        train_time_minutes = UNIT_TRAINING_TIME.get(unit_type, 0.5)
        train_time_seconds = train_time_minutes * 60.0

        unit_obj = UnitInTraining(
            unit_type=UnitType(unit_type),
            turns_remaining=train_time_seconds
        )
        district.units_in_training.append(unit_obj)

        return True, "valid_action"
