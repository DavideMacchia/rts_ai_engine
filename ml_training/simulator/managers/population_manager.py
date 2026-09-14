"""
Population management for the RTS simulator.
Handles happiness, fertility and starvation.

Population lives in a district, eats that district's food, and is governed by that
district's policies. A district can starve while another one is well fed.

Fertility is CONDITIONAL (invariant I2 in docs/population_and_policies.md): it is a
product of the conditions a couple faces — food, housing, happiness — times a modest
policy modifier. Policies never overwrite the outcome; they move the conditions and
nudge the propensity. Calibration: with no policy active and food on the shelves every
factor equals exactly 1.0, so the birth chance is `BIRTH_CHANCE_PER_TURN`, which is
what this simulator did before policies existed.

`fertility` is a PER-WOMAN rate: each fertile woman draws against it, so the district's
total birth rate scales with how many fertile women it has. That is what lets population
climb toward its target rather than settle at a fixed equilibrium (D36).
"""

import numpy as np
from ..game_state import District, Faction
from ..config import (
    BASE_HAPPINESS,
    BIRTH_CHANCE_PER_TURN,
    HAPPINESS_FERTILITY_FLOOR,
    MIN_FOOD_PER_CAPITA_FOR_GROWTH,
    MIN_HAPPINESS_FOR_GROWTH,
    STARVATION_DEATH_RATE,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class PopulationManager:
    """Manages happiness, population growth and starvation."""

    def update_population(self, faction: Faction, delta_time: float):
        """
        Update population of every district of the faction.

        Args:
            faction: The faction to update
            delta_time: Time elapsed in seconds
        """
        for district in faction.districts:
            self.update_district_population(district, delta_time)

    def update_district_population(self, district: District, delta_time: float):
        """
        Age the cohorts, then update happiness, growth and starvation.

        Ageing is deterministic and runs first, so it cannot perturb the stochastic
        draws below.

        Args:
            district: The district to update
            delta_time: Time elapsed in seconds
        """
        district.age_sims(delta_time)

        if district.population <= 0:
            return

        district.calculate_happiness()

        total_food = district.get_total_food()

        # Conception. `fertility` is a PER-WOMAN rate (BIRTHS_PER_LIFE over one woman's adult
        # life), so every fertile woman draws against it independently; she then gestates for
        # PREGNANCY_DURATION before the birth (handled in age_sims). The district's total
        # birth rate is therefore proportional to how many fertile women it has — which is
        # what lets population grow toward TARGET_DISTRICT_POPULATION instead of being pinned
        # at a fixed equilibrium (D36). Dividing the rate by the number of women, as this once
        # did, cancelled that count and made births a district constant.
        #
        # Each woman's own rate is further lowered by her PARITY: the third child and beyond
        # come progressively rarer (D37), so growth is spread across many women rather than a
        # few carrying the whole district. Her post-birth cooldown is enforced by can_conceive.
        from ..config import (
            PREGNANCY_DURATION_SECONDS, HIGH_PARITY_FERTILITY_DECAY, HIGH_PARITY_THRESHOLD,
        )
        per_woman_rate = self.fertility(district)
        if per_woman_rate > 0.0:
            for woman in district.sims:
                if not woman.can_conceive:
                    continue
                parity_factor = HIGH_PARITY_FERTILITY_DECAY ** max(
                    0, woman.children_born - HIGH_PARITY_THRESHOLD + 1)
                rate = per_woman_rate * parity_factor
                if np.random.random() < rate * delta_time / 3600.0:
                    woman.pregnant_remaining = PREGNANCY_DURATION_SECONDS

        # Starvation
        if total_food <= 0:
            death_rate_per_second = STARVATION_DEATH_RATE / 3600.0
            deaths = district.population * death_rate_per_second * delta_time

            if deaths >= 1.0:
                district.remove_people(int(deaths))
            elif np.random.random() < deaths:
                district.remove_people(1)

    # ---- fertility -------------------------------------------------------

    def fertility(self, district: District) -> float:
        """Per-woman, per-hour birth chance: a product of conditions and one policy nudge.

        Every factor is 1.0 for an un-governed district with surplus food and free
        housing, so this reduces to `BIRTH_CHANCE_PER_TURN` at the nominal operating
        point. Below a threshold a factor goes to 0 and births stop, which is how the
        old hard gates are recovered.
        """
        return (
            BIRTH_CHANCE_PER_TURN
            * self._food_factor(district)
            * self._housing_factor(district)
            * self._happiness_factor(district)
            * district.policies.fertility_modifier()
        )

    def _food_factor(self, district: District) -> float:
        """Can we feed a child? Hard threshold, as before."""
        return 1.0 if district.get_food_per_capita() >= MIN_FOOD_PER_CAPITA_FOR_GROWTH else 0.0

    def _housing_factor(self, district: District) -> float:
        """Is there room for a child?"""
        return 1.0 if district.population < district.population_capacity else 0.0

    def _happiness_factor(self, district: District) -> float:
        """Continuous: fertility falls off linearly with unhappiness, down to a floor.

        Exactly 1.0 at BASE_HAPPINESS whatever the floor, so the un-governed district
        is unaffected. It bottoms out at HAPPINESS_FERTILITY_FLOOR rather than 0:
        a hard zero would make misery an absorbing state during training. Unhappiness
        should cost the agent population, not remove its way back.

        Only starvation and prohibition can still take fertility to exactly zero —
        the food factor and `block_families` respectively.
        """
        span = BASE_HAPPINESS - MIN_HAPPINESS_FOR_GROWTH
        if span <= 0:
            ramp = 1.0 if district.happiness >= MIN_HAPPINESS_FOR_GROWTH else 0.0
        else:
            ramp = _clamp01((district.happiness - MIN_HAPPINESS_FOR_GROWTH) / span)

        return HAPPINESS_FERTILITY_FLOOR + (1.0 - HAPPINESS_FERTILITY_FLOOR) * ramp
