"""
District policies: the four levers a district uses to steer self-managing sims.

See docs/population_and_policies.md. A policy is a pair **(effect, cost)** — a policy
that only subtracts happiness is a bug, and that is what all of them were before this
module existed.

The levers, ordered by how much autonomy they take from the population:

    L1 AFFORDANCE  changes what is POSSIBLE      minimum_food (rationing)
    L2 INCENTIVE   changes what is ATTRACTIVE    population_rate
    L3 PROHIBITION masks an option               block_families
    L4 COERCION    overrides the choice          forced_conscription

Cost in happiness rises with coerciveness (-15 / -10 / -25 in POLICY_EFFECTS). That
ordering is the balancing invariant: keep it.

**Invariant I1** — prohibition never masks a vital need; coercion may force one.
`minimum_food` is therefore rationing (L1: less food is available per head), never a
ban on eating. Forbidding a vital need deadlocks the population; forcing one is merely
unpopular.

**Not modelled yet.** `siege_mode`, `cant_leave`, `force_eat`, `force_sleep` and
`elders_retirement` have no honest counterpart while the simulator has no per-sim
needs layer and no age cohorts. They are deliberately absent rather than faked; see
docs/population_and_policies.md §9 steps 1-2 and 6.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from .config import POLICY_EFFECTS


class PopulationRate(Enum):
    """L2 incentive: the district's stance on births."""
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    INCREASE = "increase"


# L2 is an *incentive*, not an override: a modest nudge on top of the conditions
# (food, housing, happiness) that actually decide whether a couple has a child.
# Invariant I2 - policies act on conditions, they never overwrite outcomes.
FERTILITY_POLICY_MODIFIER: Dict[PopulationRate, float] = {
    PopulationRate.DECREASE: 0.5,
    PopulationRate.MAINTAIN: 1.0,
    PopulationRate.INCREASE: 1.5,
}


@dataclass
class DistrictPolicies:
    """The policies active in one district. Defaults reproduce the un-governed district."""

    # --- L1 AFFORDANCE: changes what is possible ---
    minimum_food: bool = False

    # --- L2 INCENTIVE: changes what is attractive ---
    population_rate: PopulationRate = PopulationRate.MAINTAIN

    # --- L3 PROHIBITION: masks an option (never a vital need, see I1) ---
    block_families: bool = False

    # --- L4 COERCION: overrides the sim's choice ---
    forced_conscription: bool = False

    def active(self) -> List[str]:
        """Names of the policies currently in force (for logging and observation)."""
        names = []
        if self.minimum_food:
            names.append('minimum_food')
        if self.block_families:
            names.append('block_families')
        if self.forced_conscription:
            names.append('forced_conscription')
        return names

    def happiness_cost(self) -> float:
        """Total happiness penalty (a non-positive number). The COST half of a policy."""
        return sum(POLICY_EFFECTS[name] for name in self.active())

    def fertility_modifier(self) -> float:
        """The EFFECT half, for births.

        L3 masks the option entirely (no courtship, no children). L2 only nudges: the
        conditions still decide, which is why this is a factor and not an override.
        """
        if self.block_families:
            return 0.0
        return FERTILITY_POLICY_MODIFIER[self.population_rate]

    def ration_factor(self) -> float:
        """The EFFECT half, for food.

        L1 constrains the *resource*, not the action: sims still choose to eat, they
        just find less. Strictly between 0 and 1 — a factor of 0 would be a ban on
        eating, which I1 forbids.
        """
        from .config import MINIMUM_FOOD_RATION_FACTOR
        return MINIMUM_FOOD_RATION_FACTOR if self.minimum_food else 1.0


# --- the district agent's policy action space ---------------------------------
# Each entry is (apply, is_already_in_force). The second half is what lets the action
# mask hide a policy action that would be a no-op, so the agent cannot burn a decision
# re-asserting a policy it already holds.

def _set_rate(rate: PopulationRate):
    return lambda p: setattr(p, 'population_rate', rate)


def _set_flag(name: str, value: bool):
    return lambda p: setattr(p, name, value)


def _rate_is(rate: PopulationRate):
    return lambda p: p.population_rate is rate


def _flag_is(name: str, value: bool):
    return lambda p: getattr(p, name) is value


def _build_policy_tables():
    from .actions import ActionType

    spec = {
        ActionType.SET_POPULATION_RATE_INCREASE:
            (_set_rate(PopulationRate.INCREASE), _rate_is(PopulationRate.INCREASE)),
        ActionType.SET_POPULATION_RATE_MAINTAIN:
            (_set_rate(PopulationRate.MAINTAIN), _rate_is(PopulationRate.MAINTAIN)),
        ActionType.SET_POPULATION_RATE_DECREASE:
            (_set_rate(PopulationRate.DECREASE), _rate_is(PopulationRate.DECREASE)),
        ActionType.ENABLE_MINIMUM_FOOD:
            (_set_flag('minimum_food', True), _flag_is('minimum_food', True)),
        ActionType.DISABLE_MINIMUM_FOOD:
            (_set_flag('minimum_food', False), _flag_is('minimum_food', False)),
        ActionType.ENABLE_FORCED_CONSCRIPTION:
            (_set_flag('forced_conscription', True), _flag_is('forced_conscription', True)),
        ActionType.DISABLE_FORCED_CONSCRIPTION:
            (_set_flag('forced_conscription', False), _flag_is('forced_conscription', False)),
    }
    return {a: apply for a, (apply, _) in spec.items()}, {a: noop for a, (_, noop) in spec.items()}


POLICY_ACTIONS, POLICY_ACTION_IS_NOOP = _build_policy_tables()


def policy_action_is_available(action_type, policies: DistrictPolicies) -> bool:
    """False when the action would change nothing (already in force)."""
    return not POLICY_ACTION_IS_NOOP[action_type](policies)
