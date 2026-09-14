"""
Modular Reward Component System
Separate reward calculation into trackable components
"""

from typing import Dict, Optional, List
from abc import ABC, abstractmethod
import numpy as np


class RewardComponent(ABC):
    """Base class for reward components"""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    @abstractmethod
    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        """
        Calculate reward for this component

        Args:
            state: Current game state/faction
            previous_state: Previous game state/faction
            action: Action taken
            delta_time: Time elapsed

        Returns:
            Reward value (will be multiplied by weight)
        """
        pass

    def get_weighted_reward(self, state, previous_state, action: str, delta_time: float) -> float:
        """Calculate and apply weight"""
        return self.calculate(state, previous_state, action, delta_time) * self.weight


class SurvivalReward(RewardComponent):
    """Reward for staying alive"""

    def __init__(self, reward_per_second: float = 1.0, weight: float = 1.0):
        super().__init__("survival", weight)
        self.reward_per_second = reward_per_second

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        # Simple: reward for each second alive
        return self.reward_per_second * delta_time


class EconomicReward(RewardComponent):
    """Reward for economic development"""

    def __init__(self, resource_value: float = 0.001, weight: float = 1.0):
        super().__init__("economic", weight)
        self.resource_value = resource_value

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        # Reward for total resources
        if hasattr(state, 'resources'):
            total_resources = sum(state.resources.values())
            return total_resources * self.resource_value
        return 0.0


class PopulationReward(RewardComponent):
    """Reward for population growth"""

    def __init__(self, per_pop: float = 10.0, growth_bonus: float = 50.0, weight: float = 1.0):
        super().__init__("population", weight)
        self.per_pop = per_pop
        self.growth_bonus = growth_bonus

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        reward = 0.0

        # Reward for current population
        if hasattr(state, 'population'):
            reward += state.population * self.per_pop

        # Bonus for population growth
        if previous_state and hasattr(previous_state, 'population'):
            growth = state.population - previous_state.population
            if growth > 0:
                reward += growth * self.growth_bonus

        return reward


class MilitaryReward(RewardComponent):
    """Reward for military strength"""

    def __init__(self, per_strength: float = 5.0, training_bonus: float = 20.0, weight: float = 1.0):
        super().__init__("military", weight)
        self.per_strength = per_strength
        self.training_bonus = training_bonus

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        reward = 0.0

        # Reward for military strength
        if hasattr(state, 'military_strength'):
            reward += state.military_strength * self.per_strength

        # Bonus for training units
        if action in ['TRAIN_SOLDIER', 'TRAIN_ARCHER']:
            reward += self.training_bonus

        return reward


class BuildingReward(RewardComponent):
    """Reward for constructing buildings"""

    def __init__(self, per_building: float = 15.0, variety_bonus: float = 10.0, weight: float = 1.0):
        super().__init__("building", weight)
        self.per_building = per_building
        self.variety_bonus = variety_bonus

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        reward = 0.0

        # Reward for total buildings
        if hasattr(state, 'buildings'):
            total_buildings = sum(state.buildings.values())
            reward += total_buildings * self.per_building

            # Bonus for building variety
            num_types = sum(1 for count in state.buildings.values() if count > 0)
            reward += num_types * self.variety_bonus

        # Bonus for building action
        if action.startswith('BUILD_'):
            reward += self.per_building * 2  # Double reward for building action

        return reward


class ActionValidityReward(RewardComponent):
    """Reward/penalty based on action validity"""

    def __init__(
        self,
        valid_action: float = 0.0,
        invalid_action: float = -5.0,
        do_nothing_penalty: float = -0.5,
        weight: float = 1.0
    ):
        super().__init__("action_validity", weight)
        self.valid_action = valid_action
        self.invalid_action = invalid_action
        self.do_nothing_penalty = do_nothing_penalty

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        # This would need action success/failure info
        # For now, penalize DO_NOTHING
        if action == 'DO_NOTHING':
            return self.do_nothing_penalty * delta_time

        return self.valid_action


class EfficiencyReward(RewardComponent):
    """Reward for efficient resource usage"""

    def __init__(self, weight: float = 1.0):
        super().__init__("efficiency", weight)

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        reward = 0.0

        # Reward for not wasting population capacity
        if hasattr(state, 'population') and hasattr(state, 'population_capacity'):
            if state.population_capacity > 0:
                utilization = state.population / state.population_capacity
                # Reward for 70-90% utilization (not too full, not too empty)
                if 0.7 <= utilization <= 0.9:
                    reward += 5.0
                elif utilization > 0.95:
                    reward -= 2.0  # Penalty for overcrowding

        return reward


class CombatReward(RewardComponent):
    """Reward for combat actions and victories"""

    def __init__(
        self,
        attack_reward: float = 10.0,
        damage_dealt_multiplier: float = 2.0,
        weight: float = 1.0
    ):
        super().__init__("combat", weight)
        self.attack_reward = attack_reward
        self.damage_dealt_multiplier = damage_dealt_multiplier

    def calculate(self, state, previous_state, action: str, delta_time: float) -> float:
        reward = 0.0

        # Reward for attacking
        if action == 'ATTACK':
            reward += self.attack_reward

        # TODO: Add reward for damage dealt (needs combat results)

        return reward


class ModularRewardCalculator:
    """
    Modular reward calculator that tracks individual components

    Usage:
        calculator = ModularRewardCalculator()
        calculator.add_component(SurvivalReward())
        calculator.add_component(EconomicReward())
        reward, breakdown = calculator.calculate_with_breakdown(state, prev_state, action)
    """

    def __init__(self):
        self.components: List[RewardComponent] = []

    def add_component(self, component: RewardComponent):
        """Add a reward component"""
        self.components.append(component)

    def remove_component(self, name: str):
        """Remove a reward component by name"""
        self.components = [c for c in self.components if c.name != name]

    def calculate(
        self,
        state,
        previous_state,
        action: str,
        delta_time: float = 1.0
    ) -> float:
        """Calculate total reward"""
        total = 0.0
        for component in self.components:
            total += component.get_weighted_reward(state, previous_state, action, delta_time)
        return total

    def calculate_with_breakdown(
        self,
        state,
        previous_state,
        action: str,
        delta_time: float = 1.0
    ) -> tuple[float, Dict[str, float]]:
        """
        Calculate total reward with per-component breakdown

        Returns:
            (total_reward, breakdown_dict)
        """
        breakdown = {}
        total = 0.0

        for component in self.components:
            component_reward = component.get_weighted_reward(state, previous_state, action, delta_time)
            breakdown[component.name] = component_reward
            total += component_reward

        return total, breakdown

    def print_configuration(self):
        """Print current reward configuration"""
        print("=" * 80)
        print("⚙️  REWARD CALCULATOR CONFIGURATION")
        print("=" * 80)

        print(f"\nTotal Components: {len(self.components)}")
        print(f"\n{'Component':<20} {'Weight':<10} {'Type'}")
        print("-" * 80)

        for component in self.components:
            print(f"{component.name:<20} {component.weight:<10.2f} {component.__class__.__name__}")

        print("=" * 80)


def create_default_reward_calculator() -> ModularRewardCalculator:
    """
    Create a reward calculator with balanced default components

    Returns:
        ModularRewardCalculator with default configuration
    """
    calculator = ModularRewardCalculator()

    # Add components with balanced weights
    calculator.add_component(SurvivalReward(reward_per_second=1.0, weight=1.0))
    calculator.add_component(EconomicReward(resource_value=0.01, weight=0.5))
    calculator.add_component(PopulationReward(per_pop=10.0, growth_bonus=50.0, weight=1.0))
    calculator.add_component(MilitaryReward(per_strength=5.0, training_bonus=20.0, weight=1.2))
    calculator.add_component(BuildingReward(per_building=15.0, variety_bonus=10.0, weight=1.0))
    calculator.add_component(ActionValidityReward(
        valid_action=0.0,
        invalid_action=-5.0,
        do_nothing_penalty=-1.0,
        weight=0.5
    ))
    calculator.add_component(EfficiencyReward(weight=0.3))
    calculator.add_component(CombatReward(attack_reward=10.0, weight=1.0))

    return calculator


def create_exploration_reward_calculator() -> ModularRewardCalculator:
    """
    Create a reward calculator that encourages exploration and action variety

    Returns:
        ModularRewardCalculator configured for exploration
    """
    calculator = ModularRewardCalculator()

    # Lower survival, higher action rewards
    calculator.add_component(SurvivalReward(reward_per_second=0.5, weight=0.5))
    calculator.add_component(EconomicReward(resource_value=0.01, weight=0.7))
    calculator.add_component(PopulationReward(per_pop=5.0, growth_bonus=100.0, weight=1.5))  # Bonus for growth
    calculator.add_component(MilitaryReward(per_strength=3.0, training_bonus=50.0, weight=1.5))  # Bonus for training
    calculator.add_component(BuildingReward(per_building=10.0, variety_bonus=30.0, weight=1.5))  # High variety bonus
    calculator.add_component(ActionValidityReward(
        valid_action=2.0,  # Reward valid actions
        invalid_action=-10.0,
        do_nothing_penalty=-3.0,  # Strong penalty
        weight=1.0
    ))
    calculator.add_component(CombatReward(attack_reward=25.0, weight=1.5))  # Encourage combat

    return calculator


def create_exploitation_reward_calculator() -> ModularRewardCalculator:
    """
    Create a reward calculator focused on optimizing learned strategies

    Returns:
        ModularRewardCalculator configured for exploitation
    """
    calculator = ModularRewardCalculator()

    # Focus on outcomes, not actions
    calculator.add_component(SurvivalReward(reward_per_second=2.0, weight=1.0))
    calculator.add_component(EconomicReward(resource_value=0.02, weight=1.0))
    calculator.add_component(PopulationReward(per_pop=20.0, growth_bonus=20.0, weight=1.0))  # Value size
    calculator.add_component(MilitaryReward(per_strength=10.0, training_bonus=5.0, weight=1.0))  # Value strength
    calculator.add_component(BuildingReward(per_building=20.0, variety_bonus=5.0, weight=1.0))  # Value quantity
    calculator.add_component(ActionValidityReward(
        valid_action=0.0,
        invalid_action=-5.0,
        do_nothing_penalty=-0.5,  # Small penalty
        weight=0.3
    ))
    calculator.add_component(EfficiencyReward(weight=1.0))
    calculator.add_component(CombatReward(attack_reward=5.0, weight=0.8))

    return calculator
