"""
Manager modules for RTS simulator.
Each manager handles a specific aspect of the simulation.
"""

from .population_manager import PopulationManager
from .resource_manager import ResourceManager
from .combat_manager import CombatManager
from .building_manager import BuildingManager
from .training_manager import TrainingManager
from .reward_calculator import RewardCalculator

__all__ = [
    'PopulationManager',
    'ResourceManager',
    'CombatManager',
    'BuildingManager',
    'TrainingManager',
    'RewardCalculator',
]
