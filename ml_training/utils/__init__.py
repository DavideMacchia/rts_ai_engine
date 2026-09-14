"""
ML Training Utilities
Debugging, monitoring, and analysis tools for ML training
"""

from .reward_debugger import RewardDebugger, RewardComponentTracker
from .action_monitor import ActionMonitor, ActionDistributionAnalyzer
from .episode_stats import EpisodeStatsTracker, EpisodeAnalyzer
from .curriculum import DifficultyScheduler, CurriculumManager, AdaptiveOpponentAI, DifficultyLevel
from .reward_components import (
    ModularRewardCalculator,
    RewardComponent,
    SurvivalReward,
    EconomicReward,
    PopulationReward,
    MilitaryReward,
    BuildingReward,
    ActionValidityReward,
    EfficiencyReward,
    CombatReward,
    create_default_reward_calculator,
    create_exploration_reward_calculator,
    create_exploitation_reward_calculator
)
from .testing import RewardTester, EnvironmentTester, test_training_stability
from .checkpoint_comparison import CheckpointComparator, quick_compare, find_best_checkpoint

__all__ = [
    # Debugging
    'RewardDebugger',
    'RewardComponentTracker',

    # Monitoring
    'ActionMonitor',
    'ActionDistributionAnalyzer',
    'EpisodeStatsTracker',
    'EpisodeAnalyzer',

    # Curriculum Learning
    'DifficultyScheduler',
    'CurriculumManager',
    'AdaptiveOpponentAI',
    'DifficultyLevel',

    # Reward Components
    'ModularRewardCalculator',
    'RewardComponent',
    'SurvivalReward',
    'EconomicReward',
    'PopulationReward',
    'MilitaryReward',
    'BuildingReward',
    'ActionValidityReward',
    'EfficiencyReward',
    'CombatReward',
    'create_default_reward_calculator',
    'create_exploration_reward_calculator',
    'create_exploitation_reward_calculator',

    # Testing
    'RewardTester',
    'EnvironmentTester',
    'test_training_stability',

    # Checkpoint Comparison
    'CheckpointComparator',
    'quick_compare',
    'find_best_checkpoint',
]
