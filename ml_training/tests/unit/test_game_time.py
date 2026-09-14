"""
Test Game Time Mechanics
Validates that game time limits work correctly and games end appropriately
"""

import sys
import os
import numpy as np
import pytest

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
ml_training_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, ml_training_dir)

from simulator.realtime_simulator import RealTimeRTSSimulator
from simulator.actions import Action, ActionType


class TestGameTime:
    """Test suite for game time mechanics"""

    def test_game_time_initialization(self):
        """Test that game time starts at 0"""
        sim = RealTimeRTSSimulator(num_factions=2)
        assert sim.game_time == 0.0, "Game time should start at 0"
        assert sim.max_game_time is None, "Default max_game_time should be None"

    def test_game_time_with_max_limit(self):
        """Test that max_game_time is set correctly"""
        max_time = 3600.0
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)
        assert sim.max_game_time == max_time, f"max_game_time should be {max_time}"

    def test_game_time_progression(self):
        """Test that game time advances with each step"""
        sim = RealTimeRTSSimulator(num_factions=2)
        initial_time = sim.game_time

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        delta_time = 60.0
        sim.step(actions, delta_time=delta_time)

        assert sim.game_time == initial_time + delta_time, \
            f"Game time should advance by {delta_time} seconds"

    def test_the_episode_truncates_at_max_time_but_the_game_does_not_end(self):
        """`max_game_time` bounds the EPISODE, not the game.

        The simulator keeps a coherent state past the horizon; it is the environment
        that stops stepping and reports `truncated`, so the value function bootstraps
        rather than learning that the world ends.
        """
        max_time = 180.0
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        steps = 0
        while not sim.is_out_of_time() and steps < 10:
            _, _, terminated = sim.step(actions, delta_time=60.0)
            assert not terminated, "nobody lost a warehouse"
            steps += 1

        assert sim.is_out_of_time()
        assert sim.game_time >= max_time
        assert not sim.state.game_over, "the clock is a truncation, not a termination"
        assert sim.state.winner is None

    def test_game_no_time_limit(self):
        """Test that game doesn't end from time when max_game_time is None"""
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=None)

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        # Step many times
        for _ in range(100):
            _, _, done = sim.step(actions, delta_time=60.0)
            if done:
                break

        # Game should not end from time limit (only from warehouse destruction)
        # Since both factions are doing nothing, game should not be over
        assert not sim.state.game_over, \
            "Game should not end from time when max_game_time is None"

    def test_running_out_of_time_produces_no_winner(self):
        """The clock is a truncation, not a result. Nobody wins by outlasting it.

        There used to be a development score that broke the tie at the time limit,
        weighting `military_strength x 100`. It taught the agent to stockpile an army
        and never attack: 60/60 wins, 0 conquests. See design_decisions.md D12/D13.
        """
        np.random.seed(0)

        max_time = 180.0
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)
        # Make faction 0 overwhelmingly "better developed" by the OLD score.
        for _ in range(5):
            sim.state.factions[0].add_building('farm')
        sim.state.factions[0].add_unit('soldier', 50)
        sim.state.factions[0].calculate_military_strength()

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }
        while not sim.is_out_of_time():
            _, _, terminated = sim.step(actions, delta_time=60.0)
            assert not terminated, "nobody lost a warehouse, so nobody can have won"

        assert sim.is_out_of_time()
        assert not sim.state.game_over, "the time limit does not end the game"
        assert sim.state.winner is None, "an army is not a victory"

    def test_the_only_victory_is_razing_every_warehouse(self):
        np.random.seed(0)

        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=180.0)
        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        _, _, terminated = sim.step(actions, delta_time=1.0)
        assert not terminated and sim.state.winner is None

        sim.state.factions[1].capital.remove_building('warehouse')
        _, _, terminated = sim.step(actions, delta_time=1.0)

        assert terminated
        assert sim.state.game_over
        assert sim.state.winner == 0

    def test_building_construction_time(self):
        """Test that buildings take realistic time to complete"""
        # Farm takes 2 minutes = 120 seconds
        sim = RealTimeRTSSimulator(num_factions=2)
        faction = sim.state.factions[0]

        # Give faction resources to build a farm
        faction.resources['wood'] = 500
        faction.resources['stone'] = 500

        # Start building a farm
        action = Action(faction_id=0, action_type=ActionType.BUILD_FARM)
        actions = {
            0: action,
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        sim.step(actions, delta_time=60.0)

        # Farm should be in progress
        assert len(faction.buildings_in_progress) > 0, \
            "Farm should be under construction"
        assert faction.get_building_count('farm') == 0, \
            "Farm should not be completed immediately"

        # Check construction time remaining
        building_in_progress = faction.buildings_in_progress[0]
        # Farm takes 2 minutes = 120 seconds, we advanced 60 seconds
        expected_remaining = 120 - 60
        assert abs(building_in_progress.turns_remaining - expected_remaining) < 1.0, \
            f"Construction should have ~{expected_remaining} seconds remaining"

    def test_building_completes_after_construction_time(self):
        """Test that building completes after sufficient time has passed"""
        sim = RealTimeRTSSimulator(num_factions=2)
        faction = sim.state.factions[0]

        # Give faction resources
        faction.resources['wood'] = 500
        faction.resources['stone'] = 500

        # Start building a farm
        action = Action(faction_id=0, action_type=ActionType.BUILD_FARM)
        actions = {
            0: action,
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }
        sim.step(actions, delta_time=60.0)

        # Advance time by 2 minutes (farm construction time)
        do_nothing_actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        # Farm takes 2 minutes = 120 seconds
        # We already advanced 60 seconds, so advance 120 more to ensure completion
        sim.step(do_nothing_actions, delta_time=120.0)

        # Farm should be completed
        assert len(faction.buildings_in_progress) == 0, \
            "No buildings should be in progress"
        assert faction.get_building_count('farm') > 0, \
            "Farm should be completed"

    def test_max_steps_with_realistic_game_time(self):
        """Test that 1000 steps with 60s delta_time reaches realistic game duration"""
        sim = RealTimeRTSSimulator(num_factions=2)

        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }

        # Simulate 1000 steps at 60 seconds each
        max_steps = 1000
        for _ in range(max_steps):
            sim.step(actions, delta_time=60.0)

        # 1000 steps * 60 seconds = 60000 seconds = 16.67 hours
        expected_time = 60000.0
        assert abs(sim.game_time - expected_time) < 1.0, \
            f"After 1000 steps, game time should be ~{expected_time} seconds"

    def test_realistic_benchmark_game_length(self):
        """Test that 30-minute game time allows for meaningful gameplay"""
        # 30 minutes should allow:
        # - Multiple buildings to complete (1-5 minute build times)
        # - Resource production
        # - Military training
        # - Attacks

        max_time = 1800.0  # 30 minutes
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)

        # With 60s steps, 30 minutes = 30 steps
        expected_steps = max_time / 60.0

        assert expected_steps == 30, \
            "30 minutes should allow 30 steps at 60s intervals"

        # Buildings that can complete in 30 minutes (1800 seconds):
        # - Quick builds (1m = 60s): 30 can complete
        # - Standard builds (2m = 120s): 15 can complete
        # - Complex builds (3-4m = 180-240s): 7-10 can complete
        # - Advanced builds (5m = 300s): 6 can complete

        # This is realistic for a fast-paced RTS game benchmark


class TestGameTimeIntegration:
    """Integration tests for game time with full gameplay"""

    def test_full_game_with_time_limit(self):
        """Test a full game with buildings, units, and time limit"""
        max_time = 600.0  # 10 minutes
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)
        faction = sim.state.factions[0]

        # Give resources
        faction.resources['wood'] = 1000
        faction.resources['stone'] = 1000

        # Build a farm (2 minutes)
        actions = {
            0: Action(faction_id=0, action_type=ActionType.BUILD_FARM),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }
        sim.step(actions, delta_time=60.0)

        # Advance 3 minutes to complete farm
        do_nothing = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }
        sim.step(do_nothing, delta_time=180.0)  # 3 minutes

        # Farm should be done
        assert faction.get_building_count('farm') > 0, "Farm should be built"

        # Continue until the time limit. Neither faction attacks, so neither can win:
        # the episode is truncated, and the game itself is still undecided.
        while not sim.is_out_of_time():
            _, _, terminated = sim.step(do_nothing, delta_time=60.0)
            assert not terminated

        assert not sim.state.game_over, "the clock does not end the game"
        assert sim.state.winner is None, "a stalemate has no winner"

    def test_reset_preserves_max_game_time(self):
        """Test that reset preserves max_game_time setting"""
        max_time = 3600.0
        sim = RealTimeRTSSimulator(num_factions=2, max_game_time=max_time)

        # Advance some time
        actions = {
            0: Action(faction_id=0, action_type=ActionType.DO_NOTHING),
            1: Action(faction_id=1, action_type=ActionType.DO_NOTHING)
        }
        sim.step(actions, delta_time=60.0)

        # Reset
        sim.reset()

        assert sim.max_game_time == max_time, \
            "max_game_time should be preserved after reset"
        assert sim.game_time == 0.0, \
            "game_time should reset to 0"


def test_benchmark_configuration():
    """Test that benchmark configuration is appropriate"""
    # Recommended test configuration:
    # - max_game_time: 1800 seconds (30 minutes)
    # - delta_time: 60 seconds (1 minute decisions)
    # - max_steps: 1000 (prevents infinite loops)

    max_game_time = 1800.0  # 30 minutes
    delta_time = 60.0
    max_steps = 1000

    # Calculate how many steps until time limit
    steps_until_limit = max_game_time / delta_time

    assert steps_until_limit == 30, \
        "30-minute game should end at 30 steps"

    assert steps_until_limit < max_steps, \
        "Time limit should trigger before max_steps"

    # Verify buildings can complete
    farm_build_time = 120.0  # 2 minutes
    barracks_build_time = 240.0  # 4 minutes

    assert max_game_time > farm_build_time, \
        "Game should be long enough to build farms"
    assert max_game_time > barracks_build_time, \
        "Game should be long enough to build barracks"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
