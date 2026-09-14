"""
TestingBot - a balanced, economy-first bot used only as a test/regression fixture.

It grows steadily, never rushes, and (by a known, pre-existing bug) never builds a
lumberyard, so it stalls and does not put up a real fight — which is why it is no longer
in the training curriculum. The 13 baseline test failures document that bug on purpose;
the tests are right, the bot is not fixed. For actual sparring use the combat bots
(easy / medium / hard). See design_decisions.md D16 and the opponent registry.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import ScriptedOpponent
from ..actions import Action
from ..game_state import GameState

# Import behavior tree components
from ..behavior_tree.nodes import Selector, Sequence, Condition, Action as BTAction
from ..behavior_tree.rts_helpers import (
    # Conditions
    is_under_attack,
    can_attack_opponent,
    needs_housing,
    has_critically_low_stone,
    has_critically_low_wood,
    has_low_wood,
    has_low_stone,
    has_low_food,
    has_few_lumberyards,
    has_few_quarries,
    has_few_farms,
    has_no_barracks,
    has_barracks,
    needs_more_military,
    has_good_economy,
    has_worker_surplus,
    has_worker_deficit,
    # Actions
    try_build_house,
    try_build_hunting_shed,
    try_build_lumberyard,
    try_build_quarry,
    try_build_farm,
    try_build_well,
    try_build_barracks,
    try_train_soldier,
    try_train_archer,
    try_attack_opponent,
    try_build_mill,
    try_build_bakery,
    do_nothing,
)


def _needs_hunting_shed(context: dict) -> bool:
    """No meat source yet. Grain is not edible, so a district without a hunting shed
    (or a completed bread chain) starves once its starting stock runs out."""
    faction = context['faction']
    return faction.get_building_count('hunting_shed') < 1


class TestingBot(ScriptedOpponent):
    """
    Balanced economy-first bot (test fixture only).

    Behavior Tree Structure:
    Root: Selector
        ├─ Housing Branch (HIGHEST PRIORITY - population growth)
        │   └─ Build houses to maintain capacity (50 pop target)
        ├─ Worker Deficit Branch (prevent worker shortage)
        │   └─ Build houses if we need more workers
        ├─ Critical Resources Branch (emergency resource recovery)
        │   ├─ Build quarry if stone critically low
        │   └─ Build lumberyard if wood critically low
        ├─ Defense Branch (if under attack)
        │   └─ Build military quickly
        ├─ Attack Branch (if strong enough)
        │   └─ Attack opponent
        ├─ Economy Branch
        │   ├─ Resource production (prioritize stone > food > wood)
        │   └─ Advanced economy (mills, bakeries)
        ├─ Military Development
        │   ├─ Build barracks
        │   └─ Train units
        └─ Fallback: Do Nothing
    """

    def __init__(self, faction_id: int):
        super().__init__(faction_id)
        self.name = "TestingBot"
        self.difficulty = "testing"
        self.behavior_tree = self._build_tree()

    def _build_tree(self):
        """
        Construct the behavior tree for Normal bot.

        Tree executes from top to bottom, first SUCCESS wins.
        """
        return Selector(
            children=[
                # ========================================
                # BRANCH 1: HOUSING (Highest Priority)
                # ========================================
                # Population growth is critical - prioritize housing above all else
                Sequence(
                    children=[
                        Condition(needs_housing, "Needs Housing?"),
                        BTAction(try_build_house, "Build House"),
                    ],
                    name="Housing Priority"
                ),

                # ========================================
                # BRANCH 2: WORKER DEFICIT (Build houses for more workers)
                # ========================================
                # If we don't have enough workers AND population is near capacity, build houses
                Sequence(
                    children=[
                        Condition(has_worker_deficit, "Worker Deficit?"),
                        Condition(needs_housing, "Near Capacity?"),
                        BTAction(try_build_house, "Build House (More Workers Needed)"),
                    ],
                    name="Worker Deficit Recovery"
                ),

                # ========================================
                # BRANCH 3: CRITICAL RESOURCES (Emergency Recovery)
                # ========================================
                # If resources are critically low, prioritize recovery to avoid deadlock
                Selector(
                    children=[
                        # Critical stone shortage - build quarry immediately
                        Sequence(
                            children=[
                                Condition(has_critically_low_stone, "Stone Critically Low?"),
                                BTAction(try_build_quarry, "Build Quarry (EMERGENCY)"),
                            ],
                            name="Emergency Stone Recovery"
                        ),
                        # Critical wood shortage - build lumberyard immediately
                        Sequence(
                            children=[
                                Condition(has_critically_low_wood, "Wood Critically Low?"),
                                BTAction(try_build_lumberyard, "Build Lumberyard (EMERGENCY)"),
                            ],
                            name="Emergency Wood Recovery"
                        ),
                    ],
                    name="Critical Resources Branch"
                ),

                # ========================================
                # BRANCH 4: DEFENSE
                # ========================================
                # If under heavy attack, prioritize military defense
                Sequence(
                    children=[
                        Condition(is_under_attack, "Is Under Attack?"),
                        Selector(
                            children=[
                                # Try to train soldiers if we have barracks
                                Sequence(
                                    children=[
                                        Condition(has_barracks, "Has Barracks?"),
                                        BTAction(try_train_soldier, "Train Soldier (Defense)"),
                                    ],
                                    name="Emergency Military Training"
                                ),
                                # Or build barracks if we don't have one
                                BTAction(try_build_barracks, "Build Barracks (Emergency)"),
                            ],
                            name="Defensive Actions"
                        )
                    ],
                    name="Defense Branch"
                ),

                # ========================================
                # BRANCH 5: ATTACK (When Ready)
                # ========================================
                # Attack if we have military advantage
                Sequence(
                    children=[
                        Condition(can_attack_opponent, "Strong Enough to Attack?"),
                        BTAction(try_attack_opponent, "Attack Opponent"),
                    ],
                    name="Attack Branch"
                ),

                # ========================================
                # BRANCH 6: ECONOMY (Core Growth)
                # ========================================
                # Build economy in order: wood → food → stone
                # Ensure food production to prevent starvation!
                Selector(
                    children=[
                        # Wood production (HIGHEST PRIORITY - needed for all construction)
                        Sequence(
                            children=[
                                Condition(has_few_lumberyards, "Few Lumberyards?"),
                                BTAction(try_build_lumberyard, "Build Lumberyard"),
                            ],
                            name="Wood Production"
                        ),

                        # Edible food FIRST (prevent starvation). Grain is not food:
                        # a hunting shed yields meat directly, no processing chain.
                        Sequence(
                            children=[
                                Condition(_needs_hunting_shed, "No Hunting Shed?"),
                                BTAction(try_build_hunting_shed, "Build Hunting Shed"),
                            ],
                            name="Food Security"
                        ),

                        # Grain production (feeds the bread chain and cattle later)
                        Sequence(
                            children=[
                                Condition(has_few_farms, "Few Farms?"),
                                BTAction(try_build_farm, "Build Farm"),
                            ],
                            name="Grain Production"
                        ),

                        # Stone production (THIRD - after food secured)
                        Sequence(
                            children=[
                                Condition(has_few_quarries, "Few Quarries?"),
                                BTAction(try_build_quarry, "Build Quarry"),
                            ],
                            name="Stone Production"
                        ),
                    ],
                    name="Economy Branch"
                ),

                # ========================================
                # BRANCH 7: MILITARY DEVELOPMENT
                # ========================================
                # Build military gradually (not rushing, but staying prepared)
                Selector(
                    children=[
                        # Get first barracks (mid-priority)
                        Sequence(
                            children=[
                                Condition(has_no_barracks, "No Barracks?"),
                                Condition(has_good_economy, "Economy Established?"),
                                BTAction(try_build_barracks, "Build First Barracks"),
                            ],
                            name="Barracks Construction"
                        ),

                        # Train military units (if we need more)
                        Sequence(
                            children=[
                                Condition(has_barracks, "Has Barracks?"),
                                Condition(needs_more_military, "Needs More Military?"),
                                Selector(
                                    children=[
                                        BTAction(try_train_soldier, "Train Soldier"),
                                        BTAction(try_train_archer, "Train Archer"),
                                    ],
                                    name="Unit Training"
                                )
                            ],
                            name="Military Training"
                        ),
                    ],
                    name="Military Development Branch"
                ),

                # ========================================
                # BRANCH 8: FALLBACK (Always Succeeds)
                # ========================================
                # If nothing else can be done, do nothing and save resources
                BTAction(do_nothing, "Do Nothing (Saving Resources)"),
            ],
            name="TestingBot Root"
        )

    def act(self, game_state: GameState, game_time: float) -> Action:
        """
        Execute behavior tree to choose action.

        Args:
            game_state: Current game state
            game_time: Current game time in seconds

        Returns:
            Action to execute
        """
        # Get faction and opponent
        faction = self._get_faction(game_state)
        opponent = self._get_opponent(game_state)

        if not faction:
            # Fallback if faction not found
            from ..actions import ActionType
            return Action(faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)

        # Build context for behavior tree
        context = {
            'game_state': game_state,
            'faction': faction,
            'opponent': opponent,
            'faction_id': self.faction_id,
            'game_time': game_time,
            'chosen_action': None,  # Will be set by Action nodes
        }

        # Execute behavior tree
        self.behavior_tree.tick(context)

        # Return chosen action (or DO_NOTHING if tree failed)
        action = context.get('chosen_action')
        if action is None:
            from ..actions import ActionType
            action = Action(faction_id=self.faction_id, action_type=ActionType.DO_NOTHING)

        return action


# Deprecated alias: the class was NormalBot before it was demoted to a test fixture.
# Kept so existing tests that `from ... import NormalBot` still resolve.
NormalBot = TestingBot
