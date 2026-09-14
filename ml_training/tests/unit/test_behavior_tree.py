"""
Unit tests for behavior tree nodes.
Tests the core behavior tree framework.
"""

import sys
import os

# Add path to behavior_tree module directly
behavior_tree_path = os.path.join(os.path.dirname(__file__), '..', '..', 'simulator', 'behavior_tree')
sys.path.insert(0, behavior_tree_path)

import pytest
from nodes import (
    NodeStatus,
    Selector,
    Sequence,
    Condition,
    Action,
    Inverter,
    AlwaysSucceed,
)


class TestConditionNode:
    """Test Condition node behavior."""

    def test_condition_success(self):
        """Condition returns SUCCESS when true."""
        condition = Condition(lambda ctx: True, "Always True")
        result = condition.tick({})
        assert result == NodeStatus.SUCCESS

    def test_condition_failure(self):
        """Condition returns FAILURE when false."""
        condition = Condition(lambda ctx: False, "Always False")
        result = condition.tick({})
        assert result == NodeStatus.FAILURE

    def test_condition_uses_context(self):
        """Condition can access context data."""
        condition = Condition(lambda ctx: ctx.get('test_value', 0) > 5, "Check Value")

        result1 = condition.tick({'test_value': 3})
        assert result1 == NodeStatus.FAILURE

        result2 = condition.tick({'test_value': 10})
        assert result2 == NodeStatus.SUCCESS

    def test_condition_handles_exception(self):
        """Condition returns FAILURE on exception."""
        def failing_fn(ctx):
            raise ValueError("Test error")

        condition = Condition(failing_fn, "Failing Condition")
        result = condition.tick({})
        assert result == NodeStatus.FAILURE


class TestActionNode:
    """Test Action node behavior."""

    def test_action_success(self):
        """Action returns SUCCESS when action is performed."""
        action = Action(lambda ctx: "some_action", "Perform Action")
        context = {}
        result = action.tick(context)

        assert result == NodeStatus.SUCCESS
        assert context['chosen_action'] == "some_action"

    def test_action_failure(self):
        """Action returns FAILURE when it returns None."""
        action = Action(lambda ctx: None, "Failed Action")
        context = {}
        result = action.tick(context)

        assert result == NodeStatus.FAILURE
        assert 'chosen_action' not in context

    def test_action_handles_exception(self):
        """Action returns FAILURE on exception."""
        def failing_fn(ctx):
            raise ValueError("Test error")

        action = Action(failing_fn, "Failing Action")
        result = action.tick({})
        assert result == NodeStatus.FAILURE


class TestSelectorNode:
    """Test Selector (OR) composite node."""

    def test_selector_first_child_succeeds(self):
        """Selector succeeds on first child success."""
        selector = Selector([
            Condition(lambda ctx: True, "True"),
            Condition(lambda ctx: False, "False"),
        ])

        result = selector.tick({})
        assert result == NodeStatus.SUCCESS

    def test_selector_tries_until_success(self):
        """Selector tries children until one succeeds."""
        call_count = {'count': 0}

        def increment(ctx):
            call_count['count'] += 1
            return False

        selector = Selector([
            Condition(increment, "Fail 1"),
            Condition(increment, "Fail 2"),
            Condition(lambda ctx: True, "Success on 3rd"),
        ])

        result = selector.tick({})
        assert result == NodeStatus.SUCCESS
        assert call_count['count'] == 2  # First two children called before third succeeds

    def test_selector_fails_when_all_fail(self):
        """Selector fails when all children fail."""
        selector = Selector([
            Condition(lambda ctx: False, "Fail 1"),
            Condition(lambda ctx: False, "Fail 2"),
            Condition(lambda ctx: False, "Fail 3"),
        ])

        result = selector.tick({})
        assert result == NodeStatus.FAILURE

    def test_selector_empty_children(self):
        """Selector with no children fails."""
        selector = Selector([])
        result = selector.tick({})
        assert result == NodeStatus.FAILURE


class TestSequenceNode:
    """Test Sequence (AND) composite node."""

    def test_sequence_all_succeed(self):
        """Sequence succeeds when all children succeed."""
        sequence = Sequence([
            Condition(lambda ctx: True, "True 1"),
            Condition(lambda ctx: True, "True 2"),
            Condition(lambda ctx: True, "True 3"),
        ])

        result = sequence.tick({})
        assert result == NodeStatus.SUCCESS

    def test_sequence_stops_on_failure(self):
        """Sequence stops executing on first failure."""
        call_count = {'count': 0}

        def increment(ctx):
            call_count['count'] += 1
            return True

        sequence = Sequence([
            Condition(increment, "Inc 1"),
            Condition(lambda ctx: False, "Fail"),
            Condition(increment, "Inc 2"),  # Should not be called
        ])

        result = sequence.tick({})
        assert result == NodeStatus.FAILURE
        assert call_count['count'] == 1  # Only first increment called

    def test_sequence_fails_on_any_failure(self):
        """Sequence fails if any child fails."""
        sequence = Sequence([
            Condition(lambda ctx: True, "True"),
            Condition(lambda ctx: False, "False"),
            Condition(lambda ctx: True, "True"),
        ])

        result = sequence.tick({})
        assert result == NodeStatus.FAILURE

    def test_sequence_empty_children(self):
        """Sequence with no children succeeds."""
        sequence = Sequence([])
        result = sequence.tick({})
        assert result == NodeStatus.SUCCESS


class TestInverterNode:
    """Test Inverter decorator node."""

    def test_inverter_inverts_success(self):
        """Inverter turns SUCCESS into FAILURE."""
        inverter = Inverter(Condition(lambda ctx: True, "True"))
        result = inverter.tick({})
        assert result == NodeStatus.FAILURE

    def test_inverter_inverts_failure(self):
        """Inverter turns FAILURE into SUCCESS."""
        inverter = Inverter(Condition(lambda ctx: False, "False"))
        result = inverter.tick({})
        assert result == NodeStatus.SUCCESS


class TestAlwaysSucceedNode:
    """Test AlwaysSucceed decorator node."""

    def test_always_succeed_on_success(self):
        """AlwaysSucceed returns SUCCESS when child succeeds."""
        always = AlwaysSucceed(Condition(lambda ctx: True, "True"))
        result = always.tick({})
        assert result == NodeStatus.SUCCESS

    def test_always_succeed_on_failure(self):
        """AlwaysSucceed returns SUCCESS even when child fails."""
        always = AlwaysSucceed(Condition(lambda ctx: False, "False"))
        result = always.tick({})
        assert result == NodeStatus.SUCCESS


class TestComplexTree:
    """Test complex behavior tree combinations."""

    def test_nested_selectors_and_sequences(self):
        """Test tree with nested composites."""
        # Tree: Selector([Sequence([Fail, True]), True])
        # Should: Try first sequence, fail on first condition, then succeed on second selector child
        tree = Selector([
            Sequence([
                Condition(lambda ctx: False, "Fail in Sequence"),
                Condition(lambda ctx: True, "True in Sequence"),
            ]),
            Condition(lambda ctx: True, "Fallback Success"),
        ])

        result = tree.tick({})
        assert result == NodeStatus.SUCCESS

    def test_context_modification(self):
        """Test that actions can modify context."""
        def set_value(ctx):
            ctx['value'] = 42
            return "action_performed"

        tree = Sequence([
            Action(set_value, "Set Value"),
            Condition(lambda ctx: ctx.get('value') == 42, "Check Value"),
        ])

        context = {}
        result = tree.tick(context)
        assert result == NodeStatus.SUCCESS
        assert context['value'] == 42


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
