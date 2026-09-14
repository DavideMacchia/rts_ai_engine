"""
Behavior tree nodes for RTS bot AI.

Node types:
- Composite nodes: Selector, Sequence
- Leaf nodes: Condition, Action
- Decorator nodes: Inverter, Repeater

Status: SUCCESS, FAILURE, RUNNING
"""

from enum import Enum
from typing import Callable, List, Optional, Any
from abc import ABC, abstractmethod


class NodeStatus(Enum):
    """Execution status of a behavior tree node."""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BTNode(ABC):
    """Base class for all behavior tree nodes."""

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.status = NodeStatus.FAILURE

    @abstractmethod
    def tick(self, context: dict) -> NodeStatus:
        """
        Execute this node.

        Args:
            context: Dict containing game_state, faction, opponent, game_time, etc.

        Returns:
            NodeStatus indicating result
        """
        pass

    def reset(self):
        """Reset node state."""
        self.status = NodeStatus.FAILURE


# ============================================================================
# COMPOSITE NODES - Have children
# ============================================================================

class Selector(BTNode):
    """
    Selector (OR node) - Succeeds if ANY child succeeds.
    Executes children in order until one succeeds.

    Use for: Trying alternatives (do A OR B OR C)
    """

    def __init__(self, children: List[BTNode], name: str = "Selector"):
        super().__init__(name)
        self.children = children

    def tick(self, context: dict) -> NodeStatus:
        """Execute children until one succeeds."""
        for child in self.children:
            status = child.tick(context)

            if status == NodeStatus.SUCCESS:
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

        # All children failed
        self.status = NodeStatus.FAILURE
        return NodeStatus.FAILURE


class Sequence(BTNode):
    """
    Sequence (AND node) - Succeeds only if ALL children succeed.
    Executes children in order, stops at first failure.

    Use for: Multi-step plans (do A AND THEN B AND THEN C)
    """

    def __init__(self, children: List[BTNode], name: str = "Sequence"):
        super().__init__(name)
        self.children = children

    def tick(self, context: dict) -> NodeStatus:
        """Execute children in order, all must succeed."""
        for child in self.children:
            status = child.tick(context)

            if status == NodeStatus.FAILURE:
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE

            if status == NodeStatus.RUNNING:
                self.status = NodeStatus.RUNNING
                return NodeStatus.RUNNING

        # All children succeeded
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


# ============================================================================
# LEAF NODES - Do actual work
# ============================================================================

class Condition(BTNode):
    """
    Condition node - Tests a condition, returns SUCCESS or FAILURE.
    Never returns RUNNING.

    Use for: Checking game state (is resource low? is enemy nearby?)
    """

    def __init__(self, condition_fn: Callable[[dict], bool], name: str = "Condition"):
        """
        Args:
            condition_fn: Function that takes context dict and returns bool
            name: Descriptive name for debugging
        """
        super().__init__(name)
        self.condition_fn = condition_fn

    def tick(self, context: dict) -> NodeStatus:
        """Evaluate condition."""
        try:
            result = self.condition_fn(context)
            self.status = NodeStatus.SUCCESS if result else NodeStatus.FAILURE
            return self.status
        except Exception as e:
            # Condition failed due to error
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


class Action(BTNode):
    """
    Action node - Executes an action, returns SUCCESS, FAILURE, or RUNNING.

    Use for: Performing actions (build, attack, train)
    """

    def __init__(self, action_fn: Callable[[dict], Any], name: str = "Action"):
        """
        Args:
            action_fn: Function that takes context and returns Action or None
            name: Descriptive name for debugging
        """
        super().__init__(name)
        self.action_fn = action_fn

    def tick(self, context: dict) -> NodeStatus:
        """Execute action."""
        try:
            result = self.action_fn(context)

            # If action returns an Action object, that's success
            if result is not None:
                # Store action in context for bot to return
                context['chosen_action'] = result
                self.status = NodeStatus.SUCCESS
                return NodeStatus.SUCCESS
            else:
                # Action couldn't be executed (not affordable, etc.)
                self.status = NodeStatus.FAILURE
                return NodeStatus.FAILURE
        except Exception as e:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE


# ============================================================================
# DECORATOR NODES - Modify child behavior
# ============================================================================

class Inverter(BTNode):
    """
    Inverter - Inverts child result (SUCCESS <-> FAILURE).
    RUNNING stays RUNNING.

    Use for: Negating conditions (if NOT condition)
    """

    def __init__(self, child: BTNode, name: str = "Inverter"):
        super().__init__(name)
        self.child = child

    def tick(self, context: dict) -> NodeStatus:
        """Invert child result."""
        status = self.child.tick(context)

        if status == NodeStatus.SUCCESS:
            self.status = NodeStatus.FAILURE
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:
            # RUNNING stays RUNNING
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING


class AlwaysSucceed(BTNode):
    """
    Always returns SUCCESS regardless of child result.

    Use for: Optional actions that shouldn't block sequence
    """

    def __init__(self, child: BTNode, name: str = "AlwaysSucceed"):
        super().__init__(name)
        self.child = child

    def tick(self, context: dict) -> NodeStatus:
        """Execute child but always return success."""
        self.child.tick(context)
        self.status = NodeStatus.SUCCESS
        return NodeStatus.SUCCESS


class RepeatUntilFail(BTNode):
    """
    Repeats child until it fails.

    Use for: Executing multiple actions in priority order
    """

    def __init__(self, child: BTNode, name: str = "RepeatUntilFail"):
        super().__init__(name)
        self.child = child

    def tick(self, context: dict) -> NodeStatus:
        """Repeat child execution."""
        status = self.child.tick(context)

        if status == NodeStatus.FAILURE:
            self.status = NodeStatus.SUCCESS
            return NodeStatus.SUCCESS
        else:
            self.status = NodeStatus.RUNNING
            return NodeStatus.RUNNING
