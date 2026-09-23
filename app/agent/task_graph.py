"""Task Graph Engine for V5 — DAG-based task planning."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GraphNodeType(str, Enum):
    TASK = "TASK"
    SUBTASK = "SUBTASK"
    CONDITION = "CONDITION"
    PARALLEL_GROUP = "PARALLEL_GROUP"


class GraphNode(BaseModel):
    """A node in the task graph."""
    node_id: str
    node_type: GraphNodeType = GraphNodeType.SUBTASK
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)
    status: str = "PENDING"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskGraph:
    """DAG-based task graph for planning."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._execution_order: list[str] = []

    def add_node(self, node: GraphNode) -> None:
        self._nodes[node.node_id] = node

    def add_dependency(self, node_id: str, dependency_id: str) -> None:
        if node_id in self._nodes:
            if dependency_id not in self._nodes[node_id].dependencies:
                self._nodes[node_id].dependencies.append(dependency_id)

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def get_ready_nodes(self) -> list[GraphNode]:
        """Get nodes whose dependencies are all satisfied."""
        ready: list[GraphNode] = []
        for node in self._nodes.values():
            if node.status != "PENDING":
                continue
            deps_met = all(
                self._nodes.get(dep) is not None and self._nodes[dep].status == "COMPLETED"
                for dep in node.dependencies
            )
            if deps_met:
                ready.append(node)
        return ready

    def get_dependents(self, node_id: str) -> list[GraphNode]:
        """Get nodes that depend on the given node."""
        return [n for n in self._nodes.values() if node_id in n.dependencies]

    def mark_completed(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].status = "COMPLETED"

    def mark_failed(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].status = "FAILED"

    def mark_in_progress(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].status = "IN_PROGRESS"

    def has_cycles(self) -> bool:
        """Detect cycles in the graph using DFS."""
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dep in self._nodes.get(node_id, GraphNode(node_id="")).dependencies:
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.discard(node_id)
            return False

        for node_id in self._nodes:
            if node_id not in visited:
                if dfs(node_id):
                    return True
        return False

    def topological_sort(self) -> list[str]:
        """Return nodes in topological order."""
        visited: set[str] = set()
        order: list[str] = []

        def dfs(node_id: str) -> None:
            visited.add(node_id)
            for dep in self._nodes.get(node_id, GraphNode(node_id="")).dependencies:
                if dep not in visited:
                    dfs(dep)
            order.append(node_id)

        for node_id in self._nodes:
            if node_id not in visited:
                dfs(node_id)

        return order

    def get_independent_groups(self) -> list[list[str]]:
        """Find groups of nodes that can execute in parallel."""
        ready = self.get_ready_nodes()
        if not ready:
            return []

        groups: list[list[str]] = []
        remaining = [n.node_id for n in ready]

        while remaining:
            group: list[str] = []
            not_ready: list[str] = []

            for node_id in remaining:
                node = self._nodes[node_id]
                deps_in_remaining = [d for d in node.dependencies if d in remaining]
                if not deps_in_remaining:
                    group.append(node_id)
                else:
                    not_ready.append(node_id)

            if not group:
                groups.append(remaining)
                break

            groups.append(group)
            remaining = not_ready

        return groups

    def summary(self) -> dict[str, Any]:
        return {
            "total_nodes": len(self._nodes),
            "pending": sum(1 for n in self._nodes.values() if n.status == "PENDING"),
            "completed": sum(1 for n in self._nodes.values() if n.status == "COMPLETED"),
            "failed": sum(1 for n in self._nodes.values() if n.status == "FAILED"),
            "has_cycles": self.has_cycles(),
            "independent_groups": len(self.get_independent_groups()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self._nodes.values()],
            "execution_order": self.topological_sort(),
            "summary": self.summary(),
        }
