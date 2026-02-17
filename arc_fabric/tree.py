"""Trajectory tree data structure for Arc Fabric.

Tracks all checkpoints across sessions as nodes in a branching tree.
Supports lineage queries, fork-point detection, and D3.js export for the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TreeNode:
    """A single node in the trajectory tree, corresponding to one checkpoint."""

    checkpoint_id: str
    session_id: str
    frame_index: int
    prompt: str
    action: dict | None        # Action that led to this node (JSON-serializable)
    parent_id: str | None      # Parent node's checkpoint_id
    children_ids: list[str]    # Child node checkpoint_ids
    metadata: dict             # Extra info (timing, success flag, etc.)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this node to a JSON-safe dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "prompt": self.prompt,
            "action": self.action,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "metadata": dict(self.metadata),
        }


class TrajectoryTree:
    """Tree structure for managing branching trajectories across sessions.

    The tree is append-only: nodes are never deleted, even after rewind.
    A single tree instance is shared across all sessions to maintain a
    unified view of the full exploration history.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, TreeNode] = {}
        self.root_id: str | None = None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_node(
        self,
        checkpoint_id: str,
        session_id: str,
        frame_index: int,
        prompt: str,
        action: dict | None = None,
        parent_id: str | None = None,
        metadata: dict | None = None,
    ) -> TreeNode:
        """Create a new tree node and wire it into the tree.

        If *parent_id* is provided the new node is appended as a child of that
        parent.  If the tree is empty (no root), the new node becomes the root
        regardless of *parent_id*.

        Raises:
            KeyError: If *parent_id* is given but does not exist in the tree.
            ValueError: If *checkpoint_id* already exists.
        """
        if checkpoint_id in self.nodes:
            raise ValueError(
                f"Node with checkpoint_id '{checkpoint_id}' already exists in the tree"
            )

        if parent_id is not None and parent_id not in self.nodes:
            raise KeyError(
                f"Parent node '{parent_id}' not found in the tree"
            )

        node_metadata = metadata if metadata is not None else {}
        node_metadata.setdefault("created_at", time.time())

        node = TreeNode(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            frame_index=frame_index,
            prompt=prompt,
            action=action,
            parent_id=parent_id,
            children_ids=[],
            metadata=node_metadata,
        )

        self.nodes[checkpoint_id] = node

        # Wire into parent
        if parent_id is not None:
            self.nodes[parent_id].children_ids.append(checkpoint_id)

        # First node in the tree becomes the root
        if self.root_id is None:
            self.root_id = checkpoint_id

        return node

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, checkpoint_id: str) -> TreeNode:
        """Return the node for *checkpoint_id*.

        Raises:
            KeyError: If the checkpoint does not exist.
        """
        try:
            return self.nodes[checkpoint_id]
        except KeyError:
            raise KeyError(
                f"Node '{checkpoint_id}' not found in the trajectory tree"
            ) from None

    def get_children(self, checkpoint_id: str) -> list[TreeNode]:
        """Return the direct children of *checkpoint_id*."""
        node = self.get_node(checkpoint_id)
        return [self.nodes[cid] for cid in node.children_ids]

    def get_lineage(self, checkpoint_id: str) -> list[TreeNode]:
        """Walk from *checkpoint_id* up to the root, returning the path.

        The returned list starts at *checkpoint_id* and ends at the root.
        """
        path: list[TreeNode] = []
        current_id: str | None = checkpoint_id
        visited: set[str] = set()

        while current_id is not None:
            if current_id in visited:
                raise RuntimeError(
                    f"Cycle detected at '{current_id}' while walking lineage"
                )
            visited.add(current_id)
            node = self.get_node(current_id)
            path.append(node)
            current_id = node.parent_id

        return path

    def get_branch(self, checkpoint_id: str) -> list[TreeNode]:
        """Return all nodes from root down to *checkpoint_id* (inclusive).

        This is the lineage in chronological (root-first) order.
        """
        return list(reversed(self.get_lineage(checkpoint_id)))

    def get_leaves(self) -> list[TreeNode]:
        """Return all leaf nodes (nodes with no children)."""
        return [
            node for node in self.nodes.values()
            if len(node.children_ids) == 0
        ]

    def get_fork_points(self) -> list[TreeNode]:
        """Return all nodes that have more than one child (branch points)."""
        return [
            node for node in self.nodes.values()
            if len(node.children_ids) > 1
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tree as a JSON-compatible dictionary.

        Returns a dict with ``root_id`` and a flat ``nodes`` mapping keyed
        by checkpoint_id.
        """
        return {
            "root_id": self.root_id,
            "nodes": {
                cid: node.to_dict() for cid, node in self.nodes.items()
            },
        }

    def to_d3_tree(self) -> dict[str, Any]:
        """Convert the tree to a D3.js-compatible nested format.

        Returns a nested dict starting at the root where each node has:
        - ``name``: human-readable label (``frame_{index}``)
        - ``checkpoint_id``, ``session_id``, ``frame_index``
        - ``prompt``, ``action``, ``metadata``
        - ``children``: list of child dicts (recursive)

        Returns an empty dict if the tree has no nodes.
        """
        if self.root_id is None:
            return {}
        return self._build_d3_subtree(self.root_id)

    def _build_d3_subtree(self, checkpoint_id: str) -> dict[str, Any]:
        """Recursively build a D3.js subtree rooted at *checkpoint_id*."""
        node = self.nodes[checkpoint_id]
        return {
            "name": f"frame_{node.frame_index}",
            "checkpoint_id": node.checkpoint_id,
            "session_id": node.session_id,
            "frame_index": node.frame_index,
            "prompt": node.prompt,
            "action": node.action,
            "metadata": dict(node.metadata),
            "children": [
                self._build_d3_subtree(cid) for cid in node.children_ids
            ],
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self.nodes

    def __repr__(self) -> str:
        return (
            f"TrajectoryTree(nodes={len(self.nodes)}, "
            f"root={self.root_id!r})"
        )
