#!/usr/bin/env python3
"""Seven-node educational demo of pre-order escape-pointer traversal."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    data: int  # -1 means internal
    escape: int


NODES = [
    Node("I0", -1, -1),
    Node("I1", -1, 4),
    Node("L2", 2, 3),
    Node("L3", 3, 4),
    Node("I4", -1, -1),
    Node("L5", 5, 6),
    Node("L6", 6, -1),
]


def traverse(hit_by_name: dict[str, bool]) -> tuple[list[str], list[int]]:
    """Run the document's state machine with predetermined AABB hit results."""
    visited: list[str] = []
    collisions: list[int] = []
    idx = 0
    while idx != -1:
        node = NODES[idx]
        visited.append(node.name)
        if not hit_by_name.get(node.name, False):
            idx = node.escape
        elif node.data == -1:
            idx += 1
        else:
            collisions.append(node.data)
            idx = node.escape
    return visited, collisions


def main() -> None:
    cases = {
        "hit everything": {node.name: True for node in NODES},
        "miss left subtree": {"I0": True, "I1": False, "I4": True, "L5": True, "L6": False},
        "hit only L3": {"I0": True, "I1": True, "L2": False, "L3": True, "I4": False},
    }
    for label, hit_map in cases.items():
        visited, collisions = traverse(hit_map)
        print(f"{label:18} visited={visited} collisions={collisions}")


if __name__ == "__main__":
    main()
