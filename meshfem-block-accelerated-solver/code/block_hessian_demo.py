#!/usr/bin/env python3
"""A tiny, NumPy-only illustration of MeshFEM's block-sparse viewpoint.

This is an educational proxy, not a reimplementation of BlockCatamari and not
a reproduction of the paper's benchmark.  Each edge contributes a 2-D spring
energy.  The demo assembles the same Hessian as (1) scalar dense storage and
(2) an upper-triangular block dictionary, then takes one exact Newton step.
"""

from __future__ import annotations

import numpy as np


def build_block_hessian(
    node_positions: np.ndarray,
    edges: list[tuple[int, int]],
    stiffness: float = 3.0,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Assemble a 2-D edge-energy Hessian in scalar and block forms."""
    n_nodes, dimension = node_positions.shape
    if dimension != 2:
        raise ValueError("This compact demo expects 2-D coordinates")
    scalar = np.zeros((n_nodes * dimension, n_nodes * dimension), dtype=float)
    blocks: dict[tuple[int, int], np.ndarray] = {}

    def add_block(row: int, col: int, value: np.ndarray) -> None:
        key = (min(row, col), max(row, col))
        # Store only the upper triangle.  In a real symmetric block CSC
        # implementation this avoids carrying a duplicate transposed block.
        if key not in blocks:
            blocks[key] = np.zeros((dimension, dimension), dtype=float)
        blocks[key] += value if row <= col else value.T

    for i, j in edges:
        delta = node_positions[j] - node_positions[i]
        length = np.linalg.norm(delta)
        if length == 0:
            raise ValueError("An edge has coincident endpoints")
        direction = delta / length
        local = stiffness * np.outer(direction, direction)
        for row_node, row_sign in ((i, 1.0), (j, -1.0)):
            for col_node, col_sign in ((i, 1.0), (j, -1.0)):
                block = row_sign * col_sign * local
                r = slice(2 * row_node, 2 * row_node + 2)
                c = slice(2 * col_node, 2 * col_node + 2)
                scalar[r, c] += block
                # Scalar assembly visits both symmetric entries; the block
                # representation stores only one upper-triangular copy.
                if row_node <= col_node:
                    add_block(row_node, col_node, block)

    return scalar, blocks


def block_to_dense(blocks: dict[tuple[int, int], np.ndarray], n_nodes: int) -> np.ndarray:
    """Expand upper-triangular blocks to a scalar dense matrix for checking."""
    dense = np.zeros((2 * n_nodes, 2 * n_nodes), dtype=float)
    for (i, j), block in blocks.items():
        r = slice(2 * i, 2 * i + 2)
        c = slice(2 * j, 2 * j + 2)
        dense[r, c] += block
        if i != j:
            dense[c, r] += block.T
    return dense


def main() -> None:
    # A small mesh-like graph.  The diagonal regularizer removes rigid modes,
    # so the positive-definite Cholesky/Newton demonstration is well posed.
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.4]],
        dtype=float,
    )
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 4), (2, 4)]
    scalar_h, blocks = build_block_hessian(positions, edges)
    scalar_h += 0.2 * np.eye(scalar_h.shape[0])
    blocks[(0, 0)] += 0.2 * np.eye(2)
    for node in range(1, positions.shape[0]):
        blocks[(node, node)] = blocks.get((node, node), np.zeros((2, 2))) + 0.2 * np.eye(2)

    expanded = block_to_dense(blocks, positions.shape[0])
    if not np.allclose(scalar_h, expanded, atol=1e-12):
        raise AssertionError("block expansion does not match scalar assembly")

    block_nnz = sum(1 for block in blocks.values() if np.any(np.abs(block) > 1e-12))
    scalar_nnz = int(np.count_nonzero(np.abs(scalar_h) > 1e-12))
    d = 2
    scalar_value_slots = scalar_nnz
    block_value_slots = block_nnz * d * d
    print(f"nodes={positions.shape[0]}, edges={len(edges)}, d={d}")
    print(f"nonzero scalar entries={scalar_nnz}")
    print(f"nonzero dxd blocks={block_nnz}")
    print(f"value-slot ratio block/scalar={block_value_slots / scalar_value_slots:.3f}")

    # Quadratic energy E(x)=1/2*x^T H*x-b^T*x.  For a quadratic, the Newton
    # step reaches the minimizer in one solve; this tests the assembled system.
    b = np.linspace(-0.7, 0.8, scalar_h.shape[0])
    x0 = np.zeros_like(b)
    gradient = scalar_h @ x0 - b
    factor = np.linalg.cholesky(scalar_h)
    y = np.linalg.solve(factor, -gradient)
    newton_step = np.linalg.solve(factor.T, y)
    residual = np.linalg.norm(scalar_h @ newton_step - b)
    print(f"Newton residual={residual:.3e}")
    if residual > 1e-10:
        raise AssertionError("Newton solve residual is too large")


if __name__ == "__main__":
    main()
