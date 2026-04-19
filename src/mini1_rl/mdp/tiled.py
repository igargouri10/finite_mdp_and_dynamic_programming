r"""Tree-tiled MRPs built from a Cartesian product of 2D tiles (Mini-1, Task 7).

The assignment asks for the composition of several grid-world tiles into a
single environment *without* using 3D arrays or dense tensors, and suggests a
binary tree of height 3 (i.e. up to 8 leaf tiles) glued by Cartesian products.

We represent the composed environment as a *tree of tiles* where each internal
node records only a small amount of metadata about how its two children are
spliced together. The global state space is the disjoint union of the states
of the leaves, and the transition matrix is assembled only where needed.

Because SciPy is an optional dependency, we expose both a plain 2D dense
matrix and a hand-rolled CSR-like triple (``row``, ``col``, ``data``) that
provides the mat-vec product. Conceptually this is the "Cartesian product of
matrix tiles" structure the assignment describes, where composition is a
*pair* of sparse edges added to a block-diagonal matrix (Kronecker-style in
spirit, but we avoid the :math:`(|S_1| |S_2|)`-sized dense Kronecker product
which would blow up memory for height-3 trees).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import MarkovRewardProcess
from .gridworld import Cell, _neighbours, _valid_cells


# ---------------------------------------------------------------------------
@dataclass(eq=False)
class TileNode:
    """Binary tree of tiles; leaves carry a 2D mask, internal nodes two children."""
    mask: Optional[np.ndarray] = None            # set on leaves
    left: Optional["TileNode"] = None
    right: Optional["TileNode"] = None
    # Portal connecting `left` to `right`: cell_in_left <-> cell_in_right
    portal_left: Optional[Tuple[int, int]] = None
    portal_right: Optional[Tuple[int, int]] = None
    name: str = "tile"

    def is_leaf(self) -> bool:
        return self.mask is not None

    def height(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(
            self.left.height() if self.left else 0,
            self.right.height() if self.right else 0,
        )

    def leaves(self) -> List["TileNode"]:
        if self.is_leaf():
            return [self]
        out: List["TileNode"] = []
        if self.left is not None:
            out.extend(self.left.leaves())
        if self.right is not None:
            out.extend(self.right.leaves())
        return out


# ---------------------------------------------------------------------------
class _SparseCOO:
    """Tiny COO/CSR-style sparse matrix storing only the mat-vec product.

    We roll our own because SciPy is optional. For the tree-tiled grid
    worlds with up to a few hundred states and O(|S|) non-zeros this is
    memory-efficient and fast enough for experimental use.
    """

    __slots__ = ("n", "rows", "cols", "data")

    def __init__(self, n: int) -> None:
        self.n = n
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []

    def add(self, i: int, j: int, v: float) -> None:
        self.rows.append(i); self.cols.append(j); self.data.append(v)

    def zero_row(self, i: int) -> None:
        mask = [r != i for r in self.rows]
        self.rows = [r for r, m in zip(self.rows, mask) if m]
        self.cols = [c for c, m in zip(self.cols, mask) if m]
        self.data = [d for d, m in zip(self.data, mask) if m]

    @property
    def nnz(self) -> int:
        return len(self.data)

    def matvec(self, V: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n)
        for i, j, v in zip(self.rows, self.cols, self.data):
            out[i] += v * V[j]
        return out

    def todense(self) -> np.ndarray:
        M = np.zeros((self.n, self.n))
        for i, j, v in zip(self.rows, self.cols, self.data):
            M[i, j] += v
        return M


# ---------------------------------------------------------------------------
class TreeTiledMRP(MarkovRewardProcess):
    r"""MRP obtained by gluing tile masks along a binary tree.

    Parameters
    ----------
    root : TileNode
        Root of the binary tile tree. Each leaf has a 2D mask; each internal
        node has ``portal_left`` and ``portal_right`` giving the
        ``(row, col)`` coordinates *within the respective subtree's first leaf*
        that should be identified.
    gamma : float
        Discount factor.

    Notes
    -----
    The transition matrix is kept sparse for iterative solves, but we
    additionally materialise it as a dense :math:`|S| \times |S|` matrix
    *only* when the base class requires it. For larger trees one can swap
    that out by implementing the Bellman iteration against ``_sparse`` directly.
    """

    def __init__(
        self,
        root: TileNode,
        *,
        gamma: float = 0.9,
        name: str = "TreeTiledMRP",
    ) -> None:
        self._root = root
        leaves = root.leaves()
        if not leaves:
            raise ValueError("Tree has no leaves")
        leaf_id_to_idx: Dict[int, int] = {id(leaf): i for i, leaf in enumerate(leaves)}

        # Assign global indices to (leaf_idx, cell) pairs.
        self._leaf_cells: List[Tuple[int, Cell]] = []
        for leaf_idx, leaf in enumerate(leaves):
            cells, _ = _valid_cells(leaf.mask)
            for cell in cells:
                self._leaf_cells.append((leaf_idx, cell))
        n = len(self._leaf_cells)
        global_idx: Dict[Tuple[int, Cell], int] = {
            (li, cell): i for i, (li, cell) in enumerate(self._leaf_cells)
        }

        sparse = _SparseCOO(n)
        R = np.zeros(n)
        for (li, cell), i in global_idx.items():
            leaf = leaves[li]
            R[i] = leaf.mask[cell.r, cell.c]
            cells_in_leaf, _ = _valid_cells(leaf.mask)
            idx_in_leaf = {c: 0 for c in cells_in_leaf}
            for _, s_next in _neighbours(cell, leaf.mask, idx_in_leaf):
                sparse.add(i, global_idx[(li, s_next)], 0.25)

        # Install portals at every internal node. For each internal node we
        # identify one cell from the "first leaf" of each subtree and make
        # stepping onto it deterministically teleport to the paired cell.
        self._portals: List[Tuple[Tuple[int, Cell], Tuple[int, Cell]]] = []
        first_leaf_index: Dict[int, int] = {}

        def _assign_first_leaf(node: TileNode) -> int:
            if node.is_leaf():
                return leaf_id_to_idx[id(node)]
            l0 = _assign_first_leaf(node.left)
            r0 = _assign_first_leaf(node.right)
            first_leaf_index[id(node.left)] = l0
            first_leaf_index[id(node.right)] = r0
            return l0

        _assign_first_leaf(root)

        def _install_portals(node: TileNode) -> None:
            if node.is_leaf():
                return
            if node.portal_left is not None and node.portal_right is not None:
                li = first_leaf_index[id(node.left)]
                ri = first_leaf_index[id(node.right)]
                cell_l = Cell(*node.portal_left)
                cell_r = Cell(*node.portal_right)
                if (li, cell_l) in global_idx and (ri, cell_r) in global_idx:
                    i = global_idx[(li, cell_l)]
                    j = global_idx[(ri, cell_r)]
                    # Zero out i-th and j-th rows, then redirect entirely.
                    sparse.zero_row(i)
                    sparse.zero_row(j)
                    sparse.add(i, j, 1.0)
                    sparse.add(j, i, 1.0)
                    self._portals.append(((li, cell_l), (ri, cell_r)))
            _install_portals(node.left)
            _install_portals(node.right)

        _install_portals(root)
        self._sparse = sparse

        # The base class validates that P is row-stochastic; for the sizes
        # used in Mini-1 the dense materialisation is negligible cost.
        P_dense = sparse.todense()
        states = [(li, cell) for (li, cell) in self._leaf_cells]
        super().__init__(
            states, P_dense, R,
            gamma=gamma, terminal_states=(), name=name,
        )

    # -- sparse-aware Bellman operator -------------------------------------
    def bellman_operator_sparse(self, V: np.ndarray) -> np.ndarray:
        return self._R + self.gamma * self._sparse.matvec(V)

    def nnz(self) -> int:
        return self._sparse.nnz

    def tile_of(self, global_state_index: int) -> int:
        return self._leaf_cells[global_state_index][0]
