"""Maze data structure.

The maze is a grid of cells. Each cell stores a 4-bit mask describing which of
its sides are walled (see ``config.N/E/S/W``). A wall is shared between two
neighbouring cells, so carving a passage always clears the bit on *both* sides.

Coordinates are ``(x, y)`` where ``x`` is the column and ``y`` is the row, with
``y`` increasing downward so that grid coordinates map directly onto screen
coordinates.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator

import numpy as np

from ..config import COST_FLOOR, DELTA, DIRECTIONS, N, E, S, W, OPPOSITE

Cell = tuple[int, int]

ALL_WALLS = N | E | S | W


class Maze:
    """A rectangular grid maze with per-cell terrain costs."""

    __slots__ = ("width", "height", "walls", "cost")

    def __init__(self, width: int, height: int) -> None:
        if width < 2 or height < 2:
            raise ValueError("maze must be at least 2x2")
        self.width = int(width)
        self.height = int(height)
        # Start fully walled; generators carve passages out of this block.
        self.walls = np.full((self.height, self.width), ALL_WALLS, dtype=np.uint8)
        self.cost = np.full((self.height, self.width), COST_FLOOR, dtype=np.float32)

    # -- basic queries ----------------------------------------------------
    def in_bounds(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def has_wall(self, cell: Cell, direction: int) -> bool:
        x, y = cell
        return bool(self.walls[y, x] & direction)

    def cell_cost(self, cell: Cell) -> float:
        x, y = cell
        return float(self.cost[y, x])

    def is_mud(self, cell: Cell) -> bool:
        return self.cell_cost(cell) > COST_FLOOR

    @property
    def area(self) -> int:
        return self.width * self.height

    def cells(self) -> Iterator[Cell]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    # -- structural edits -------------------------------------------------
    def carve(self, cell: Cell, direction: int) -> None:
        """Open the wall between ``cell`` and its neighbour in ``direction``."""
        x, y = cell
        dx, dy = DELTA[direction]
        nx, ny = x + dx, y + dy
        if not self.in_bounds((nx, ny)):
            return
        # Mask to a byte: numpy refuses to store the negative value that a bare
        # ``~direction`` produces on a Python int.
        self.walls[y, x] &= ~direction & 0xFF
        self.walls[ny, nx] &= ~OPPOSITE[direction] & 0xFF

    def build_wall(self, cell: Cell, direction: int) -> None:
        """Close the wall between ``cell`` and its neighbour in ``direction``."""
        x, y = cell
        dx, dy = DELTA[direction]
        nx, ny = x + dx, y + dy
        self.walls[y, x] |= direction
        if self.in_bounds((nx, ny)):
            self.walls[ny, nx] |= OPPOSITE[direction]

    def set_cost(self, cell: Cell, value: float) -> None:
        x, y = cell
        self.cost[y, x] = value

    # -- graph view -------------------------------------------------------
    def neighbours(self, cell: Cell) -> list[Cell]:
        """Cells reachable from ``cell`` in one step (no wall in between)."""
        x, y = cell
        mask = self.walls[y, x]
        out: list[Cell] = []
        for d in DIRECTIONS:
            if mask & d:
                continue
            dx, dy = DELTA[d]
            nb = (x + dx, y + dy)
            if self.in_bounds(nb):
                out.append(nb)
        return out

    def degree(self, cell: Cell) -> int:
        return len(self.neighbours(cell))

    def move_cost(self, _src: Cell, dst: Cell) -> float:
        """Cost of stepping onto ``dst``. Entering mud is expensive.

        Note this makes the graph **directed**: a step costs whatever the cell
        you arrive at costs, so walking A->B and B->A differ whenever the two
        cells have different terrain. That is the natural model for a game (you
        pay for the ground you land on), but it means a distance field computed
        *outward from* a cell is not the same thing as the cost *to reach* that
        cell - see :meth:`distance_field`.
        """
        return self.cell_cost(dst)

    # -- whole-grid analyses ---------------------------------------------
    def distance_field(
        self, source: Cell, weighted: bool = True, reverse: bool = False
    ) -> np.ndarray:
        """Shortest cost between ``source`` and every cell (``inf`` if cut off).

        ``reverse=False`` (default) gives the cost of travelling **from**
        ``source`` **to** each cell. ``reverse=True`` gives the cost of
        travelling **from each cell to** ``source`` - which is exactly ``h*``,
        the quantity a heuristic is trying to estimate.

        The two differ because :meth:`move_cost` charges for the destination
        cell. Writing ``D(c)`` for the cost from ``c`` to the source::

            D(source) = 0
            D(c)      = min over neighbours v of [ cost(v) + D(v) ]

        so the reverse sweep relaxes with the cost of the node being *expanded*
        rather than the neighbour being reached. Getting this backwards inflates
        or deflates every label by ``cost(goal) - cost(cell)``, which is exactly
        the kind of quiet off-by-a-little that makes a "provably optimal"
        algorithm score 0.999 against its own ground truth.

        One sweep labels every cell at once, which is what makes building the
        training set for the learned heuristic cheap.
        """
        dist = np.full((self.height, self.width), np.inf, dtype=np.float64)
        sx, sy = source
        dist[sy, sx] = 0.0

        if not weighted:
            # Unweighted: every step costs 1, so direction cannot matter.
            queue = deque([source])
            while queue:
                cur = queue.popleft()
                base = dist[cur[1], cur[0]]
                for nb in self.neighbours(cur):
                    if dist[nb[1], nb[0]] == np.inf:
                        dist[nb[1], nb[0]] = base + 1.0
                        queue.append(nb)
            return dist

        import heapq

        heap: list[tuple[float, Cell]] = [(0.0, source)]
        settled = np.zeros((self.height, self.width), dtype=bool)
        while heap:
            d, cur = heapq.heappop(heap)
            cx, cy = cur
            if settled[cy, cx]:
                continue
            settled[cy, cx] = True
            step_out = self.cell_cost(cur) if reverse else None
            for nb in self.neighbours(cur):
                nd = d + (step_out if reverse else self.move_cost(cur, nb))
                if nd < dist[nb[1], nb[0]]:
                    dist[nb[1], nb[0]] = nd
                    heapq.heappush(heap, (nd, nb))
        return dist

    def cost_to(self, goal: Cell) -> np.ndarray:
        """``h*`` field: true cheapest cost from every cell to ``goal``."""
        return self.distance_field(goal, weighted=True, reverse=True)

    def reachable_from(self, source: Cell) -> set[Cell]:
        seen = {source}
        stack = [source]
        while stack:
            cur = stack.pop()
            for nb in self.neighbours(cur):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return seen

    def is_fully_connected(self) -> bool:
        return len(self.reachable_from((0, 0))) == self.area

    def dead_ends(self) -> list[Cell]:
        return [c for c in self.cells() if self.degree(c) == 1]

    # -- serialisation ----------------------------------------------------
    def copy(self) -> "Maze":
        clone = Maze(self.width, self.height)
        clone.walls = self.walls.copy()
        clone.cost = self.cost.copy()
        return clone

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "walls": self.walls.tolist(),
            "cost": self.cost.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Maze":
        maze = cls(data["width"], data["height"])
        maze.walls = np.array(data["walls"], dtype=np.uint8)
        maze.cost = np.array(data["cost"], dtype=np.float32)
        return maze

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Maze {self.width}x{self.height}>"

    def to_ascii(self) -> str:
        """Render the maze as text. Handy for tests and terminal debugging."""
        lines = ["+" + "".join("--+" if self.has_wall((x, 0), N) else "  +"
                               for x in range(self.width))]
        for y in range(self.height):
            row = "|" if self.has_wall((0, y), W) else " "
            floor = "+"
            for x in range(self.width):
                row += "  " if not self.is_mud((x, y)) else "~~"
                row += "|" if self.has_wall((x, y), E) else " "
                floor += "--+" if self.has_wall((x, y), S) else "  +"
            lines.append(row)
            lines.append(floor)
        return "\n".join(lines)


def path_cost(maze: Maze, path: Iterable[Cell]) -> float:
    """Total terrain cost of walking ``path`` (the start cell is free)."""
    cells = list(path)
    if len(cells) < 2:
        return 0.0
    return float(sum(maze.move_cost(a, b) for a, b in zip(cells, cells[1:])))
