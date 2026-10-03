"""
generator.py — Generador de instancias de Kakuro (para el dataset y los benchmarks).

1. Genera un patrón de bloques negros aleatorio y lo repara hasta que toda celda
   blanca pertenezca a una suma horizontal y a una vertical de longitud 2..9.
2. Rellena las celdas con dígitos que no se repiten en ninguna suma (CSP resuelto
   con el motor propio y orden de valores aleatorio).
3. Calcula las pistas. Opcionalmente exige solución única (verificada con el solver).
"""
from __future__ import annotations

import random
from typing import List, Optional

from .cp_engine import CPSolver
from .puzzle import Puzzle


def _repair(white: List[List[int]], rng: random.Random) -> None:
    R, C = len(white), len(white[0])
    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if not white[r][c]:
                    continue
                # longitud de la suma horizontal y vertical que contienen a (r, c)
                c0 = c
                while c0 - 1 >= 0 and white[r][c0 - 1]:
                    c0 -= 1
                c1 = c
                while c1 + 1 < C and white[r][c1 + 1]:
                    c1 += 1
                r0 = r
                while r0 - 1 >= 0 and white[r0 - 1][c]:
                    r0 -= 1
                r1 = r
                while r1 + 1 < R and white[r1 + 1][c]:
                    r1 += 1
                h, v = c1 - c0 + 1, r1 - r0 + 1
                if h == 1 or v == 1:
                    white[r][c] = 0; changed = True
                elif h > 9:
                    white[r][rng.randint(c0 + 2, c1 - 2)] = 0; changed = True
                elif v > 9:
                    white[rng.randint(r0 + 2, r1 - 2)][c] = 0; changed = True


def random_layout(rows: int, cols: int, rng: random.Random, density: float = 0.28,
                  symmetric: bool = True) -> List[List[int]]:
    while True:
        white = [[0] * cols for _ in range(rows)]
        for r in range(1, rows):
            for c in range(1, cols):
                white[r][c] = 1
        for r in range(1, rows):
            for c in range(1, cols):
                if rng.random() < density:
                    white[r][c] = 0
                    if symmetric:  # simetría rotacional (estética de los puzzles publicados)
                        rr, cc = rows - r, cols - c
                        if 1 <= rr < rows and 1 <= cc < cols:
                            white[rr][cc] = 0
        _repair(white, rng)
        n_white = sum(map(sum, white))
        if n_white >= 0.45 * (rows - 1) * (cols - 1):
            return white


def _fill(white: List[List[int]], rng: random.Random) -> Optional[List[List[int]]]:
    tmp = Puzzle(len(white), len(white[0]), white,
                 [[0] * len(white[0]) for _ in white], [[0] * len(white[0]) for _ in white])
    cells = tmp.white_cells()
    idx = {c: i for i, c in enumerate(cells)}
    cons = [([idx[c] for c in run.cells], 0) for run in tmp.runs()]  # total 0 = solo alldiff
    eng = CPSolver(len(cells), cons, rng=rng)
    sols = eng.solve(limit=1, time_limit=1.0)
    if not sols:
        return None
    sol = [[0] * len(white[0]) for _ in white]
    for c, v in zip(cells, sols[0]):
        sol[c[0]][c[1]] = v
    return sol


def _clues(p: Puzzle) -> None:
    for run in p.runs():
        s = sum(p.solution[r][c] for r, c in run.cells)
        (p.across if run.direction == "across" else p.down)[run.clue_cell[0]][run.clue_cell[1]] = s


def _count(p: Puzzle, cap: int):
    cells = p.white_cells()
    idx = {c: i for i, c in enumerate(cells)}
    eng = CPSolver(len(cells), [([idx[c] for c in r.cells], r.total) for r in p.runs()])
    sols = eng.solve(limit=cap, time_limit=2)
    return [{c: s[i] for c, i in idx.items()} for s in sols]


def _make_unique(p: Puzzle, rng: random.Random, steps: int = 400, cap: int = 8) -> bool:
    """Búsqueda local sobre los dígitos de la solución hasta que las pistas
    determinen una única solución (estrategia típica de generadores de Kakuro)."""
    runs_of = {c: [] for c in p.white_cells()}
    for run in p.runs():
        for c in run.cells:
            runs_of[c].append(run)
    sols = _count(p, cap)
    if not sols:          # tiempo agotado: instancia demasiado abierta
        return False
    for _ in range(steps):
        if len(sols) == 1:
            return True
        other = sols[1] if sols[0] == {c: p.solution[c[0]][c[1]] for c in sols[0]} else sols[0]
        diff = [c for c in other if other[c] != p.solution[c[0]][c[1]]]
        cell = rng.choice(diff)
        used = set()
        for run in runs_of[cell]:
            used |= {p.solution[r][c] for r, c in run.cells if (r, c) != cell}
        cand = [v for v in range(1, 10) if v not in used and v != p.solution[cell[0]][cell[1]]]
        if not cand:
            continue
        old = p.solution[cell[0]][cell[1]]
        p.solution[cell[0]][cell[1]] = rng.choice(cand)
        _clues(p)
        new = _count(p, cap)
        if new and (len(new) <= len(sols) or rng.random() < 0.1):
            sols = new
        else:
            p.solution[cell[0]][cell[1]] = old
            _clues(p)
    return len(sols) == 1


def make_puzzle(rows: int, cols: int, seed: int = 0, density: float = 0.33,
                unique: bool = True, max_tries: int = 30) -> Puzzle:
    """Genera un Kakuro válido de rows×cols (incluye la fila/columna de pistas)."""
    rng = random.Random(seed)
    p = None
    for attempt in range(max_tries):
        white = random_layout(rows, cols, rng, density)
        sol = _fill(white, rng)
        if sol is None:
            continue
        p = Puzzle(rows, cols, white, [[0] * cols for _ in range(rows)],
                   [[0] * cols for _ in range(rows)], sol)
        _clues(p)
        if not unique:
            p.meta = {"seed": seed, "unique": None}
            return p
        if _make_unique(p, rng):
            p.meta = {"seed": seed, "unique": True, "attempts": attempt + 1}
            return p
    p.meta = {"seed": seed, "unique": False}
    return p
