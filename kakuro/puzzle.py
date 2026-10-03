"""
puzzle.py — Representación de una instancia de Kakuro (estructura de datos de la Fase 1
que consume la Fase 2).

Formato JSON (independiente de la imagen):
{
  "rows": R, "cols": C,
  "white":  [[0/1, ...], ...],   # 1 = celda blanca (variable), 0 = celda negra (bloque)
  "down":   [[int, ...], ...],   # pista vertical en celdas negras (0 = sin pista)
  "across": [[int, ...], ...]    # pista horizontal en celdas negras (0 = sin pista)
}
Opcionalmente "solution": [[int, ...]] con 0 en las celdas negras.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple

Cell = Tuple[int, int]


@dataclass
class Run:
    """Una "suma" (entry) del Kakuro: celdas blancas consecutivas con una pista."""
    clue_cell: Cell          # celda negra que contiene la pista
    direction: str           # 'across' o 'down'
    cells: List[Cell]        # celdas blancas que la componen
    total: int               # suma objetivo

    def __len__(self) -> int:
        return len(self.cells)


@dataclass
class Puzzle:
    rows: int
    cols: int
    white: List[List[int]]
    down: List[List[int]]
    across: List[List[int]]
    solution: Optional[List[List[int]]] = None
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def from_dict(d: Dict) -> "Puzzle":
        return Puzzle(d["rows"], d["cols"], d["white"], d["down"], d["across"],
                      d.get("solution"), d.get("meta", {}))

    def to_dict(self, with_solution: bool = True) -> Dict:
        d = {"rows": self.rows, "cols": self.cols, "white": self.white,
             "down": self.down, "across": self.across}
        if with_solution and self.solution is not None:
            d["solution"] = self.solution
        if self.meta:
            d["meta"] = self.meta
        return d

    @staticmethod
    def load(path: str) -> "Puzzle":
        with open(path, encoding="utf-8") as f:
            return Puzzle.from_dict(json.load(f))

    def save(self, path: str, with_solution: bool = True) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(with_solution), f, indent=1)

    # ------------------------------------------------------------ estructura
    def white_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols) if self.white[r][c]]

    def runs(self) -> List[Run]:
        """Extrae todas las sumas: desde cada celda negra hacia la derecha / hacia abajo."""
        out: List[Run] = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.white[r][c]:
                    continue
                # horizontal
                cells = []
                cc = c + 1
                while cc < self.cols and self.white[r][cc]:
                    cells.append((r, cc)); cc += 1
                if cells:
                    out.append(Run((r, c), "across", cells, int(self.across[r][c])))
                # vertical
                cells = []
                rr = r + 1
                while rr < self.rows and self.white[rr][c]:
                    cells.append((rr, c)); rr += 1
                if cells:
                    out.append(Run((r, c), "down", cells, int(self.down[r][c])))
        return out

    def validate(self) -> List[str]:
        """Chequeos de consistencia estructural (útiles para detectar errores de visión)."""
        errs = []
        runs = self.runs()
        covered = {cell: [0, 0] for cell in self.white_cells()}
        for run in runs:
            lo, hi = sum_bounds(len(run))
            if len(run) > 9:
                errs.append(f"suma {run.direction} en {run.clue_cell} tiene {len(run)} celdas (>9)")
            if run.total == 0:
                errs.append(f"falta pista {run.direction} en {run.clue_cell}")
            elif not (lo <= run.total <= hi):
                errs.append(f"pista {run.total} imposible para {len(run)} celdas en {run.clue_cell}")
            for cell in run.cells:
                covered[cell][0 if run.direction == "across" else 1] += 1
        for cell, (a, d) in covered.items():
            if a == 0 or d == 0:
                errs.append(f"celda {cell} no pertenece a una suma horizontal y otra vertical")
        tot_a = sum(r.total for r in runs if r.direction == "across")
        tot_d = sum(r.total for r in runs if r.direction == "down")
        if tot_a != tot_d:
            errs.append(f"suma de pistas horizontales ({tot_a}) != verticales ({tot_d})")
        return errs

    def check_solution(self, sol: List[List[int]]) -> bool:
        """Verifica que una solución cumpla todas las reglas del Kakuro."""
        for run in self.runs():
            vals = [sol[r][c] for r, c in run.cells]
            if any(not 1 <= v <= 9 for v in vals):
                return False
            if len(set(vals)) != len(vals) or sum(vals) != run.total:
                return False
        return True

    def stats(self) -> Dict:
        runs = self.runs()
        return {"rows": self.rows, "cols": self.cols, "white_cells": len(self.white_cells()),
                "runs": len(runs), "max_run": max((len(r) for r in runs), default=0)}

    def __str__(self) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if self.white[r][c]:
                    v = self.solution[r][c] if self.solution else 0
                    row.append(f"  {v if v else '.'}  ")
                else:
                    d, a = self.down[r][c], self.across[r][c]
                    if d or a:
                        row.append(f"{d or '':>2}\\{a or '':<2}")
                    else:
                        row.append(" ### ")
            lines.append("|".join(row))
        return "\n".join(lines)


# --------------------------------------------------------------- combinatoria
def sum_bounds(length: int) -> Tuple[int, int]:
    """Suma mínima y máxima de `length` dígitos distintos en 1..9."""
    length = max(0, min(9, length))
    return sum(range(1, length + 1)), sum(range(10 - length, 10))


@lru_cache(maxsize=None)
def combos(length: int, total: int) -> Tuple[Tuple[int, ...], ...]:
    """Todos los conjuntos de `length` dígitos distintos de 1..9 cuya suma es `total`."""
    return tuple(c for c in combinations(range(1, 10), length) if sum(c) == total)


@lru_cache(maxsize=None)
def allowed_digits(length: int, total: int) -> Tuple[int, ...]:
    """Unión de los dígitos que aparecen en alguna combinación válida (filtrado de dominio)."""
    s = set()
    for c in combos(length, total):
        s.update(c)
    return tuple(sorted(s))
