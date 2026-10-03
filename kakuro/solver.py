"""
solver.py — Fase 2: modelo de Constraint Programming parametrizado para Kakuro.

Backends
--------
* "ortools":  Google OR-Tools CP-SAT (backend principal, recomendado).
* "native":   motor CP propio (kakuro.cp_engine) — respaldo sin dependencias.
* "auto":     OR-Tools si está instalado; si no, el motor propio.

Modelos (todos reciben un Puzzle cualquiera: nada está "hardcodeado")
---------------------------------------------------------------------
* "global"     : x_c ∈ {1..9};  AllDifferent(run) ∧ Sum(run) = pista        (modelo base)
* "global_dom" : "global" + filtrado de dominios por combinaciones válidas
                 + restricción de tabla (AddAllowedAssignments) en sumas cortas (≤ 4)
* "reified"    : codificación booleana one-hot con restricciones reificadas
                 b_{c,d} ⇔ (x_c = d) y AtMostOne por dígito (alldifferent descompuesto)

La verificación de unicidad usa restricciones (semi)reificadas:
    diff_c ⇒ x_c ≠ s_c   y   OR_c diff_c      (al menos una celda distinta de la 1ª solución)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional

from .puzzle import Puzzle, allowed_digits, combos
from . import cp_engine

try:  # OR-Tools es opcional para que el resto del pipeline funcione igual
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except Exception:  # pragma: no cover
    cp_model = None
    HAS_ORTOOLS = False

MODELS = ("global", "global_dom", "reified")


@dataclass
class SolveResult:
    status: str                       # "OPTIMAL/FEASIBLE", "INFEASIBLE", "TIMEOUT"
    solution: Optional[List[List[int]]]
    unique: Optional[bool]
    time_s: float                     # tiempo de la primera solución
    time_unique_s: float              # tiempo adicional de la prueba de unicidad
    backend: str
    model: str
    stats: Dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.solution is not None


def solve(puzzle: Puzzle, backend: str = "auto", model: str = "global_dom",
          check_unique: bool = True, time_limit: float = 60.0, workers: int = 8) -> SolveResult:
    if model not in MODELS:
        raise ValueError(f"modelo desconocido: {model}")
    if backend == "auto":
        backend = "ortools" if HAS_ORTOOLS else "native"
    if backend == "ortools":
        if not HAS_ORTOOLS:
            raise ImportError("OR-Tools no está instalado: pip install ortools")
        return _solve_ortools(puzzle, model, check_unique, time_limit, workers)
    return _solve_native(puzzle, model, check_unique, time_limit)


# =========================================================== OR-Tools CP-SAT
def _build_ortools(puzzle: Puzzle, model_name: str):
    m = cp_model.CpModel()
    cells = puzzle.white_cells()
    runs = puzzle.runs()

    # ---- dominios: {1..9} o filtrados por las combinaciones válidas de sus dos sumas
    dom = {c: set(range(1, 10)) for c in cells}
    if model_name == "global_dom":
        for run in runs:
            ok = set(allowed_digits(len(run), run.total))
            for c in run.cells:
                dom[c] &= ok
    x = {}
    for (r, c) in cells:
        vals = sorted(dom[(r, c)]) or [1]  # dominio vacío -> el modelo será infactible
        x[(r, c)] = m.NewIntVarFromDomain(cp_model.Domain.FromValues(vals), f"x_{r}_{c}")
    if any(not dom[c] for c in cells):
        bad = m.NewBoolVar("empty_domain")      # contradicción explícita
        m.Add(bad == 1)
        m.Add(bad == 0)

    b = {}
    if model_name == "reified":
        # ---- one-hot reificado: b[c,d] <=> (x_c == d)
        for c in cells:
            for d in range(1, 10):
                lit = m.NewBoolVar(f"b_{c[0]}_{c[1]}_{d}")
                m.Add(x[c] == d).OnlyEnforceIf(lit)
                m.Add(x[c] != d).OnlyEnforceIf(lit.Not())
                b[(c, d)] = lit
            m.AddExactlyOne(b[(c, d)] for d in range(1, 10))

    for run in runs:
        xs = [x[c] for c in run.cells]
        m.Add(sum(xs) == run.total)                              # restricción global sum
        if model_name == "reified":
            for d in range(1, 10):                               # alldifferent descompuesto
                m.AddAtMostOne(b[(c, d)] for c in run.cells)
        else:
            m.AddAllDifferent(xs)                                # restricción global alldifferent
            if model_name == "global_dom" and len(run) <= 4:
                tuples = [p for comb in combos(len(run), run.total) for p in permutations(comb)]
                m.AddAllowedAssignments(xs, tuples)              # restricción global table
    return m, x


def _solve_ortools(puzzle, model_name, check_unique, time_limit, workers) -> SolveResult:
    m, x = _build_ortools(puzzle, model_name)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    t0 = time.perf_counter()
    st = solver.Solve(m)
    t1 = time.perf_counter() - t0
    stats = {"branches": solver.NumBranches(), "conflicts": solver.NumConflicts(),
             "wall_time": solver.WallTime(), "num_vars": len(m.Proto().variables),
             "num_constraints": len(m.Proto().constraints)}
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        name = "INFEASIBLE" if st == cp_model.INFEASIBLE else "TIMEOUT"
        return SolveResult(name, None, None, t1, 0.0, "ortools", model_name, stats)

    sol = [[0] * puzzle.cols for _ in range(puzzle.rows)]
    for (r, c), var in x.items():
        sol[r][c] = int(solver.Value(var))

    unique, t2 = None, 0.0
    if check_unique:
        # Restricciones semi-reificadas: diff_c => x_c != s_c, y al menos un diff_c verdadero.
        diffs = []
        for (r, c), var in x.items():
            d = m.NewBoolVar(f"diff_{r}_{c}")
            m.Add(var != sol[r][c]).OnlyEnforceIf(d)
            diffs.append(d)
        m.AddBoolOr(diffs)
        t0 = time.perf_counter()
        st2 = solver.Solve(m)
        t2 = time.perf_counter() - t0
        unique = st2 == cp_model.INFEASIBLE
    return SolveResult("FEASIBLE", sol, unique, t1, t2, "ortools", model_name, stats)


# ============================================================ motor propio
def _solve_native(puzzle, model_name, check_unique, time_limit) -> SolveResult:
    cells = puzzle.white_cells()
    idx = {c: i for i, c in enumerate(cells)}
    runs = puzzle.runs()
    cons = [([idx[c] for c in run.cells], run.total) for run in runs]
    init = [cp_engine.FULL] * len(cells)
    if model_name == "global_dom":
        for run in runs:
            mask = cp_engine.domain_mask(allowed_digits(len(run), run.total))
            for c in run.cells:
                init[idx[c]] &= mask
    # "global_dom" y "reified" usan GAC; "global" usa la propagación por cotas (más débil)
    level = "bounds" if model_name == "global" else "gac"
    eng = cp_engine.CPSolver(len(cells), cons, level=level)
    limit = 2 if check_unique else 1
    t0 = time.perf_counter()
    if any(d == 0 for d in init):
        sols = []
    else:
        sols = eng.solve(init, limit=limit, time_limit=time_limit)
    t = time.perf_counter() - t0
    stats = dict(eng.stats, num_vars=len(cells), num_constraints=len(cons), level=level)
    if not sols:
        status = "TIMEOUT" if eng.stats.get("timeout") else "INFEASIBLE"
        return SolveResult(status, None, None, t, 0.0, "native", model_name, stats)
    sol = [[0] * puzzle.cols for _ in range(puzzle.rows)]
    for c, v in zip(cells, sols[0]):
        sol[c[0]][c[1]] = v
    unique = (len(sols) == 1 and not eng.stats.get("timeout")) if check_unique else None
    return SolveResult("FEASIBLE", sol, unique, t, 0.0, "native", model_name, stats)
