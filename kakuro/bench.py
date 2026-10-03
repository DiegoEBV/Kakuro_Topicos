"""
bench.py — Análisis empírico del solver: tiempo y esfuerzo de búsqueda vs. tamaño.
"""
from __future__ import annotations

import time
from typing import Dict, List

from .generator import make_puzzle
from .solver import HAS_ORTOOLS, solve


def benchmark(sizes=(6, 8, 10, 12, 14, 16, 18, 20), per_size: int = 3, unique: bool = False,
              configs=None, time_limit: float = 60.0, seed: int = 7) -> List[Dict]:
    """Ejecuta cada configuración (backend, modelo) sobre puzzles generados de cada tamaño."""
    if configs is None:
        configs = [("native", "global"), ("native", "global_dom")]
        if HAS_ORTOOLS:
            configs += [("ortools", "global"), ("ortools", "global_dom"), ("ortools", "reified")]
    rows = []
    for n in sizes:
        for k in range(per_size):
            t0 = time.perf_counter()
            p = make_puzzle(n, n, seed=seed * 1000 + n * 10 + k, unique=unique)
            gen_s = time.perf_counter() - t0
            st = p.stats()
            for backend, model in configs:
                res = solve(p, backend=backend, model=model, check_unique=True, time_limit=time_limit)
                rows.append({"size": n, "instance": k, "white": st["white_cells"], "runs": st["runs"],
                             "backend": backend, "model": model, "status": res.status,
                             "correct": bool(res.solved and p.check_solution(res.solution)),
                             "unique": res.unique, "time_ms": 1000 * res.time_s,
                             "time_unique_ms": 1000 * res.time_unique_s,
                             "nodes": res.stats.get("nodes", res.stats.get("branches")),
                             "failures": res.stats.get("failures", res.stats.get("conflicts")),
                             "gen_s": gen_s})
    return rows
